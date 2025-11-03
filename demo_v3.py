import datetime
import os
import threading
import time
import concurrent.futures
import logging
from urllib import request

from ultralytics import YOLO
import cv2
import numpy as np
import pytz
from flask import Flask, request, jsonify

live_videos_flag = True
live_videos = []
video_readers_flag = True
video_readers = {}
video_streams_flag = True
video_streams = {}
alarm_dir = r"C:\Users\26601\Desktop\alarm_pic"
ori_dir = r"C:\Users\26601\Desktop\ori_pic"

os.makedirs(alarm_dir, exist_ok=True)
os.makedirs(ori_dir, exist_ok=True)
app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
batch_size = 3

model1 = YOLO(r"C:\Users\26601\Desktop\phone.pt")
model2 = YOLO("head.pt")


# video_streams是一个字典，用来存储视频流编号和流信息,key是该视频流的名称，值是一个VideoStream对象
class VideoStream:
    def __init__(self, video_stream_rtsp_url=None, video_link_time=None, video_desc=None,
                 video_source=None, conf=0.7, video_alarm_diff=300):
        self.video_stream_rtsp_url = video_stream_rtsp_url
        self.video_alarm_time = None
        self.video_link_time = video_link_time
        self.video_desc = video_desc
        self.video_source = video_source
        self.video_alarm_diff = video_alarm_diff
        self.conf = conf

    def set_init(self):
        self.video_stream_rtsp_url = None
        self.video_alarm_time = None
        self.video_link_time = None
        self.video_desc = None
        self.video_source = None
        self.conf = 0.7
        self.video_alarm_diff = 300


class FrameReader(threading.Thread):
    def __init__(self, name, rtsp_url, buffer_size=1):
        super().__init__(daemon=True)
        self.name = name
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.stopped = False
        self.status = None
        self.buffer_size = buffer_size

    def run(self):
        retry = 0
        logging.info(f"[{self.name}] 启动视频流读取线程")
        while not self.stopped:
            self.status = 1
            ok, frame = self.cap.read()
            if not ok:
                retry = retry + 1
                if retry > 3:
                    logging.warning(f"[{self.name}]  读取失败，退出读取...")
                    self.latest_frame = None
                    break
                logging.warning(f"[{self.name}] 第{retry}次 读取失败，尝试重连...")
                time.sleep(2)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                continue
            with self.frame_lock:
                self.latest_frame = frame
            self.status = 0
            time.sleep(1)  # 控制读取帧率

    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()
        logging.info(f"[{self.name}] 读取线程结束")


@app.route('/addVideo', methods=['POST'])
def add_video():
    global video_readers
    global video_streams
    global live_videos
    global video_readers_flag, video_streams_flag, live_videos_flag
    if request.json is None or request.json == {}:
        return "101，数据错误"
    rtsp_url = request.json['video_stream_rtsp_url']
    video = VideoStream(video_stream_rtsp_url=rtsp_url, video_alarm_diff=request.json['video_alarm_diff'])
    while video_streams_flag:
        video_streams_flag = False
        video_streams[video.video_stream_rtsp_url] = video
        video_streams_flag = True
        break

    while live_videos_flag:
        live_videos_flag = False
        live_videos.append(rtsp_url)
        live_videos_flag = True
        break

    reader = FrameReader(rtsp_url, rtsp_url=rtsp_url)
    while video_readers_flag:
        video_readers_flag = False
        video_readers[rtsp_url] = reader
        video_readers_flag = True
        break

    reader.start()

    return jsonify({"status": 200, "message": f"{rtsp_url}启动成功"})


@app.route('/deleteVideo', methods=['POST'])
def delete_video():
    global video_readers
    global video_streams
    global live_videos
    global video_readers_flag, video_streams_flag, live_videos_flag

    if request.json is None or request.json == {}:
        return "101，数据错误"
    rtsp_url = request.json['video_stream_rtsp_url']
    if rtsp_url not in live_videos:
        return jsonify({"status": 101, "message": f"{rtsp_url}不存在系统视频流库中"})
    reader = video_readers[rtsp_url]
    while reader.status == 0:
        reader.stop()
    while video_streams_flag:
        video_streams_flag = False
        video_streams.pop(rtsp_url)
        video_streams_flag = True
        break

    while video_readers_flag:
        video_readers_flag = False
        video_readers.pop(rtsp_url)
        video_readers_flag = True
        break

    while live_videos_flag:
        live_videos_flag = False
        live_videos.remove(rtsp_url)
        live_videos_flag = True
        break
    return jsonify({"status": 200, "message": f"{rtsp_url}删除成功"})


@app.route('/getVideos', methods=['GET'])
def get_videos():
    global live_videos
    return jsonify({"status": 200, "num": len(live_videos), "videos": live_videos})


@app.route('/stopVideos', methods=['GET'])
def stop_videos():
    global video_readers
    for reader in video_readers.values():
        while reader.status == 0:
            reader.stop()
    return jsonify({"status": 200})


@app.route('/startVideos', methods=['GET'])
def start_videos():
    global video_readers
    global live_videos
    for video in live_videos:
        if video not in video_readers:
            reader = FrameReader(video, video)
            video_readers[video] = reader
    for reader in video_readers.values():
        reader.start()
    return jsonify({"status": 200})


def get_now():
    return datetime.datetime.now(pytz.timezone('Asia/Shanghai'))


def detect_frame(frame, video_stream: VideoStream):
    alarm_flag = False
    alarm_time = video_stream.video_alarm_time
    alarm_diff = video_stream.video_alarm_diff
    rtsp_url = video_stream.video_stream_rtsp_url
    rtsp_conf = video_stream.conf
    logging.info(f"开始检测流{rtsp_url}")
    if frame is None:
        logging.warning(f"于{get_now()}读取{rtsp_url}的一帧失败")
        return
    if not isinstance(frame, np.ndarray):
        logging.warning(f'{rtsp_url}的frame不是numpy array类型')
        return

    if alarm_time:
        now = get_now()
        dt = pytz.timezone('Asia/Shanghai').localize(datetime.datetime.strptime(alarm_time, '%Y-%m-%d-%H-%M-%S'))
        time_diff = (now - dt).total_seconds()
        if time_diff < alarm_diff:
            logging.info(f"流{rtsp_url}距离上次报警不足{time_diff}秒，跳过检测")
            return
    r = model1.predict(frame)[0]
    boxes = r.boxes
    if len(boxes) == 0:
        now = get_now().strftime('%Y-%m-%d-%H-%M-%S')

        logging.info(f"{rtsp_url}在{get_now()}时读的一帧没有出现目标")
        return
    names = r.names
    img_copy = r.orig_img.copy()

    for id, box in enumerate(boxes):
        conf = box.conf.item()
        if conf > rtsp_conf:
            x1, y1, x2, y2 = [int(round(x)) for x in box.xyxy[0].cpu().numpy()]
            box_name = names[int(box.cls.item())]
            color = (0, 255, 0)  # 绿色边框
            thickness = 2
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

            # 绘制标签背景框
            label = f"phone:{conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_copy, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)

            # 绘制标签文字
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)

            if not alarm_flag:
                alarm_flag = True
    now = None
    if alarm_flag:
        now = get_now().strftime('%Y-%m-%d-%H-%M-%S')
        video_stream.video_alarm_time = now
        logging.info(f"{rtsp_url}在{now}时有报警信息")
    os.makedirs('alarm_pic', exist_ok=True)
    cv2.imwrite(r"C:\Users\26601\Desktop\ori_pic" + f"\\{now}.jpg", r.orig_img)
    cv2.imwrite(r"C:\Users\26601\Desktop\alarm_pic" + f"\\{now}.jpg", img_copy)


def detect_video_stream(video_streams, video_readers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            if len(live_videos) > 0:
                for key, reader in video_readers.items():
                    frame = reader.get_latest_frame()
                    if frame is not None:
                        executor.submit(detect_frame, frame, video_streams[key])
                    else:
                        logging.info(f'{key}读的一帧错误')
                    time.sleep(1)
                logging.info(f'一轮读取完成，等待3秒')
                time.sleep(3)
            else:
                logging.info('当前没有视频流，等待10秒')
                time.sleep(10)


if __name__ == '__main__':
    video_thread = threading.Thread(target=detect_video_stream, args=(video_streams, video_readers,), daemon=True)
    # video = VideoStream(video_stream_rtsp_url=r"D:\edgeDownload\33007863281-1-100024.mp4",
    #                     video_link_time=get_now().strftime('%Y-%m-%d-%H-%M-%S'), video_alarm_diff=8)
    video_thread.start()
    app.run(host='127.0.0.1', port=8080)

# 添加第二个视频流时会报错
# 2025-10-24 13:53:24 [INFO] [rtsp://admin:codvision122@192.168.201.122:554/Streaming/Channels/1] [rtsp://admin:codvision122@192.168.201.122:554/Streaming/Channels/1] 启动视频流读取线程
# 2025-10-24 13:53:24 [INFO] [Thread-5 (process_request_thread)] 127.0.0.1 - - [24/Oct/2025 13:53:24] "POST /addVideo HTTP/1.1" 200 -
# Exception in thread Thread-1 (detect_video_stream):
# Traceback (most recent call last):
#   File "D:\softWareCode\anaconda\envs\pytorch-study\Lib\threading.py", line 1073, in _bootstrap_inner
#     self.run()
#   File "D:\softWareCode\anaconda\envs\pytorch-study\Lib\threading.py", line 1010, in run
#     self._target(*self._args, **self._kwargs)
#   File "D:\projects\pythonProjects\pytorch-study\demo_v3.py", line 271, in detect_video_stream
#     for key, reader in video_readers.items():
# RuntimeError: dictionary changed size during iteration

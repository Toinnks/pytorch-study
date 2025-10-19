import datetime
import os
import threading
import time
import concurrent.futures
import logging
from ultralytics import YOLO
import cv2
import numpy as np
import pytz
import flask
app = flask.Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
batch_size = 3
model_path = "head.pt"
model = YOLO(model_path)


# video_streams是一个字典，用来存储视频流编号和流信息,key是该视频流的名称，值是一个VideoStream对象
class VideoStream:
    def __init__(self, video_stream_rtsp_url=None, video_link_time=None, video_desc=None,
                 video_source=None, conf=0.5, video_alarm_diff=300):
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
        self.conf = 0.5
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
        self.buffer_size = buffer_size

    def run(self):
        retry = 0
        logging.info(f"[{self.name}] 启动视频流读取线程")
        while not self.stopped:
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
            time.sleep(0.03)  # 控制读取帧率

    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()
        logging.info(f"[{self.name}] 读取线程结束")

def get_now():
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    return now


def detect_frame(frame, video_stream: VideoStream):
    alarm_flag = False
    alarm_time = video_stream.video_alarm_time
    alarm_diff = video_stream.video_alarm_diff
    rtsp_url = video_stream.video_stream_rtsp_url
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
    r = model.predict(frame)[0]
    boxes = r.boxes
    if len(boxes) == 0:
        return
        # logging.info(f"{rtsp_url}在{get_now()}时读的一帧没有出现目标")
    names = r.names
    img_copy = r.orig_img.copy()

    for id, box in enumerate(boxes):
        conf = box.conf.item()
        x1, y1, x2, y2 = [int(round(x)) for x in box.xyxy[0].cpu().numpy()]
        box_name = names[int(box.cls.item())]
        color = (0, 255, 0)  # 绿色边框
        thickness = 2
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

        # 绘制标签背景框
        label = f"head:{conf:.2f}"
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
    cv2.imwrite(f"alarm_pic\\{now}.jpg", img_copy)


def detect_video_stream(video_streams,video_readers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            for key, reader in video_readers.items():
                frame = reader.get_latest_frame()
                if frame is not None:
                    executor.submit(detect_frame, frame, video_streams[key])
                else:
                    logging.info('帧错误！！！')
            logging.info('等待10秒')
            time.sleep(10)


if __name__ == '__main__':
    video_readers = {}
    video_streams = {}
    video_thread = threading.Thread(target=detect_video_stream, args=(video_streams,video_readers,), daemon=True)
    video = VideoStream(video_stream_rtsp_url=r"D:\edgeDownload\33007863281-1-100024.mp4",
                        video_link_time=get_now().strftime('%Y-%m-%d-%H-%M-%S'),video_alarm_diff=8)
    # video.video_stream_rtsp_url='rtsp://rtspstream:abf3N_azEvzgsMF3TE224@zephyr.rtsp.stream/people'
    video_streams[1] = video
    reader = FrameReader("stream1", video.video_stream_rtsp_url)
    video_readers[1] = reader
    reader.start()
    video_thread.start()
    input("按回车键退出程序...\n")
    for reader in video_readers.values():
        reader.stop()

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
batch_size=3
model_path="head.pt"
model=YOLO(model_path)
video_streams={}
#video_streams是一个字典，用来存储视频流编号和流信息,key是该视频流的名称，值是一个VideoStream对象
class VideoStream:
    def __init__(self, video_stream_rtsp_url=None,video_alarm_time=None,video_link_time=None,video_desc=None,video_source=None,conf=0.5,video_alarm_diff=300):
        self.video_stream_rtsp_url = video_stream_rtsp_url
        self.video_alarm_time = video_alarm_time
        self.video_link_time = video_link_time
        self.video_desc = video_desc
        self.video_source = video_source
        self.video_alarm_diff=video_alarm_diff
        self.conf = conf
    def set_init(self):
        self.video_stream_rtsp_url = None
        self.video_alarm_time = None
        self.video_link_time = None
        self.video_desc = None
        self.video_source = None
        self.conf = 0.5
        self.video_alarm_diff = 300
def get_now():
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    return now

def detect_frame(frame,video_stream:VideoStream):

    alarm_flag=False
    alarm_time= video_stream.video_alarm_time
    alarm_diff = video_stream.video_alarm_diff
    rtsp_url=video_stream.video_stream_rtsp_url
    logging.info(f"开始检测流{rtsp_url}")
    if frame is None:
        logging.warning(f"于{get_now()}读取{rtsp_url}的一帧失败")
        return
    if not isinstance(frame,np.ndarray):
        logging.warning(f'{rtsp_url}的frame不是numpy array类型')
        return

    if alarm_time:
        now = get_now()
        dt = pytz.timezone('Asia/Shanghai').localize(datetime.datetime.strptime(alarm_time, '%Y-%m-%d-%H-%M-%S'))
        time_diff=(now-dt).total_seconds()
        if time_diff<alarm_diff:
            logging.info(f"流{rtsp_url}距离上次报警不足{time_diff}秒，跳过检测")
            return
    r=model.predict(frame)[0]
    boxes=r.boxes
    if len(boxes)==0:
        return
        # logging.info(f"{rtsp_url}在{get_now()}时读的一帧没有出现目标")
    names=r.names
    img_copy=r.orig_img.copy()

    for id ,box in enumerate(boxes):
        conf=box.conf.item()
        x1, y1, x2, y2 = [int(round(x)) for x in box.xyxy[0].cpu().numpy()]
        box_name=names[int(box.cls.item())]
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
            alarm_flag=True
    now=None
    if alarm_flag:
        now=get_now().strftime('%Y-%m-%d-%H-%M-%S')
        video_stream.video_alarm_time=now
        logging.info(f"{rtsp_url}在{now}时有报警信息")
    os.makedirs('alarm_pic',exist_ok=True)
    cv2.imwrite(f"alarm_pic\\{now}.jpg", img_copy)
            
            
def read_frame_from_rtsp(video_stream_rtsp_url,max_retries=3):
    opts = [
        "rtsp_transport;tcp",  # 用TCP更稳
        f"stimeout;{5000 * 1000}",  # 连接超时(微秒)
        f"timeout;{5000 * 1000}",  # 读超时(微秒)
    ]
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)

    for i in range(1, max_retries + 1):
        cap=None
        try:
            cap = cv2.VideoCapture(video_stream_rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                raise RuntimeError(f"{video_stream_rtsp_url}第{i}次打开失败")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"{video_stream_rtsp_url}第{i}次读失败")
            # 返回 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            logging.info(f"[{video_stream_rtsp_url}] 在第{i}次读成功")
            os.makedirs('read_pic', exist_ok=True)
            now=get_now().strftime('%Y-%m-%d-%H-%M-%S')
            cv2.imwrite(f"read_pic\\{now}.jpg", frame)
            return frame
        except Exception as e:
            print(f"[{video_stream_rtsp_url}] 第{i}次/{max_retries}失败: {e}")
            time.sleep(1.0 * i)  # 1s, 2s, 3s...
        finally:
            if cap is not None:
                cap.release()

    logging.error(f"[{video_stream_rtsp_url}]在{max_retries}次全失败")
    return None


def process_batch_video_stream(batch_video_streams_list):
    for item in batch_video_streams_list:
        temp=video_streams[item]
        if temp.video_stream_rtsp_url is None:
            continue
        #先读一帧
        frame=read_frame_from_rtsp(temp.video_stream_rtsp_url)
        if frame is None:
            logging.warning(f"于{get_now()}读取{item}的一帧失败")
        #对该帧进行处理
        detect_frame(frame,video_streams[item])




def detect_video_stream(video_streams):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            activate_video_streams_list = list(video_streams.keys())
            if activate_video_streams_list is None:
                logging.warning('未发现任何视频流信息')
                continue
            logging.info(f"发现了{len(activate_video_streams_list)}视频流")
            for i in range(0,len(activate_video_streams_list),batch_size):
                batch_video_streams_list = activate_video_streams_list[i:i+batch_size]
                executor.submit(process_batch_video_stream,batch_video_streams_list)
            logging.info('等待10秒')
            time.sleep(10)



if __name__ == '__main__':
    video_thread=threading.Thread(target=detect_video_stream,args=(video_streams,),daemon=True)
    video=VideoStream()
    # video.video_stream_rtsp_url='rtsp://rtspstream:abf3N_azEvzgsMF3TE224@zephyr.rtsp.stream/people'
    video.video_stream_rtsp_url=r"D:\edgeDownload\抖音-记录美好生活 (1).mp4"
    video.video_link_time=get_now()
    video.video_source="自己添加"
    video.video_alarm_diff=30
    video.video_desc='111'
    video_streams[1]=video
    video_thread.start()
    input("按回车键退出程序...\n")  # 阻塞主线程
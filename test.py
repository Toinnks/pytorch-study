# import datetime
# import time
# import pytz
# video_streams={}
# class VideoStream:
#     def __init__(self, video_stream_code=None,video_last_alarm=None,video_link_time=None,video_desc=None,video_source=None):
#         self.video_stream_code = video_stream_code
#         self.video_last_alarm = video_last_alarm
#         self.video_link_time = video_link_time
#         self.video_desc = video_desc
#         self.video_source = video_source
#     def set_none(self):
#         self.video_stream_code = None
#         self.video_last_alarm = None
#         self.video_link_time = None
#         self.video_desc = None
#         self.video_source = None
# now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
# print(now)
# video1=VideoStream(1,2,3,4)
# video2=VideoStream(3,4,5,6)
# video3=VideoStream(4,5,6,7)
# video4=VideoStream(5,6,7,8)
# video_streams={"1":video1,"2":video2,"3":video3,"4":video4}
# print(video_streams)
# my_video=video_streams["4"]
# del my_video
# del video_streams["4"]
# print(video_streams)
import cv2
# def read_frame_from_rtsp(video_stream_code,max_retries=3):
#     opts = [
#         "rtsp_transport;tcp",  # 用TCP更稳
#         f"stimeout;{5000 * 1000}",  # 连接超时(微秒)
#         f"timeout;{5000 * 1000}",  # 读超时(微秒)
#     ]
#     os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)
#
#     for i in range(1, max_retries + 1):
#         cap=None
#         try:
#             cap = cv2.VideoCapture(video_stream_code, cv2.CAP_FFMPEG)
#             if not cap.isOpened():
#                 raise RuntimeError(f"{video_stream_code}第{i}次打开失败")
#             ok, frame = cap.read()
#             if not ok or frame is None:
#                 raise RuntimeError(f"{video_stream_code}第{i}次读失败")
#             # 返回 RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             logging.info(f"[{video_stream_code}] 在第{i}次读成功")
#             return frame
#         except Exception as e:
#             print(f"[{video_stream_code}] 第{i}次/{max_retries}失败: {e}")
#             time.sleep(1.0 * i)  # 1s, 2s, 3s...
#         finally:
#             if cap is not None:
#                 cap.release()
#
#     logging.error(f"[{video_stream_code}]在{max_retries}次全失败")
#     return None
# if __name__ == '__main__':
#     rtsp_url="rtsp://rtspstream:abf3N_azEvzgsMF3TE224@zephyr.rtsp.stream/people"
#     frame=read_frame_from_rtsp(rtsp_url)
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite("snapshot.jpg", frame)
#     cv2.imwrite("snapshot——gray.jpg", gray_frame)


from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt


def analyze_boxes_structure():
    model = YOLO('head.pt')
    results = model('https://ultralytics.com/images/bus.jpg')
    r = results[0]
    boxes = r.boxes

    print(f"boxes 类型: {type(boxes)}")
    print(f"boxes 是否可迭代: {hasattr(boxes, '__iter__')}")
    print(f"boxes 长度: {len(boxes)}")

    # 检查主要属性
    print(f"\nboxes 的主要属性:")
    print(f"- cls: {boxes.cls}")  # 类别ID
    print(f"- conf: {boxes.conf}")  # 置信度
    print(f"- xyxy: {boxes.xyxy}")  # 边界框坐标 [x1, y1, x2, y2]
    print(f"- data: {boxes.data}")  # 完整数据

    # 虽然 boxes 不是列表，但可以迭代
    print(f"\n迭代 boxes:")
    for i, box in enumerate(boxes):
        print(f"box {i} 类型: {type(box)}")
        print(f"box {i} 数据: {box}")
        break  # 只看第一个


def explain_box_formats():
    model = YOLO('head.pt')
    results = model('read_pic/2025-10-11-19-58-54.jpg')
    r = results[0]
    orig_img = r.orig_img.copy()
    names=r.names
    boxes = r.boxes
    print(f"boxes的类型是{type(boxes)}")
    print(f"boxes的长度是{len(boxes)}")
    for i, box in enumerate(boxes):
        box_name=names[int(box.cls[0].cpu().numpy())]
        conf=box.conf.item()
        print(f"这是第{i}个预测框,该框的类型是{box_name}")
        print(f"conf是{conf}")
        x1,y1,x2,y2 = [int(round(x)) for x in box.xyxy[0].cpu().numpy().tolist()]
        print(f"xyxy是{box.xyxy},x1是{x1},y1是{y1},x2是{x2},y2是{y2}")

        color = (0, 255, 0)  # 绿色边框
        thickness = 2
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, thickness)

        # 绘制标签背景框
        label = f"{box_name} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(orig_img, (x1, y1 - text_h - baseline), (x1 + text_w, y1), (45, 54, 59), -1)

        # 绘制标签文字
        cv2.putText(orig_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite('custom_plot_result.jpg', orig_img)


explain_box_formats()

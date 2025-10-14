import datetime

import cv2
import pytz


#
# # rtsp='rtsp://rtspstream:abf3N_azEvzgsMF3TE224@zephyr.rtsp.stream/people'
# rtsp=r"D:\edgeDownload\抖音-记录美好生活.mp4"
# cap=cv2.VideoCapture(rtsp)
# if not cap.isOpened():
#     raise RuntimeError(f"{rtsp}次打开失败")
# ok, frame = cap.read()
# if not ok:
#     print("帧错误")
# cv2.imwrite('frame.jpg',frame)
def get_now():
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    return now
time_str = "2025-10-11-19-58-54"
naive = datetime.datetime.strptime(time_str, '%Y-%m-%d-%H-%M-%S')
dt=pytz.timezone('Asia/Shanghai').localize(naive)
temp=(get_now()-dt).total_seconds()
print(temp)

if head_area <= phone_area:

    print(f"人头框面积{head_area} <= 手机框面积{phone_area}，跳过")
    continue
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
import cv2

# 读取图像
frame = cv2.imread(r"D:\projects\pythonProjects\pytorch-study\bus.jpg")

# 检查图像是否加载成功
if frame is None:
    print("❌ 图像加载失败，请检查路径是否正确")
    exit()

# 获取图像高度和宽度
h, w, _ = frame.shape  # 正确解包 shape

# 将图像分割为四个子块，并记录每个块在原图中的左上角坐标
crops_data = [
    (frame[0:h//2, 0:w//2], (0, 0)),
    (frame[0:h//2, w//2:w], (w//2, 0)),
    (frame[h//2:h, 0:w//2], (0, h//2)),
    (frame[h//2:h, w//2:w], (w//2, h//2))
]

# 遍历每个子块并显示
for i, (image, position) in enumerate(crops_data):
    window_name = f'Crop {i+1} - Position {position}'
    cv2.imshow(window_name, image)

# 等待按键退出（按任意键关闭所有窗口）
cv2.waitKey(0)
cv2.destroyAllWindows()
from flask import Flask, jsonify, request
import requests
import base64
import datetime
import pytz
import time
import threading
import numpy as np
from ultralytics import YOLO
import concurrent.futures
from ffmpeg_live import ffmpeg_live
import subprocess
import json
import warnings
import cv2
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

# YOLOv8 模型
smoke_model = YOLO('./best.pt')
person_model = YOLO('./person.pt')

send_url1 = "http://172.16.252.138/api/slalarm/add"
send_url2 = "http://172.16.252.137:8090/prod-api/maquan/event/add"  # 目标地址
conf = 0.7  # 置信度阈值
polling_interval = 5  # 轮询间隔（秒）
batch_size = 10  # 每轮处理的视频流数量
num_threads = 4  # 线程池大小
video_streams = {}  # 维护所有的 RTSP 直播流信息

def read_frame_from_rtsp(rtsp_url):
    """使用 OpenCV 读取 RTSP 视频流的一帧"""
    try:
        # 使用 OpenCV 打开 RTSP 流
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            print(f"[{rtsp_url}] 无法连接到 RTSP 流")
            return None

        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print(f"[{rtsp_url}] 读取帧失败")
            cap.release()
            return None

        cap.release()  # 读取完毕后释放 VideoCapture 对象

        # 确保返回的帧是正确的
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"[{rtsp_url}]")
        return frame
    except Exception as e:
        print(f"[{rtsp_url}] 读取 RTSP 流时发生错误: {e}")
        return None


def _expand_box(x1, y1, x2, y2, img_w, img_h, expand_ratio=0.3):
    """按比例向四周扩张边界框，并裁剪到图像范围内（坐标均为像素）"""
    w = x2 - x1
    h = y2 - y1
    dx = w * expand_ratio
    dy = h * expand_ratio
    nx1 = max(0, int(x1 - dx))
    ny1 = max(0, int(y1 - dy))
    nx2 = min(img_w - 1, int(x2 + dx))
    ny2 = min(img_h - 1, int(y2 + dy))
    return nx1, ny1, nx2, ny2


def _has_intersection(a, b):
    """两框是否相交（a,b 皆为 x1,y1,x2,y2）"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return inter_x2 > inter_x1 and inter_y2 > inter_y1


def detect_frame(stream_name, rtsp_url, frame, conf):
    """先检测烟；若有烟，再检测烟附近是否有人；满足条件才报警"""
    global video_streams

    if frame is None or not isinstance(frame, np.ndarray):
        print(f"[{stream_name}] 读取到的 frame 无效")
        return

    # 5分钟防抖
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    last_alarm_time = video_streams.get(stream_name, {}).get("last_alarm_time", None)
    if last_alarm_time and (now - last_alarm_time).total_seconds() < 300:
        print(f"[{stream_name}] 距离上次报警不足5分钟，跳过报警")
        return

    # OpenCV 用 BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame.copy()
    H, W = frame_bgr.shape[:2]

    # ---------------------------
    # 阶段1：只跑“烟”检测
    # ---------------------------
    try:
        smoke_results = smoke_model.predict(frame_bgr, conf=conf, verbose=False, save=False)
    except Exception as e:
        print(f"[{stream_name}] 烟检测异常: {e}")
        return

    smoke_boxes = []
    for r in smoke_results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for det, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if s >= conf:
                x1, y1, x2, y2 = map(int, det)
                smoke_boxes.append({"box": (x1, y1, x2, y2), "cls": int(c), "score": float(s)})

    if len(smoke_boxes) == 0:
        print(f"[{stream_name}] 未检测到烟，结束本帧")
        return

    # ---------------------------
    # 阶段2：有烟，再跑“人”检测（给人一个单独阈值）
    # ---------------------------
    person_conf = 0.7  # 你可按需求调整
    try:
        person_results = person_model.predict(frame_bgr, conf=person_conf, verbose=False, save=False)
    except Exception as e:
        print(f"[{stream_name}] 人检测异常: {e}")
        return

    person_boxes = []
    for r in person_results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for det, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if s >= person_conf:
                x1, y1, x2, y2 = map(int, det)
                person_boxes.append({"box": (x1, y1, x2, y2), "cls": int(c), "score": float(s)})

    if len(person_boxes) == 0:
        print(f"[{stream_name}] 检出烟，但附近无人")
        return

    # ---------------------------
    # 关联：只统计“附近有人”的烟
    # ---------------------------
    NEARBY_EXPAND_RATIO = 0.30  # 把烟框按比例放大判断“附近”
    smoke_with_people = []      # [(smoke, [persons...]), ...]
    for s in smoke_boxes:
        sx1, sy1, sx2, sy2 = s["box"]
        ex1, ey1, ex2, ey2 = _expand_box(sx1, sy1, sx2, sy2, W, H, expand_ratio=NEARBY_EXPAND_RATIO)
        expanded = (ex1, ey1, ex2, ey2)

        near_persons = []
        for p in person_boxes:
            if _has_intersection(expanded, p["box"]):
                near_persons.append(p)

        if len(near_persons) > 0:
            smoke_with_people.append((s, near_persons))

    if len(smoke_with_people) == 0:
        print(f"[{stream_name}] 检出烟，但附近无人（按扩张比 {NEARBY_EXPAND_RATIO:.2f} 判定）")
        return

    # ---------------------------
    # 可视化：红色画烟框，绿色画人框，并标注置信度
    # ---------------------------
    for s, persons in smoke_with_people:
        x1, y1, x2, y2 = s["box"]
        label = f"smoke {s['score']:.2f}"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色(BGR)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_bgr, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 255), -1)
        cv2.putText(frame_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for p in persons:
            px1, py1, px2, py2 = p["box"]
            plabel = f"person {p['score']:.2f}"
            cv2.rectangle(frame_bgr, (px1, py1), (px2, py2), (0, 255, 0), 2)  # 绿色(BGR)
            (pw, ph), _ = cv2.getTextSize(plabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_bgr, (px1, py1 - ph - 10), (px1 + pw, py1), (0, 255, 0), -1)
            cv2.putText(frame_bgr, plabel, (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ---------------------------
    # 触发告警（按“烟附近有人”的烟框数量计数）
    # ---------------------------
    event_count = len(smoke_with_people)

    # 生成报警时间和文件名
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    alarm_time = now.strftime('%Y-%m-%d_%H-%M-%S')
    alarm_filename = f"{stream_name}-{alarm_time}-smoking.jpg"
    alarm_filepath = f"/smoking/alarmpic/{alarm_filename}"

    video_filename = f"{stream_name}-{alarm_time}-smoking.mp4"
    video_output_file = f"/smoking/alarmvideo/{video_filename}"

    # 确保目录存在（磁盘存 ./alarmpic、./alarmvideo；上报用 /smoking/... 路径）
    os.makedirs("./alarmvideo", exist_ok=True)
    os.makedirs("./alarmpic", exist_ok=True)
    cv2.imwrite(f"./alarmpic/{alarm_filename}", frame_bgr)

    payload = {
        "alarmName": "smoking",
        "cameraCode": stream_name,
        "alarmTime": now.strftime('%Y-%m-%d %H:%M:%S'),
        "alarmPic": alarm_filepath,
        "alarmVideo": video_output_file
    }

    send_alarm(payload, send_url1, send_url2, stream_name, event_count)
    save_alarm_video(video_filename, rtsp_url)

    # 更新上次报警时间
    video_streams[stream_name]["last_alarm_time"] = now

def save_alarm_video(video_filename, rtsp_url):
    """使用 FFmpeg 在后台线程中录制 5 秒报警视频"""
    video_output_file = f'./alarmvideo/{video_filename}'
    command = [
        "ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", rtsp_url,
        "-t", "5", "-c:v", "copy", video_output_file
    ]

    def run_ffmpeg():
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 以后台线程执行 FFmpeg 录制，避免阻塞主线程
    threading.Thread(target=run_ffmpeg, daemon=True).start()

def send_alarm(payload, primary_url, secondary_url, stream_name, event_count):
    """发送报警信息到两个接口（带二次重试机制）"""
    try:
        response = requests.post(primary_url, json=payload)
        print(f"[{stream_name}] 预警发送状态: {response.status_code}")
        if response.status_code == 200:
            print(f"[{stream_name}] 预警发送成功")
            try:
                response_data = response.json()
                event_id = response_data.get("data", {}).get("id")
                imageUrl = response_data.get("data", {}).get("alarmPic")
                videoUrl = response_data.get("data", {}).get("alarmVideo")
                if event_id:
                    ret = {
                        "eventId": event_id,
                        "eventType": payload["alarmName"],
                        "deviceId": stream_name,
                        "eventTime": payload["alarmTime"],
                        "eventCount": event_count,
                        "imageUrl": imageUrl,
                        "alarmVideo": videoUrl,
                        "videoUrl": videoUrl
                    }
                    print(f"[{stream_name}] 发送给 send_url2 的数据:\n", json.dumps(ret, indent=2, ensure_ascii=False), flush=True)
                    # 第一次发送
                    res = requests.post(secondary_url, json=ret)
                    if res.status_code == 200:
                        print(f"[{secondary_url}] 发送成功: {res.status_code}")
                        print(res.json().get("code"))
                        print(res.json().get("msg"))
                    else:
                        print(f"[{secondary_url}] 第一次发送失败，状态码: {res.status_code}，尝试重试...")
                        # 第二次重试
                        res_retry = requests.post(secondary_url, json=ret)
                        if res_retry.status_code == 200:
                            print(f"[{secondary_url}] 重试成功: {res_retry.status_code}")
                            print(res_retry.json().get("code"))
                            print(res_retry.json().get("msg"))
                        else:
                            print(f"[{secondary_url}] 重试仍然失败: {res_retry.status_code}")
            except ValueError:
                print("响应不是有效的 JSON")
        else:
            print(f"[{stream_name}] 预警失败: {response.status_code}")
    except requests.RequestException as e:
        print(f"[{stream_name}] 预警发送异常: {e}")


def process_batch(stream_batch, video_streams):
    """批量处理一组视频流"""
    for stream_name in stream_batch:
        input_stream = video_streams.get(stream_name)
        if not input_stream:
            continue

        conf = input_stream.get("conf", 0.7)

        # 读取 RTSP 视频帧
        frame = read_frame_from_rtsp(input_stream["input_stream"])
        if frame is None:
            print(f"[{stream_name}] 读取视频帧失败")
            continue

        detect_frame(stream_name, input_stream["input_stream"], frame, conf)


def process_video_streams(video_streams):
    """轮询所有视频流，按批次分组处理"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        while True:
            active_streams = list(video_streams.keys())
            if not active_streams:
                time.sleep(1)
                continue

            print(f"[轮询] 发现 {len(active_streams)} 个视频流，按批次处理...")

            for i in range(0, len(active_streams), batch_size):
                stream_batch = active_streams[i:i + batch_size]
                executor.submit(process_batch, stream_batch, video_streams)

            time.sleep(polling_interval)


@app.route('/streams/status', methods=['POST'])
def get_video_stream_status():
    """查询正在检测的视频流"""
    if not video_streams:
        return jsonify({"message": "没有正在检测的视频流"}), 200

    active_streams = list(video_streams.keys())  # 获取所有正在检测的视频流的名称
    return jsonify({"active_video_streams": active_streams}), 200


@app.route('/streams/add', methods=['POST'])
def add_streams():
    """添加多个视频流"""
    data = request.json
    streams = data.get('streams', [])
    conf = data.get('conf', 0.7)

    if not streams or not isinstance(streams, list) or not isinstance(conf, float):
        return jsonify({"error": "Invalid streams format"}), 400

    for stream in streams:
        stream_name = stream.get('stream_name')
        input_stream = stream.get('input_stream')

        if not stream_name or not input_stream:
            continue

        if stream_name in video_streams:
            continue

        video_streams[stream_name] = {
            "input_stream": input_stream,
            "conf": conf
        }

    return jsonify({"status": "Streams added", "streams": list(video_streams.keys())}), 200


@app.route('/streams/delete', methods=['POST'])
def delete_streams():
    """删除视频流"""
    data = request.json
    stream_names = data.get('stream_names', [])

    if not stream_names or not isinstance(stream_names, list):
        return jsonify({"error": "Invalid stream_names format"}), 400

    for stream_name in stream_names:
        if stream_name in video_streams:
            del video_streams[stream_name]

    return jsonify({"status": "Streams deleted"}), 200


# 维护所有推流任务的字典
current_stream = None
stream_threads = {}
stream_controls = {}


@app.route('/live/start', methods=['POST'])
def start_stream():
    global current_stream  # 使用全局变量来控制流
    success_add_lst = []
    data = request.get_json()
    streams = data.get('streams')
    
    if not streams or not isinstance(streams, list):
        return jsonify({"error": "Invalid streams format, must be a list"}), 400
    
    for stream in streams:
        stream_name = stream.get("stream_name")
        input_stream = stream.get("input_stream")

        if not stream_name or not input_stream:
            return jsonify({"error": "缺少 stream_name 或 input_stream"}), 400

        # 如果已经有流在推送，停止当前流
        if current_stream and current_stream != stream_name:
            # 停止当前流的推流
            print(f"Stopping current stream {current_stream}")
            stream_controls[current_stream]["is_live_stream"] = False
            stream_threads[current_stream].join()  # 等待当前线程结束
            current_stream = None  # 清除当前流
        
        # 控制变量，动态启停推流
        stream_controls[stream_name] = {"is_live_stream": True}

        # 创建并启动推流线程
        thread = threading.Thread(target=ffmpeg_live, args=(model, stream_name, input_stream, stream_controls), daemon=True)
        thread.start()
        
        # 更新当前正在运行的流
        current_stream = stream_name
        stream_threads[stream_name] = thread
        success_add_lst.append(stream_name)

    return jsonify({"message": f"推流{str('、'.join(success_add_lst))}已启动"}), 200


@app.route('/live/stop', methods=['POST'])
def stop_stream():
    success_delete_lst = []
    error_lst = []
    data = request.json
    stream_names = data.get('stream_names')  # 接收一个流名称列表

    if not stream_names or not isinstance(stream_names, list):
        return jsonify({"error": "Invalid stream_names format, must be a list"}), 400

    for stream_name in stream_names:
        if not stream_name or stream_name not in stream_threads or stream_name not in stream_controls:
            error_lst.append(stream_name)
            continue

        # 关闭推流 & 检测
        stream_controls[stream_name]["is_live_stream"] = False
        time.sleep(2)

        # 停止推流线程
        stream_threads[stream_name].join()  # 等待线程完成
        del stream_threads[stream_name]
        del stream_controls[stream_name]
        success_delete_lst.append(stream_name)
    if len(error_lst) == 0:
        return jsonify({"message": f"推流 {str('、'.join(success_delete_lst))} 已停止"}), 200
    elif len(success_delete_lst) == 0:
        return jsonify({"error": f"流 {str('、'.join(error_lst))} 不在推流"}), 200
    else:
        return jsonify({"message": f"推流 {str('、'.join(success_delete_lst))} 已停止",
                        "error": f"流 {str('、'.join(error_lst))} 不在推流"}), 200


if __name__ == '__main__':
    video_streams = {}

    # 启动视频流处理线程
    video_thread = threading.Thread(target=process_video_streams, args=(video_streams,), daemon=True)
    video_thread.start()

    # 启动 Flask 服务器
    app.run(host='0.0.0.0', port=1002,debug=False)

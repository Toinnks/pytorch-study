from flask import Flask, jsonify, request
import cv2
import base64
import requests
import datetime
import pytz
import time
import threading
from ultralytics import YOLO
from ffmpeg_live import ffmpeg_live
import concurrent.futures
import ffmpeg
import numpy as np
import subprocess
import re
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# 加载模型
model = YOLO("./best2.pt")
send_url1 = "http://172.16.252.138/api/slalarm/add"
send_url2 = "http://172.16.252.137:8090/prod-api/maquan/event/add"  # 目标地址
conf = 0.7  # 置信度阈值
polling_interval = 2  # 轮询间隔（秒）
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
        return frame
    except Exception as e:
        print(f"[{rtsp_url}] 读取 RTSP 流时发生错误: {e}")
        return None

def detect_frame(stream_name, rtsp_url, frame, conf):
    """检测未戴安全帽或未穿安全服（但不是都缺）并报警"""
    global video_streams
    if frame is None or not isinstance(frame, np.ndarray):
        print(f"[{stream_name}] 读取到的 frame 无效")
        return

    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    last_alarm_time = video_streams.get(stream_name, {}).get("last_alarm_time", None)
    if last_alarm_time and (now - last_alarm_time).total_seconds() < 300:
        print(f"[{stream_name}] 距离上次报警不足5分钟，跳过报警")
        return

    frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model.predict(frame, save=False)
    no_safety_count = 0

    for m in results:
        boxes = m.boxes
        xyxy = boxes.xyxy.tolist()
        cls = boxes.cls.tolist()
        confs = boxes.conf.tolist()

        persons, helmets, self_clothes, safety_clothes, heads = [], [], [], [], []
        for det, c, confs in zip(xyxy, cls, confs):
            if confs < conf:
                continue
            c = int(c)
            if c == 0:
                persons.append(det)
            elif c == 1:
                helmets.append(det)
            elif c == 2:
                self_clothes.append(det)
            elif c == 3:
                safety_clothes.append(det)
            elif c == 4:
                heads.append(det)

        def is_inside(inner, outer):
            xi1, yi1, xi2, yi2 = inner
            xo1, yo1, xo2, yo2 = outer
            return xi1 >= xo1 and yi1 >= yo1 and xi2 <= xo2 and yi2 <= yo2

        for person_box in persons:
            has_helmet = any(is_inside(h, person_box) for h in helmets)
            has_self_clothes = any(is_inside(s, person_box) for s in self_clothes)
            has_safety_clothes = any(is_inside(s, person_box) for s in safety_clothes)
            has_head = any(is_inside(hd, person_box) for hd in heads)

            # ✅ 新报警逻辑
            if (has_helmet and has_self_clothes) or (has_head and has_safety_clothes):
                x1, y1, x2, y2 = map(int, person_box)
                label = "Missing Vest" if has_helmet else "Missing Helmet"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                no_safety_count += 1

    if no_safety_count > 0:
        # 保存图像/视频
        alarm_time_str = now.strftime('%Y-%m-%d_%H-%M-%S')
        alarm_filename = f"{stream_name}-{alarm_time_str}-no_helmet_or_vest.jpg"
        alarm_filepath = f"/no_helmet_or_vest/alarmpic/{alarm_filename}"
        video_filename = f"{stream_name}-{alarm_time_str}-no_helmet_or_vest.mp4"
        video_output_file = f"/no_helmet_or_vest/alarmvideo/{video_filename}"

        os.makedirs("./alarmpic", exist_ok=True)
        os.makedirs("./alarmvideo", exist_ok=True)
        cv2.imwrite(f"./alarmpic/{alarm_filename}", frame)

        payload = {
            "alarmName": "no_helmet_or_vest",
            "cameraCode": stream_name,
            "alarmTime": now.strftime('%Y-%m-%d %H:%M:%S'),
            "alarmPic": alarm_filepath,
            "alarmVideo": video_output_file
        }

        send_alarm(payload, send_url1, send_url2, stream_name, no_safety_count)
        save_alarm_video(video_filename, rtsp_url)

        # 更新报警时间
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
    """发送报警信息到两个接口"""
    try:
        header = {"Content-Type":"application/json"}
        response = requests.post(primary_url, json=payload, headers=header)
        print(f"[{stream_name}] 预警发送状态: {response.status_code}")
        if response.status_code == 200:
            print(f"[{stream_name}] 预警发送成功")
            try:
                response_data = response.json()
                print(response_data)
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
                        "videoUrl": videoUrl
                    }
                    res = requests.post(secondary_url, json=ret)
                    if res.status_code == 200:
                        print(f"[{secondary_url}] 发送成功: {res.status_code}")
                        print(res.json().get("code"))
                        print(res.json().get("msg"))
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

            # 分批次处理
            for i in range(0, len(active_streams), batch_size):
                stream_batch = active_streams[i:i+batch_size]
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
stream_threads = {}
stream_controls = {}

@app.route('/live/start', methods=['POST'])
def start_stream():
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

        if stream_name in stream_threads:
            return jsonify({"error": f"流 {stream_name} 已在推流"}), 400

        # 控制变量，动态启停推流
        stream_controls[stream_name] = {"is_live_stream": True}

        # 创建并启动推流线程
        thread = threading.Thread(target=ffmpeg_live, args=(model, stream_name, input_stream, stream_controls), daemon=True)
        thread.start()
        
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
        return jsonify({"message": f"推流 {str('、'.join(success_delete_lst))} 已停止","error": f"流 {str('、'.join(error_lst))} 不在推流"}), 200

if __name__ == '__main__':
    video_streams = {}

    # 启动视频流处理线程
    video_thread = threading.Thread(target=process_video_streams, args=(video_streams,), daemon=True)
    video_thread.start()

    # 启动 Flask 服务器
    app.run(host='0.0.0.0', port=1006, debug=False)

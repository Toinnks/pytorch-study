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
import json
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# 加载模型
model = YOLO("./best2.pt")
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
        return frame
    except Exception as e:
        print(f"[{rtsp_url}] 读取 RTSP 流时发生错误: {e}")
        return None

def detect_frame(stream_name, rtsp_url, frame, conf):
    """目标检测并发送报警"""
    global video_streams
    # 确保 frame 不是 None 且格式正确
    if frame is None or not isinstance(frame, np.ndarray):
        print(f"[{stream_name}] 读取到的 frame 无效")
        return

    # 读取当前时间
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    # 检查上次报警时间
    last_alarm_time = video_streams.get(stream_name, {}).get("last_alarm_time", None)
    if last_alarm_time:
        time_diff = (now - last_alarm_time).total_seconds()
        if time_diff < 300:  # 5分钟 = 300秒
            print(f"[{stream_name}] 距离上次报警不足5分钟，跳过报警")
            return
    
    # 确保frame是可写的
    frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 使用 model 检测
    results = model.predict(frame, save=False, classes=[4])
    no_safety_count = 0  # 统计不合规人数
    for m in results:
        # 获取每个boxes的结果
        box = m.boxes
        # 获取预测的类别
        cls = box.cls
        # 获取置信度
        cf = box.conf
        # 检查是否有置信度大于等于阈值的目标
        has_high_conf = any(s >= conf for s in cf)
        if has_high_conf:
            # 绘制边界框和类别名及置信度
            for det, c, s in zip(box.xyxy.tolist(), cls.tolist(), cf.tolist()):
                if s >= conf:
                    x1, y1, x2, y2 = det
                    label = f"{model.names[int(c)]} {s:.2f}"
                    # 绘制边界框
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    # 获取文本大小
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    # 绘制背景填充
                    cv2.rectangle(frame, (int(x1), int(y1) - h - 10), (int(x1) + w, int(y1)), (255, 0, 0), -1)
                    # 绘制文本
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    no_safety_count += 1
            # 生成报警时间和文件名
            now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
            alarm_time = now.strftime('%Y-%m-%d_%H-%M-%S')

            alarm_filename = f"{stream_name}-{alarm_time}-no_helmet.jpg"
            alarm_filepath = f"/no_helmet/alarmpic/{alarm_filename}"

            video_filename = f"{stream_name}-{alarm_time}-no_helmet.mp4"
            video_output_file = f"/no_helmet/alarmvideo/{video_filename}"

            # 确保目录存在
            os.makedirs("./alarmvideo", exist_ok=True)
            os.makedirs("./alarmpic", exist_ok=True)
            cv2.imwrite(f"./alarmpic/{alarm_filename}", frame)

            # 发送报警
            payload = {
                "alarmName": "no_helmet",
                "cameraCode": stream_name,
                "alarmTime": now.strftime('%Y-%m-%d %H:%M:%S'),
                "alarmPic": alarm_filepath,  # 这里替换为URL
                "alarmVideo": video_output_file
            }
            save_alarm_video(video_filename, rtsp_url)
            send_alarm(payload, send_url1, send_url2, stream_name, no_safety_count)
            

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
    global current_stream
    success_delete_lst = []
    error_lst = []
    data = request.json
    stream_names = data.get('stream_names')  # 接收一个流名称列表

    if not stream_names or not isinstance(stream_names, list):
        return jsonify({"error": "Invalid stream_names format, must be a list"}), 400
    
    for stream_name in stream_names:
        # 基本校验
        if (not stream_name) or (stream_name not in stream_threads) or (stream_name not in stream_controls):
            error_lst.append(stream_name)
            continue

        # 发停止信号（不断点删结构，防止竞态）
        try:
            stream_controls[stream_name]["is_live_stream"] = False
        except Exception:
            # 极端并发下也兜底为失败
            error_lst.append(stream_name)
            continue

        # 等线程退出，最多等 10 秒（比 sleep 更可靠）
        t = stream_threads.get(stream_name)
        try:
            t.join(timeout=10)
        except Exception:
            error_lst.append(stream_name)
            continue

        # 线程未如期退出，判为失败（不清理字典，避免悬空）
        if t.is_alive():
            error_lst.append(stream_name)
            continue

        # 线程已退出，安全清理
        try:
            del stream_threads[stream_name]
        except KeyError:
            pass
        try:
            del stream_controls[stream_name]
        except KeyError:
            pass

        # 若恰好是当前流，置空
        if current_stream == stream_name:
            current_stream = None

        success_delete_lst.append(stream_name)

    if len(error_lst) == 0:
        return jsonify({"message": f"推流{str('、'.join(success_delete_lst))}已停止"}), 200
    elif len(success_delete_lst) == 0:
        return jsonify({"error": f"流{str('、'.join(error_lst))}不在推流"}), 200
    else:
        return jsonify({"message": f"推流{str('、'.join(success_delete_lst))}已停止", "error": f"流{str('、'.join(error_lst))}不在推流"}), 200


if __name__ == '__main__':
    video_streams = {}

    # 启动视频流处理线程
    video_thread = threading.Thread(target=process_video_streams, args=(video_streams,), daemon=True)
    video_thread.start()

    # 启动 Flask 服务器
    app.run(host='0.0.0.0', port=1001, debug=False)

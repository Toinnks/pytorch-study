from flask import Flask, jsonify, request
import cv2
import base64
import requests
import datetime
import pytz
import time
import threading
from ultralytics import YOLO
# from ffmpeg_live import ffmpeg_live
import concurrent.futures
import ffmpeg
import numpy as np
import subprocess
import re
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# 加载吃东西检测模型
model = YOLO("./best.pt")
send_url1 = "http://10.0.101.1:9062/slalarm/add"
# send_url2 = "http://172.16.252.137:8090/prod-api/maquan/event/add" 
conf = 0.6  # 置信度阈值
polling_interval = 5  # 轮询间隔（秒）
batch_size = 10  # 每轮处理的视频流数量
num_threads = 4  # 线程池大小
video_streams = {}  # 维护所有的 RTSP 直播流信息

def read_frame_from_rtsp(rtsp_url):
    """使用 OpenCV 读取 RTSP 视频流的一帧"""
    try:
        # 使用 OpenCV 打开 RTSP 流
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000"
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
    """吃东西检测并发送报警"""
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

    results = model.predict(frame, save=False, conf=conf,classes=[0,1])
    eating_count = 0  # 统计检测到的吃东西行为数量

    stream_info = video_streams.get(stream_name, {})
    stream_source = stream_info.get("stream_source", "")
    stream_vehicleCode = stream_info.get("stream_vehicleCode", "")
    stream_vehicleOid = stream_info.get("stream_vehicleOid", "")
    stream_vehiclePlateNo = stream_info.get("stream_vehiclePlateNo", "")
    
    for m in results:
        # 获取每个boxes的结果
        box = m.boxes
        if box is None or len(box) == 0:
            continue
            
        # 获取预测的类别
        cls = box.cls
        # 获取置信度
        cf = box.conf
        # 检查是否有置信度大于等于阈值的目标
        # any判断是否有至少一个为 True
        has_high_conf = any(s >= conf for s in cf)
        
        if has_high_conf:
            print(f"[{stream_name}] 检测到吃东西行为，置信度: {[f'{s:.2f}' for s in cf if s >= conf]}")
            
            # 绘制边界框和类别名及置信度
            for det, c, s in zip(box.xyxy.tolist(), cls.tolist(), cf.tolist()):
                if s >= conf:
                    x1, y1, x2, y2 = det
                    class_name = model.names.get(int(c), f"class_{int(c)}")
                    label = f"{class_name} {s:.2f}"
                    
                    # 绘制边界框 - 使用红色表示检测到吃东西
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    # 获取文本大小
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    # 绘制背景填充
                    cv2.rectangle(frame, (int(x1), int(y1) - h - 10), (int(x1) + w, int(y1)), (0, 0, 255), -1)
                    # 绘制文本
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 添加吃东西标识
                    cv2.putText(frame, "EATING DETECTED", (int(x1), int(y2) + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    eating_count += 1
            
            # 生成报警时间和文件名
            now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
            alarm_time = now.strftime('%Y-%m-%d_%H-%M-%S')

            alarm_filename = f"{stream_name}-{alarm_time}-eating.jpg"
            alarm_pic_path = f"/data/clearingvehicle/pic_vid/eating/alarmpic/{alarm_filename}"

            video_filename = f"{stream_name}-{alarm_time}-eating.mp4"
            video_output_file = f"/data/clearingvehicle/pic_vid/eating/alarmvideo/{video_filename}"

            # 确保目录存在
            os.makedirs("/data/clearingvehicle/pic_vid/eating/alarmpic", exist_ok=True)
            os.makedirs("/data/clearingvehicle/pic_vid/eating/alarmvideo", exist_ok=True)
            
            # 保存报警图片
            cv2.imwrite(alarm_pic_path, frame)
            print(f"[{stream_name}] 保存报警图片: {alarm_pic_path}")
            
            # 发送报警
            payload = {
                "alarmName": "eating",
                "alarmType": "eating",
                "targetCode": stream_name,
                "alarmTime": now.strftime('%Y-%m-%d %H:%M:%S'),
                "alarmPic": f"/eating/alarmpic/{alarm_filename}",
                "alarmVideo": f"/eating/alarmvideo/{video_filename}",
                "source": stream_source,
                "cameraCode": stream_vehicleCode,
                "oid": stream_vehicleOid,
                "alarmCode": stream_vehiclePlateNo
            }

            print(f"[{stream_name}] 检测到 {eating_count} 个吃东西行为，触发报警")
            print(f"-----------------------完整payload----------------------: {payload}")
            send_alarm(payload, send_url1, stream_name, eating_count)
            save_alarm_video(video_filename, rtsp_url)

            # 更新上次报警时间
            video_streams[stream_name]["last_alarm_time"] = now

def save_alarm_video(video_filename, rtsp_url):
    """使用 FFmpeg 在后台线程中录制 6 秒报警视频"""
    video_output_file = f'/data/clearingvehicle/pic_vid/eating/alarmvideo/{video_filename}'
    command = [
        "ffmpeg", "-y",
        "-i", rtsp_url,
        "-t", "6",  # 录制6秒
        "-vf", "scale=704:576",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-an",                      
        "-movflags", "+faststart",
        video_output_file
    ]
    
    def run_ffmpeg():
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            print(f"[{video_filename}] 报警视频录制完成")
        except subprocess.TimeoutExpired:
            print(f"[{video_filename}] FFmpeg录制超时")
        except Exception as e:
            print(f"[{video_filename}] FFmpeg录制失败: {e}")
    
    # 以后台线程执行 FFmpeg 录制，避免阻塞主线程
    threading.Thread(target=run_ffmpeg, daemon=True).start()

def send_alarm(payload, primary_url, stream_name, event_count):
    """发送吃东西报警信息到接口"""
    try:
        header = {"Content-Type": "application/json"}
        response = requests.post(primary_url, json=payload, headers=header, timeout=10)
        print(f"[{stream_name}] 吃东西报警发送状态: {response.status_code}")
        
        if response.status_code == 200:
            print(f"[{stream_name}]  吃东西报警发送成功")
            try:
                response_data = response.json()
                print(f"[{stream_name}] 服务器响应: {response_data}")
            except ValueError:
                print("响应不是有效的 JSON")
        else:
            print(f"[{stream_name}] 吃东西报警发送失败: {response.status_code}")
    except requests.RequestException as e:
        print(f"[{stream_name}] 吃东西报警发送异常: {e}")

def process_batch(stream_batch, video_streams):
    """批量处理一组视频流"""
    for stream_name in stream_batch:
        input_stream = video_streams.get(stream_name)
        if not input_stream:
            continue

        conf = input_stream.get("conf", 0.6)

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

            print(f"[轮询] 发现 {len(active_streams)} 个视频流，进行吃东西检测...")

            # 分批次处理
            for i in range(0, len(active_streams), batch_size):
                stream_batch = active_streams[i:i+batch_size]
                executor.submit(process_batch, stream_batch, video_streams)

            time.sleep(polling_interval)

# ==================== API 路由 ====================

# @app.route('/health', methods=['GET'])
# def health_check():
#     """健康检查"""
#     return jsonify({
#         "status": "healthy",
#         "service": "eating_detection",
#         "model_loaded": model is not None,
#         "active_streams": len(video_streams),
#         "detection_type": "eating_behavior"
#     })

@app.route('/streams/status', methods=['POST'])
def get_video_stream_status():
    """查询正在检测的视频流"""
    if not video_streams:
        return jsonify({"message": "没有正在进行吃东西检测的视频流"}), 200
    
    active_streams = list(video_streams.keys())
    streams_detail = {}
    
    for stream_name, info in video_streams.items():
        streams_detail[stream_name] = {
            "input_stream": info.get("input_stream"),
            "conf": info.get("conf"),
            "last_alarm_time": info.get("last_alarm_time").strftime('%Y-%m-%d %H:%M:%S') if info.get("last_alarm_time") else None
        }
    
    return jsonify({
        "message": f"共有 {len(active_streams)} 个视频流正在进行吃东西检测",
        "active_eating_detection_streams": active_streams,
        "streams_detail": streams_detail
    }), 200

@app.route('/streams/add', methods=['POST'])
def add_streams():
    """添加多个吃东西检测视频流"""
    print("/streams/add 接口被调用")
    data = request.json
    print(f"接收到的数据: {data} ")
    streams = data.get('streams', [])
    conf = data.get('conf', 0.6)

    if not streams or not isinstance(streams, list) or not isinstance(conf, float):
        return jsonify({"error": "Invalid streams format"}), 400

    added_streams = []
    
    for stream in streams:
        stream_name = stream.get('stream_name')
        input_stream = stream.get('input_stream')
        stream_source = stream.get('stream_source')   # 车来源 hik、rm
        stream_vehicleCode = stream.get('stream_vehicleCode')  # 车辆编码
        stream_vehicleOid = stream.get('stream_vehicleOid') # 车辆oid
        stream_vehiclePlateNo = stream.get('stream_vehiclePlateNo')  #车牌号
        

        if not stream_name or not input_stream:
            continue

        if stream_name in video_streams:
            print(f"[{stream_name}] 吃东西检测流已存在，跳过")
            continue

        video_streams[stream_name] = {
            "input_stream": input_stream,
            "conf": conf,
            "last_alarm_time": None,
            "stream_source": stream_source,  # 存储车来源 hik、rm
            "stream_vehicleCode": stream_vehicleCode,  # 存储车辆编码
            "stream_vehicleOid":stream_vehicleOid,   # 车辆oid
            "stream_vehiclePlateNo":stream_vehiclePlateNo    #车牌号
        }
        added_streams.append(stream_name)
        print(f"[{stream_name}] 添加吃东西检测流成功")

    return jsonify({
        "status": "Eating detection streams added", 
        "added_streams": added_streams,
        "total_streams": list(video_streams.keys())
    }), 200

@app.route('/streams/delete', methods=['POST'])
def delete_streams():
    """删除吃东西检测视频流"""
    data = request.json
    stream_names = data.get('stream_names', [])

    if not stream_names or not isinstance(stream_names, list):
        return jsonify({"error": "Invalid stream_names format"}), 400

    deleted_streams = []
    
    for stream_name in stream_names:
        if stream_name in video_streams:
            del video_streams[stream_name]
            deleted_streams.append(stream_name)
            print(f"[{stream_name}] 删除吃东西检测流成功")

    return jsonify({
        "status": "Eating detection streams deleted",
        "deleted_streams": deleted_streams,
        "remaining_streams": list(video_streams.keys())
    }), 200

# 维护所有推流任务的字典
stream_threads = {}
stream_controls = {}

@app.route('/live/start', methods=['POST'])
def start_stream():
    """启动吃东西检测推流"""
    success_add_lst = []
    data = request.get_json()
    streams = data.get('streams')
    
    if not streams or not isinstance(streams, list):
        return jsonify({"error": "Invalid streams format, must be a list"}), 400
    
    for stream in streams:
        stream_name = stream.get("stream_name")
        input_stream = stream.get("input_stream")
        stream_source = stream.get('stream_source')   # 车来源 hik、rm
        stream_vehicleCode = stream.get('stream_vehicleCode')  # 车辆编码
        stream_vehicleOid = stream.get('stream_vehicleOid') # 车辆oid
        stream_vehiclePlateNo = stream.get('stream_vehiclePlateNo')  #车牌号

        if not stream_name or not input_stream:
            return jsonify({"error": "缺少 stream_name 或 input_stream"}), 400

        if stream_name in stream_threads:
            return jsonify({"error": f"吃东西检测推流 {stream_name} 已在运行"}), 400

        # 控制变量，动态启停推流
        stream_controls[stream_name] = {
            "is_live_stream": True,
            "stream_source": stream_source,  # 存储车来源 hik、rm
            "stream_vehicleCode": stream_vehicleCode,  # 存储车辆编码
            "stream_vehicleOid":stream_vehicleOid,   # 车辆oid
            "stream_vehiclePlateNo":stream_vehiclePlateNo    #车牌号
            }

        # 创建并启动推流线程
        thread = threading.Thread(
            target=ffmpeg_live, 
            args=(model, stream_name, input_stream, stream_controls), 
            daemon=True
        )
        thread.start()
        
        stream_threads[stream_name] = thread
        success_add_lst.append(stream_name)
        print(f"[{stream_name}]  吃东西检测推流已启动")

    return jsonify({
        "message": f"吃东西检测推流 {str('、'.join(success_add_lst))} 已启动"
    }), 200

@app.route('/live/stop', methods=['POST'])
def stop_stream():
    """停止吃东西检测推流"""
    success_delete_lst = []
    error_lst = []
    data = request.json
    stream_names = data.get('stream_names')

    if not stream_names or not isinstance(stream_names, list):
        return jsonify({"error": "Invalid stream_names format, must be a list"}), 400
    
    for stream_name in stream_names:
        if not stream_name or stream_name not in stream_threads or stream_name not in stream_controls:
            error_lst.append(stream_name)
            continue

        # 关闭推流
        stream_controls[stream_name]["is_live_stream"] = False
        time.sleep(2)

        # 停止推流线程
        stream_threads[stream_name].join(timeout=5)
        del stream_threads[stream_name]
        del stream_controls[stream_name]
        success_delete_lst.append(stream_name)
        print(f"[{stream_name}] 吃东西检测推流已停止")
        
    if len(error_lst) == 0:
        return jsonify({
            "message": f"吃东西检测推流 {str('、'.join(success_delete_lst))} 已停止"
        }), 200
    elif len(success_delete_lst) == 0:
        return jsonify({
            "error": f"流 {str('、'.join(error_lst))} 不在推流"
        }), 200
    else:
        return jsonify({
            "message": f"吃东西检测推流 {str('、'.join(success_delete_lst))} 已停止",
            "error": f"流 {str('、'.join(error_lst))} 不在推流"
        }), 200


if __name__ == '__main__':
    
    os.makedirs("/data/clearingvehicle/pic_vid/eating/alarmpic", exist_ok=True)
    os.makedirs("/data/clearingvehicle/pic_vid/eating/alarmvideo", exist_ok=True)
    print("报警文件目录创建完成")
    
    video_streams = {}

    # 启动视频流处理线程
    video_thread = threading.Thread(target=process_video_streams, args=(video_streams,), daemon=True)
    video_thread.start()
    print("视频检测线程启动完成")

    # 启动 Flask 服务器
    print("Flask服务器启动中...")
    app.run(host='0.0.0.0', port=1225, debug=False)
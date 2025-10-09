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
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

app = Flask(__name__)

# 加载模型
phone_model = YOLO("./phone.pt")
head_model = YOLO("./head.pt")
send_url1 = "http://192.168.101.94:81/externalApi/api/push-dangerous-driving"
send_url2 = "http://192.168.101.35:9062/slalarm/add"
conf = 0.6  # 置信度阈值
polling_interval = 5  # 轮询间隔（秒）
batch_size = 10  # 每轮处理的视频流数量
num_threads = 4  # 线程池大小
video_streams = {}  # 维护所有的 RTSP 直播流信息


def read_frame_from_rtsp(rtsp_url, max_retries=3, open_timeout_ms=5000, read_timeout_ms=5000, backoff=1.0):
    """读取 RTSP 单帧，最多重连 max_retries 次；成功返回 RGB 帧，失败返回 None"""
    import os, time
    # 仅对 FFMPEG 后端生效的超时/传输参数
    opts = [
        "rtsp_transport;tcp",                          # 用TCP更稳
        f"stimeout;{open_timeout_ms*1000}",            # 连接超时(微秒)
        f"timeout;{read_timeout_ms*1000}",             # 读超时(微秒)
    ]
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)

    for i in range(1, max_retries + 1):
        cap = None
        try:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                raise RuntimeError("open failed")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("read failed")
            # 返回 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"[{rtsp_url}] read ok on attempt {i}")
            return frame
        except Exception as e:
            print(f"[{rtsp_url}] attempt {i}/{max_retries} failed: {e}")
            time.sleep(backoff * i)  # 1s, 2s, 3s...
        finally:
            if cap is not None:
                cap.release()

    print(f"[{rtsp_url}] all {max_retries} attempts failed.")
    return None

def detect_frame(stream_name, rtsp_url, frame, conf):
    """目标检测并发送报警"""

    global video_streams
    
    # 0) 基础检查
    if frame is None or not isinstance(frame, np.ndarray):
        print(f"[{stream_name}] 读取到的 frame 无效", flush=True)
        return

    # 1) 冷却期
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    last_alarm_time = video_streams.get(stream_name, {}).get("last_alarm_time", None)
    if last_alarm_time:
        time_diff = (now - last_alarm_time).total_seconds()
        if time_diff < 300:  # 5分钟 = 300秒
            print(f"[{stream_name}] 距离上次报警不足5分钟，跳过报警（{int(time_diff)}s）", flush=True)
            return

    # 2) 颜色空间
    # 如果你的frame来自cv2.VideoCapture，原本就是BGR；你这里用的是RGB->BGR转换。
    # 如果frame原来就是BGR，这一步会导致颜色错乱，但不至于崩。保留你的写法：
    frame_bgr = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)

    # 3) 推理（Ultralytics 风格）
    results_head = head_model.predict(frame_bgr, conf=conf, verbose=True)[0]
    results_phone = phone_model.predict(frame_bgr, conf=conf, verbose=True)[0]

    # 提取 boxes（xyxy）—— 如果是 Ultralytics YOLO:
    # 也可根据你模型返回结构调整
    heads = []
    phones = []
    if hasattr(results_head, "boxes") and results_head.boxes is not None:
        # 转成 python list of tuples [(x1,y1,x2,y2), ...]
        heads = results_head.boxes.xyxy.cpu().numpy().tolist()
    if hasattr(results_phone, "boxes") and results_phone.boxes is not None:
        phones = results_phone.boxes.xyxy.cpu().numpy().tolist()

    print(f"[{stream_name}] heads={len(heads)}, phones={len(phones)}", flush=True)

    # 4) 读取流信息（可选）
    stream_info = video_streams.get(stream_name, {})
    stream_source = stream_info.get("stream_source", "")
    stream_vehicleCode = stream_info.get("stream_vehicleCode", "")
    stream_vehicleOid = stream_info.get("stream_vehicleOid", "")
    stream_vehiclePlateNo = stream_info.get("stream_vehiclePlateNo", "")
    stream_vehicleOrgName = stream_info.get("stream_vehicleOrgName", "")

    # 5) 判定逻辑
    calling_heads = []

    for idx_h, head_box in enumerate(heads):
        hx1, hy1, hx2, hy2 = head_box
        head_w = hx2 - hx1
        head_h = hy2 - hy1

        head_area = head_w * head_h

        ry1 = hy1 - head_h * 0.1
        ry2 = hy2 + head_h * 0.2

        rx1 = hx2 - head_w * 0.2
        rx2 = hx2 + head_w * 0.6

        lx1 = hx1 - head_w * 0.6
        lx2 = hx1 + head_w * 0.2
        ly1 = hy1 + head_h * 0.2
        ly2 = hy2 + head_h * 0.4


        for phone_box in phones:
            px1, py1, px2, py2 = phone_box
            phone_w = px2 - px1
            phone_h = py2 - py1
            phone_area = phone_w * phone_h

            if head_area <= phone_area:
                print(f"人头框 <= 手机框，跳过")
                continue
            cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0

            # 叠加比例
            overlap_ratio = box_overlap_area(
                (px1, py1, px2, py2), (hx1, hy1, hx2, hy2)
            ) / (max((px2 - px1) * (py2 - py1), 1e-6))

            in_ear = (
                ((rx1 <= cx <= rx2 and ry1 <= cy <= ry2) or
                    (lx1 <= cx <= lx2 and ly1 <= cy <= ly2)) and
                overlap_ratio > 0.2 and
                cy < hy2
            )


            in_face = is_phone_in_head(
                (px1, py1, px2, py2), (hx1, hy1, hx2, hy2)
            ) and cy < hy2 + head_h * 0.2

            if in_ear or in_face:
                cv2.rectangle(frame_bgr, (int(px1), int(py1)), (int(px2), int(py2)),(0, 0, 255), 2)  # 红色手机框
                cv2.putText(frame_bgr, "PHONE", (int(px1), max(0, int(py1) - 5)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                calling_heads.append(idx_h)
                break


    # 7) 无报警则返回
    if not calling_heads:
        # 用 rtsp_url 或 stream_name，不要用未定义的 stream_url
        if 'logger' in globals() and logger:
            logger.info(f"[{stream_name}] 未检测到打电话行为，跳过")
        else:
            print(f"[{stream_name}] 未检测到打电话行为，跳过", flush=True)
        return
    # 生成报警文件名
    
    alarm_time = now.strftime('%Y-%m-%d_%H-%M-%S')
    alarm_filename = f"{stream_name}-{alarm_time}-phone.jpg"
    alarm_filepath = f"/data/clearingvehicle/pic_vid/phone/alarmpic/{alarm_filename}"
    video_filename = f"{stream_name}-{alarm_time}-phone.mp4"
    video_output_file = f"/data/clearingvehicle/pic_vid/phone/alarmvideo/{video_filename}"

    # 确保目录存在
    os.makedirs(os.path.dirname(alarm_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(video_output_file), exist_ok=True)

    # 保存图片并生成报警
    cv2.imwrite(alarm_filepath, frame_bgr)
    
    # 创建报警数据
    payload = {
        "alarmName": "phone",
        "alarmType": "phone",
        "targetCode": stream_name,
        "alarmTime": now.strftime('%Y-%m-%d %H:%M:%S'),
        "alarmPic": f"http://192.168.101.35:9002/clearing_alarm/phone/alarmpic/{alarm_filename}",
        "alarmVideo": f"http://192.168.101.35:9002/clearing_alarm/phone/alarmvideo/{video_filename}",
        "source": stream_source,
        "cameraCode": stream_vehicleCode,
        "oid": stream_vehicleOid,
        "alarmCode": stream_vehiclePlateNo,
        "stream_vehicleOrgName":stream_vehicleOrgName
    }
    print(f"-----------------------完整payload----------------------: {payload}")
    send_alarm(payload, [send_url1, send_url2], stream_name, 1)
    save_alarm_video(video_filename, rtsp_url)

    # 更新报警时间
    video_streams[stream_name]["last_alarm_time"] = now
    



def is_phone_in_head(phone_box, head_box, threshold=0.4):
    """检查手机是否在头部区域内"""
    phone_area = (phone_box[2] - phone_box[0]) * (phone_box[3] - phone_box[1])
    inter_area = box_overlap_area(phone_box, head_box)
    return (phone_area > 0) and (inter_area / phone_area > threshold)


def box_overlap_area(a, b):
    """计算两个框的重叠区域"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0
    return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

def save_alarm_video(video_filename, rtsp_url):
    """使用 FFmpeg 在后台线程中录制 5 秒报警视频"""
    video_output_file = f'/data/clearingvehicle/pic_vid/phone/alarmvideo/{video_filename}'
    command = [
        "ffmpeg", "-y",
        "-i", rtsp_url,
        "-t", "6",
        "-vf", "scale=704:576",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-an",                      
        "-movflags", "+faststart",
        video_output_file
    ]
    def run_ffmpeg():
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 以后台线程执行 FFmpeg 录制，避免阻塞主线程
    threading.Thread(target=run_ffmpeg, daemon=True).start()


def send_alarm(payload, urls, stream_name, event_count, timeout=10, retries=1, max_workers=2):
    """
    将同一份 payload 同时发送到多个接口（urls 列表）。
    - 完全不更改 payload（两个平台收到的内容一模一样）
    - 每个接口失败可重试 retries 次
    - 并行发送以减少阻塞
    - 日志与返回结果中加入中文状态
    """
    headers = {"Content-Type": "application/json"}

    def post_with_retry(url):
        last_err = None
        for attempt in range(retries + 1):
            try:
                print(f"[{stream_name}] -> 正在向 {url} 发送报警 (第 {attempt+1}/{retries+1} 次尝试)")
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                status = resp.status_code
                if 200 <= status < 300:
                    try:
                        data = resp.json()
                    except ValueError:
                        data = None
                    print(f"[{stream_name}] ✅ 成功发送至 {url} (状态码 {status}) 响应: {data}")
                    return {"ok": True, "状态": "成功", "status": status, "resp": data, "错误": None}
                else:
                    txt = resp.text[:200]
                    print(f"[{stream_name}] ❌ 发送至 {url} 失败 (状态码 {status}) 返回: {txt}")
                    last_err = f"状态码={status}, 返回={txt}"
            except requests.RequestException as e:
                print(f"[{stream_name}] ⚠️ 向 {url} 发送时发生异常: {e}")
                last_err = str(e)
        return {"ok": False, "状态": "失败", "status": None, "resp": None, "错误": last_err}

    results = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(urls))) as ex:
        futs = {ex.submit(post_with_retry, u): u for u in urls}
        for fut in as_completed(futs):
            u = futs[fut]
            try:
                results[u] = fut.result()
            except Exception as e:
                results[u] = {"ok": False, "状态": "异常", "status": None, "resp": None, "错误": str(e)}
    return results



def process_batch(stream_batch, video_streams):
    """批量处理一组视频流"""
    for stream_name in stream_batch:
        input_stream = video_streams.get(stream_name)
        if not input_stream:
            continue

        conf = input_stream.get("conf", 0.5)

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
    conf = data.get('conf', 0.5)

    if not streams or not isinstance(streams, list) or not isinstance(conf, float):
        return jsonify({"error": "Invalid streams format"}), 400

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
            continue

        video_streams[stream_name] = {
            "input_stream": input_stream,
            "conf": conf,
            "stream_source": stream_source,  # 存储车来源 hik、rm
            "stream_vehicleCode": stream_vehicleCode,  # 存储车辆编码
            "stream_vehicleOid":stream_vehicleOid,   # 车辆oid
            "stream_vehiclePlateNo":stream_vehiclePlateNo    #车牌号
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
        stream_source = stream.get('stream_source')   # 车来源 hik、rm
        stream_vehicleCode = stream.get('stream_vehicleCode')  # 车辆编码
        stream_vehicleOid = stream.get('stream_vehicleOid') # 车辆oid
        stream_vehiclePlateNo = stream.get('stream_vehiclePlateNo')  #车牌号
        stream_vehicleOrgName = stream.get('stream_vehicleOrgName') #车辆所属公

        if not stream_name or not input_stream:
            return jsonify({"error": "缺少 stream_name 或 input_stream"}), 400

        if stream_name in stream_threads:
            return jsonify({"error": f"流 {stream_name} 已在推流"}), 400

        # 控制变量，动态启停推流
        stream_controls[stream_name] = {
            "is_live_stream": True,
            "stream_source": stream_source,  # 存储车来源 hik、rm
            "stream_vehicleCode": stream_vehicleCode,  # 存储车辆编码
            "stream_vehicleOid":stream_vehicleOid,   # 车辆oid
            "stream_vehiclePlateNo":stream_vehiclePlateNo,   #车牌号
            "stream_vehicleOrgName":stream_vehicleOrgName #所属公司
            }

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
    app.run(host='0.0.0.0', port=1222, debug=False)

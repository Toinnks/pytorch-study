import cv2
import subprocess
import datetime
import pytz

conf = 0.5
output_fps = 15

# 全局状态缓存
stream_state_cache = {}

# 设置无安全带持续时间阈值（秒）
no_belt_threshold_sec = 5

def ffmpeg_live(model, stream_name, input_stream, stream_controls):
    """实时处理视频流，连续5秒未检测到安全带才绘红框标记并推流"""
    global stream_state_cache

    cap = cv2.VideoCapture(input_stream)
    if not cap.isOpened():
        print(f"[{stream_name}] ❌ 无法打开视频流")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg_proc = None

    # 初始化状态
    stream_state_cache[stream_name] = {
        "last_detect_time": datetime.datetime.now(pytz.timezone('Asia/Shanghai')),
        "alarm_active": False
    }

    while cap.isOpened():
        if not stream_controls.get(stream_name, {}).get("is_live_stream", True):
            print(f"[{stream_name}] 停止推流")
            break

        ret, frame = cap.read()
        if not ret:
            print(f"[{stream_name}] 视频帧读取失败，退出")
            break

        now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))

        # 启动推流
        if ffmpeg_proc is None:
            print(f"[{stream_name}] 启动 FFmpeg 推流进程")
            ffmpeg_command = [
                'ffmpeg',
                '-re',
                '-loglevel', 'error',
                '-f', 'rawvideo',
                '-pixel_format', 'bgr24',
                '-video_size', f'{width}x{height}',
                '-framerate', str(output_fps),
                '-i', 'pipe:0',
                '-c:v', 'libx264',
                '-preset', 'veryfast',
                '-tune', 'zerolatency',
                '-crf', '25',
                '-g', '50',
                '-f', 'flv',
                f"rtmp://10.0.4.29:1935/hls/{stream_name}"
            ]
            try:
                ffmpeg_proc = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[{stream_name}] ❌ FFmpeg 启动失败: {e}")
                break

        # 检测安全带（class=4）
        detected = False
        results = model.predict(frame, save=False, classes=[4]) if model else []

        for result in results:
            if not hasattr(result, "boxes"):
                continue
            boxes = result.boxes
            cls = boxes.cls
            cf = boxes.conf
            for det, c, s in zip(boxes.xyxy.tolist(), cls.tolist(), cf.tolist()):
                if s >= conf:
                    detected = True  # 本帧检测到安全带
                    x1, y1, x2, y2 = map(int, det)
                    label = f"{model.names[int(c)]} {s:.2f}"
                    #  即使检测到了，也画绿框辅助展示（非红框警示）
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 获取该流的检测状态
        state = stream_state_cache[stream_name]

        if detected:
            state["last_detect_time"] = now
            state["alarm_active"] = False
        else:
            elapsed = (now - state["last_detect_time"]).total_seconds()
            if elapsed >= no_belt_threshold_sec:
                if not state["alarm_active"]:
                    print(f"[ALERT] 🚨 {stream_name} 连续 {elapsed:.1f}s 未检测到安全带，标记红框")
                    state["alarm_active"] = True
                # 显示红框提示（整个画面提示）
                cv2.putText(frame, "No Seat Belt Detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 推流输出
        try:
            ffmpeg_proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            print(f"[{stream_name}] ❌ FFmpeg 推流断开")
            break

    # 清理资源
    cap.release()
    if ffmpeg_proc:
        try:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        except:
            pass
    print(f"[{stream_name}] ✅ 推流进程已结束")

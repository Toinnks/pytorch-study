import cv2
import subprocess

conf = 0.5
output_fps = 15


def ffmpeg_live(model, stream_name, input_stream, stream_controls):
    """实时处理视频流，支持推流并停止推流"""
    cap = cv2.VideoCapture(input_stream)
    if not cap.isOpened():
        print(f"[{stream_name}] 无法打开视频流: {input_stream}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg_proc = None

    while cap.isOpened():
         # 检查推流标志，若停止推流，则终止循环
        if not stream_controls.get(stream_name, {}).get("is_live_stream", True):
            print(f"[{stream_name}] 停止推流，退出检测")
            break

        ret, frame = cap.read()
        if not ret:
            print(f"{stream_name} 视频帧读取失败，结束当前处理")
            break

        # 开启推流时，启动 FFmpeg 进程（只启动一次）
        if ffmpeg_proc is None:
            print(f"[{stream_name}] 启动 FFmpeg 推流进程")
            ffmpeg_command = [
                'ffmpeg',
                '-re',
                '-loglevel', 'error',  # 减少日志噪声
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
                print(f"[{stream_name}] FFmpeg 启动失败: {e}")
                break  # 如果推流失败，不继续循环

        # 处理目标检测（如果开启）
        if stream_controls[stream_name]["is_live_stream"]:
            results = model.predict(frame, save=False,classes=[0]) if model else []
            if results:  # 先检查 results 是否为空
                for result in results:
                    if hasattr(result, "boxes"):  # 确保 result.boxes 存在
                        box = result.boxes
                        cls = box.cls
                        cf = box.conf
                        has_high_conf = any(s >= conf for s in cf)
                        if has_high_conf:
                            eating_detected = True
                            for det, c, s in zip(box.xyxy.tolist(), cls.tolist(), cf.tolist()):
                                if s >= conf:
                                    x1, y1, x2, y2 = det
                                    label = f"{model.names[int(c)]} {s:.2f}"
                                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                    cv2.rectangle(frame, (int(x1), int(y1) - h - 10), (int(x1) + w, int(y1)), (0, 0, 255), -1)
                                    cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    eating_count += 1

        # 发送帧到 FFmpeg 推流
        try:
            ffmpeg_proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            print(f"[{stream_name}] FFmpeg 进程已关闭")
            break

    # 释放资源
    cap.release()
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    print(f"[{stream_name}] 进程完全结束")

"""
优化的多路RTSP视频流检测系统
整合FFmpeg高性能流处理和Flask API服务（修复版）
"""

import subprocess
import threading
import queue
import time
import datetime
import os
import signal
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import logging

import numpy as np
import cv2
import pytz
from flask import Flask, request, jsonify
from ultralytics import YOLO

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)-12s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============ 配置类 ============
@dataclass
class CameraConfig:
    """摄像头配置"""
    camera_id: str
    rtsp_url: str
    fps: int = 25
    sample_rate: int = 5  # 每秒采样帧数
    reconnect_interval: int = 5
    width: int = 1920
    height: int = 1080
    conf_threshold: float = 0.5  # 检测置信度阈值
    alarm_interval: int = 300  # 报警间隔(秒)
    last_alarm_time: Optional[datetime.datetime] = None
    enabled: bool = True
    # internal fields
    _failure_count: int = 0

    def can_alarm(self) -> bool:
        """检查是否可以报警"""
        if self.last_alarm_time is None:
            return True
        now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
        diff = (now - self.last_alarm_time).total_seconds()
        return diff >= self.alarm_interval

    def update_alarm_time(self):
        """更新最后报警时间"""
        self.last_alarm_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))


# ============ FFmpeg帧读取器 ============
class FFmpegFrameReader:
    """基于FFmpeg的高性能帧读取器，带 stderr drain，稳健读流"""

    def __init__(self, config: CameraConfig, use_hw_decode: bool = False):
        self.config = config
        self.use_hw_decode = use_hw_decode
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.last_frame_time = 0.0
        self.frame_interval = 1.0 / max(1, config.sample_rate)
        self.consecutive_failures = 0
        self.max_failures = 10
        self._stderr_thread: Optional[threading.Thread] = None
        self._stop_stderr = threading.Event()
        self._first_frame_received = False

    def _build_ffmpeg_command(self) -> List[str]:
        """构建FFmpeg命令"""
        cmd = ['ffmpeg']

        # 减少日志输出，但保留错误信息
        cmd.extend(['-hide_banner', '-loglevel', 'error'])

        # RTSP优化参数
        cmd.extend([
            '-rtsp_transport', 'tcp',
            '-stimeout', '5000000',  # 5秒超时 (microseconds)
            '-max_delay', '500000',   # 最大延迟 0.5秒
        ])

        # 输入源
        cmd.extend(['-i', self.config.rtsp_url])

        # 视频流处理：限制帧率和分辨率
        cmd.extend([
            '-r', str(self.config.sample_rate),  # 输出帧率
            '-s', f'{self.config.width}x{self.config.height}',  # 输出分辨率
        ])

        # 输出格式：原始 BGR24
        cmd.extend([
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-an',  # 禁用音频
            '-'  # 输出到 stdout
        ])

        return cmd

    def start(self) -> bool:
        """启动FFmpeg进程"""
        if self.is_running:
            return True
        try:
            cmd = self._build_ffmpeg_command()
            logger.info(f"Camera {self.config.camera_id} 启动命令: {' '.join(cmd)}")

            # 启动进程 - Windows 下需要特殊处理
            # 使用 CREATE_NO_WINDOW 标志避免弹出控制台窗口
            startupinfo = None
            creationflags = 0

            if os.name == 'nt':  # Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                creationflags = subprocess.CREATE_NO_WINDOW

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,  # 创建 stdin 管道但不使用
                bufsize=10 ** 8,  # 100MB 缓冲
                startupinfo=startupinfo,
                creationflags=creationflags
            )

            # 立即关闭 stdin 防止 FFmpeg 等待输入
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except Exception:
                    pass

            self.is_running = True
            self.consecutive_failures = 0
            self._first_frame_received = False
            self._stop_stderr.clear()

            # 启动 stderr 读取线程
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                daemon=True,
                name=f"FFmpegStderr-{self.config.camera_id}"
            )
            self._stderr_thread.start()

            logger.info(f"Camera {self.config.camera_id} FFmpeg 进程已启动 (PID: {self.process.pid})")

            # 等待一小段时间让 FFmpeg 初始化
            time.sleep(1.0)

            # 检查进程是否立即退出
            if self.process.poll() is not None:
                logger.error(f"Camera {self.config.camera_id} FFmpeg 进程立即退出，返回码: {self.process.returncode}")
                self.is_running = False
                return False

            return True

        except Exception as e:
            logger.exception(f"Camera {self.config.camera_id} 启动失败: {e}")
            self.is_running = False
            return False

    def _drain_stderr(self):
        """持续读取 stderr，记录错误信息"""
        if not self.process or not self.process.stderr:
            return
        try:
            for line in iter(self.process.stderr.readline, b''):
                if self._stop_stderr.is_set():
                    break
                decoded = line.decode(errors='ignore').strip()
                if decoded:
                    # 记录 FFmpeg 的错误和警告
                    if 'error' in decoded.lower() or 'fail' in decoded.lower():
                        logger.error(f"[FFmpeg {self.config.camera_id}] {decoded}")
                    elif 'warning' in decoded.lower():
                        logger.warning(f"[FFmpeg {self.config.camera_id}] {decoded}")
                    else:
                        logger.debug(f"[FFmpeg {self.config.camera_id}] {decoded}")
        except Exception as e:
            logger.debug(f"stderr drain exception for {self.config.camera_id}: {e}")

    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧"""
        if not self.is_running or not self.process or not self.process.stdout:
            return None

        # 检查进程是否还活着
        if self.process.poll() is not None:
            logger.warning(f"Camera {self.config.camera_id} FFmpeg 进程已退出，返回码: {self.process.returncode}")
            self.consecutive_failures += 1
            return None

        frame_size = int(self.config.width) * int(self.config.height) * 3

        try:
            # 读取完整帧数据
            raw = self.process.stdout.read(frame_size)

            if not raw:
                logger.debug(f"Camera {self.config.camera_id} 读取到空数据")
                self.consecutive_failures += 1
                return None

            if len(raw) < frame_size:
                logger.debug(f"Camera {self.config.camera_id} 读取到不完整帧: {len(raw)}/{frame_size} bytes")
                self.consecutive_failures += 1
                return None

            # 转换为 numpy 数组
            frame = np.frombuffer(raw, dtype=np.uint8)

            try:
                frame = frame.reshape((self.config.height, self.config.width, 3))
            except ValueError as e:
                logger.error(f"Camera {self.config.camera_id} reshape 失败: {e}, 数据长度: {len(frame)}")
                self.consecutive_failures += 1
                return None

            # 更新状态
            self.last_frame_time = time.time()
            self.consecutive_failures = 0

            if not self._first_frame_received:
                self._first_frame_received = True
                logger.info(f"Camera {self.config.camera_id} 成功接收第一帧")

            return frame

        except Exception as e:
            logger.exception(f"Camera {self.config.camera_id} 读取帧时发生错误: {e}")
            self.consecutive_failures += 1
            return None

    def stop(self):
        """停止FFmpeg进程"""
        self.is_running = False
        self._stop_stderr.set()

        if self.process:
            try:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Camera {self.config.camera_id} FFmpeg 进程未响应，强制终止")
                        self.process.kill()
                        self.process.wait()
            except Exception as e:
                logger.debug(f"停止 FFmpeg 进程时出错: {e}")
            finally:
                self.process = None

        # 等待 stderr 线程退出
        if self._stderr_thread and self._stderr_thread.is_alive():
            try:
                self._stderr_thread.join(timeout=1)
            except Exception:
                pass

        logger.info(f"Camera {self.config.camera_id} FFmpeg 已停止")

    def is_healthy(self) -> bool:
        """检查连接健康状态"""
        if not self.is_running:
            return False

        # 检查进程是否还在运行
        if not self.process or self.process.poll() is not None:
            return False

        # 如果从未接收到帧，给予更长的初始化时间
        if not self._first_frame_received:
            return True

        # 检查是否长时间没有新帧
        time_since_last_frame = time.time() - self.last_frame_time
        timeout = max(5.0, 3 * self.frame_interval)

        if time_since_last_frame > timeout:
            logger.warning(f"Camera {self.config.camera_id} 超时无帧: {time_since_last_frame:.1f}s")
            return False

        return True


# ============ 帧缓存池 ============
class FrameBuffer:
    """线程安全的帧缓存池"""

    def __init__(self, buffer_size: int = 2):
        self.buffer_size = buffer_size
        self.buffers: Dict[str, deque] = {}
        self.lock = threading.Lock()

    def put(self, camera_id: str, frame: np.ndarray, timestamp: float):
        """存入帧"""
        with self.lock:
            if camera_id not in self.buffers:
                self.buffers[camera_id] = deque(maxlen=self.buffer_size)
            self.buffers[camera_id].append({
                'frame': frame,
                'timestamp': timestamp
            })

    def get_latest(self, camera_id: str, copy_frame: bool = True) -> Optional[Dict[str, Any]]:
        """获取最新帧"""
        with self.lock:
            if camera_id not in self.buffers or not self.buffers[camera_id]:
                return None
            item = self.buffers[camera_id][-1]
            if copy_frame:
                return {'frame': item['frame'].copy(), 'timestamp': item['timestamp']}
            else:
                return item

    def clear(self, camera_id: str):
        """清空缓存"""
        with self.lock:
            if camera_id in self.buffers:
                self.buffers[camera_id].clear()


# ============ 摄像头管理器 ============
class CameraManager:
    """统一管理所有摄像头的连接和帧读取"""

    def __init__(self, frame_buffer: FrameBuffer, max_workers: int = 50):
        self.frame_buffer = frame_buffer
        self.configs: Dict[str, CameraConfig] = {}
        self.readers: Dict[str, FFmpegFrameReader] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="CamWorker")
        self.running_cameras = set()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def add_camera(self, config: CameraConfig) -> bool:
        """添加摄像头"""
        with self.lock:
            if config.camera_id in self.configs:
                logger.warning(f"Camera {config.camera_id} 已存在")
                return False

            self.configs[config.camera_id] = config
            reader = FFmpegFrameReader(config, use_hw_decode=False)
            self.readers[config.camera_id] = reader

            # 启动工作线程
            self.executor.submit(self._camera_worker, config.camera_id)
            logger.info(f"Camera {config.camera_id} 已添加")
            return True

    def remove_camera(self, camera_id: str) -> bool:
        """移除摄像头"""
        with self.lock:
            if camera_id not in self.configs:
                logger.warning(f"Camera {camera_id} 不存在")
                return False

            # 停止读取器
            if camera_id in self.readers:
                try:
                    self.readers[camera_id].stop()
                except Exception:
                    pass
                del self.readers[camera_id]

            # 清理配置和缓存
            del self.configs[camera_id]
            self.frame_buffer.clear(camera_id)
            self.running_cameras.discard(camera_id)

            logger.info(f"Camera {camera_id} 已移除")
            return True

    def _camera_worker(self, camera_id: str):
        """摄像头工作线程"""
        logger.info(f"Camera worker started for {camera_id}")

        while not self.stop_event.is_set():
            with self.lock:
                reader = self.readers.get(camera_id)
                config = self.configs.get(camera_id)

            if reader is None or config is None:
                break

            if not config.enabled:
                if reader.is_running:
                    reader.stop()
                time.sleep(1)
                continue

            # 启动或重启 FFmpeg
            if not reader.is_running:
                logger.info(f"Camera {camera_id} 正在连接...")
                ok = reader.start()
                if not ok:
                    logger.warning(f"Camera {camera_id} 连接失败，{config.reconnect_interval}秒后重试")
                    time.sleep(config.reconnect_interval)
                    continue
                # 给 FFmpeg 一些启动时间
                time.sleep(1)

            # 读取帧
            frame = reader.read_frame()

            if frame is not None:
                self.frame_buffer.put(camera_id, frame, time.time())
                self.running_cameras.add(camera_id)
                config._failure_count = 0
            else:
                config._failure_count += 1

                # 健康检查
                if not reader.is_healthy() or config._failure_count >= 15:
                    logger.warning(
                        f"Camera {camera_id} 不健康 (failures={config._failure_count}), "
                        f"is_running={reader.is_running}, 准备重连..."
                    )
                    reader.stop()
                    self.running_cameras.discard(camera_id)
                    time.sleep(config.reconnect_interval)
                    continue

            # 短暂休眠，避免 CPU 占用过高
            time.sleep(0.02)

        # 清理
        if camera_id in self.readers:
            try:
                self.readers[camera_id].stop()
            except Exception:
                pass
        self.running_cameras.discard(camera_id)
        logger.info(f"Camera worker stopped for {camera_id}")

    def get_frame(self, camera_id: str, copy_frame: bool = True) -> Optional[np.ndarray]:
        """获取最新帧"""
        data = self.frame_buffer.get_latest(camera_id, copy_frame=copy_frame)
        return data['frame'] if data else None

    def get_running_cameras(self) -> List[str]:
        """获取运行中的摄像头列表"""
        return list(self.running_cameras)

    def get_config(self, camera_id: str) -> Optional[CameraConfig]:
        """获取摄像头配置"""
        return self.configs.get(camera_id)

    def stop_all(self, wait: bool = True):
        """停止所有摄像头"""
        logger.info("CameraManager 停止所有摄像头...")
        self.stop_event.set()

        with self.lock:
            for r in list(self.readers.values()):
                try:
                    r.stop()
                except Exception:
                    pass
            self.readers.clear()
            self.configs.clear()
            self.running_cameras.clear()

        try:
            self.executor.shutdown(wait=wait)
        except Exception as e:
            logger.exception("关闭 executor 失败: %s", e)

        logger.info("CameraManager 已停止")


# ============ 检测器 ============
class Detector:
    """YOLO检测器封装"""

    def __init__(self, model_path: str, save_dir: str = "./detection_results", device: str = None, warmup: bool = True):
        self.model = YOLO(model_path) if device is None else YOLO(model_path, device=device)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "alarm"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "original"), exist_ok=True)
        logger.info(f"检测模型加载完成: {model_path}")

        if warmup:
            try:
                dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                _ = self.model.predict(dummy, conf=0.5, verbose=False)
                logger.info("模型 warmup 完成")
            except Exception:
                logger.debug("模型 warmup 失败")

    def detect(self, frame: np.ndarray, config: CameraConfig) -> tuple:
        """执行检测，返回 (是否有目标, 标注图像, 原始图像)"""
        try:
            results = self.model.predict(frame, conf=config.conf_threshold, verbose=False)
            if not results or len(results) == 0:
                return False, None, frame

            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return False, None, frame

            img_copy = r.orig_img.copy()
            names = getattr(r, "names", {})

            for box in boxes:
                conf = float(getattr(box, "conf", 0.0))
                if conf >= config.conf_threshold:
                    try:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                    except Exception:
                        continue

                    cls_id = int(getattr(box, "cls", 0))
                    class_name = names.get(cls_id, 'unknown')

                    color = (0, 255, 0)
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {conf:.2f}"
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img_copy, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
                    cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            return True, img_copy, r.orig_img

        except Exception as e:
            logger.exception(f"检测错误: {e}")
            return False, None, frame

    def save_result(self, camera_id: str, annotated_img: np.ndarray,
                    original_img: np.ndarray, timestamp: str):
        """保存检测结果"""
        try:
            alarm_path = os.path.join(self.save_dir, "alarm", f"{camera_id}_{timestamp}.jpg")
            ori_path = os.path.join(self.save_dir, "original", f"{camera_id}_{timestamp}.jpg")

            cv2.imwrite(alarm_path, annotated_img)
            cv2.imwrite(ori_path, original_img)

            logger.info(f"检测结果已保存: {camera_id} @ {timestamp}")
        except Exception:
            logger.exception("保存结果失败")


# ============ 检测调度器 ============
class DetectionScheduler:
    """检测任务调度器"""

    def __init__(self, camera_manager: CameraManager, detector: Detector,
                 max_workers: int = 3, queue_size: int = 500):
        self.camera_manager = camera_manager
        self.detector = detector
        self.detection_queue = queue.Queue(maxsize=queue_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="DetWorker")
        self.stop_event = threading.Event()

    def start(self, num_schedule_threads: int = 1, num_detect_workers: int = 3):
        """启动调度器"""
        for _ in range(num_schedule_threads):
            self.executor.submit(self._schedule_worker)
        for _ in range(num_detect_workers):
            self.executor.submit(self._detection_worker)
        logger.info("检测调度器已启动")

    def _schedule_worker(self):
        """调度线程"""
        logger.info("DetectionScheduler schedule worker started")
        while not self.stop_event.is_set():
            running_cameras = self.camera_manager.get_running_cameras()
            if not running_cameras:
                time.sleep(1)
                continue

            for camera_id in running_cameras:
                config = self.camera_manager.get_config(camera_id)
                if not config or not config.enabled:
                    continue

                frame = self.camera_manager.get_frame(camera_id)
                if frame is None:
                    continue

                try:
                    self.detection_queue.put_nowait({
                        'camera_id': camera_id,
                        'frame': frame,
                        'config': config,
                        'timestamp': time.time()
                    })
                except queue.Full:
                    pass

            sleep_time = max(0.2, 1.0 / max(1, len(running_cameras)))
            time.sleep(sleep_time)

        logger.info("DetectionScheduler schedule worker stopped")

    def _detection_worker(self):
        """检测工作线程"""
        logger.info("DetectionScheduler detection worker started")
        while not self.stop_event.is_set() or not self.detection_queue.empty():
            try:
                task = self.detection_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                self._process_detection(task)
            except Exception:
                logger.exception("处理检测任务失败")
            finally:
                self.detection_queue.task_done()

        logger.info("DetectionScheduler detection worker stopped")

    def _process_detection(self, task: Dict):
        """处理检测任务"""
        camera_id = task['camera_id']
        frame = task['frame']
        config = task['config']

        has_target, annotated_img, original_img = self.detector.detect(frame, config)

        if has_target and config.can_alarm():
            timestamp = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
            try:
                self.detector.save_result(camera_id, annotated_img, original_img, timestamp)
            except Exception:
                logger.exception("保存检测结果失败")
            config.update_alarm_time()
            logger.info(f"[ALARM] Camera {camera_id} 检测到目标 @ {timestamp}")

    def stop(self, wait_for_queue: bool = True):
        """停止调度器"""
        logger.info("Stopping DetectionScheduler...")
        self.stop_event.set()
        if wait_for_queue:
            try:
                self.detection_queue.join()
            except Exception:
                pass
        try:
            self.executor.shutdown(wait=True)
        except Exception:
            pass
        logger.info("检测调度器已停止")


# ============ Flask API服务 ============
class APIServer:
    """Flask API服务"""

    def __init__(self, camera_manager: CameraManager):
        self.app = Flask(__name__)
        self.camera_manager = camera_manager
        self._register_routes()

    def _register_routes(self):
        """注册路由"""

        @self.app.route('/addVideo', methods=['POST'])
        def add_video():
            try:
                data = request.json
                if not data or 'rtsp_url' not in data:
                    return jsonify({"status": 400, "message": "缺少rtsp_url参数"}), 400

                rtsp_url = data['rtsp_url']
                camera_id = data.get('camera_id', f"cam_{int(time.time())}")

                config = CameraConfig(
                    camera_id=camera_id,
                    rtsp_url=rtsp_url,
                    sample_rate=int(data.get('sample_rate', 5)),
                    conf_threshold=float(data.get('conf_threshold', 0.5)),
                    alarm_interval=int(data.get('alarm_interval', 300)),
                    width=int(data.get('width', 1920)),
                    height=int(data.get('height', 1080)),
                )

                if self.camera_manager.add_camera(config):
                    return jsonify({"status": 200, "message": f"摄像头 {camera_id} 添加成功"})
                else:
                    return jsonify({"status": 400, "message": "添加失败,摄像头已存在"}), 400

            except Exception as e:
                logger.exception("添加摄像头失败")
                return jsonify({"status": 500, "message": str(e)}), 500

        @self.app.route('/deleteVideo', methods=['POST'])
        def delete_video():
            try:
                data = request.json
                if not data or 'camera_id' not in data:
                    return jsonify({"status": 400, "message": "缺少camera_id参数"}), 400

                camera_id = data['camera_id']

                if self.camera_manager.remove_camera(camera_id):
                    return jsonify({"status": 200, "message": f"摄像头 {camera_id} 删除成功"})
                else:
                    return jsonify({"status": 400, "message": "删除失败,摄像头不存在"}), 400

            except Exception as e:
                logger.exception("删除摄像头失败")
                return jsonify({"status": 500, "message": str(e)}), 500

        @self.app.route('/getVideos', methods=['GET'])
        def get_videos():
            with self.camera_manager.lock:
                total = len(self.camera_manager.configs)
                cameras = []
                for cam_id, cfg in self.camera_manager.configs.items():
                    cameras.append({
                        "camera_id": cam_id,
                        "rtsp_url": cfg.rtsp_url,
                        "enabled": cfg.enabled,
                        "running": cam_id in self.camera_manager.running_cameras,
                        "last_alarm": cfg.last_alarm_time.isoformat() if cfg.last_alarm_time else None
                    })
            return jsonify({
                "status": 200,
                "total": total,
                "running_count": len(self.camera_manager.running_cameras),
                "cameras": cameras
            })

        @self.app.route('/toggleCamera', methods=['POST'])
        def toggle_camera():
            try:
                data = request.json
                if not data or 'camera_id' not in data or 'enabled' not in data:
                    return jsonify({"status": 400, "message": "缺少参数 camera_id/enabled"}), 400

                cam_id = data['camera_id']
                enabled = bool(data['enabled'])
                cfg = self.camera_manager.get_config(cam_id)

                if not cfg:
                    return jsonify({"status": 404, "message": "摄像头未找到"}), 404

                cfg.enabled = enabled
                if not enabled:
                    reader = self.camera_manager.readers.get(cam_id)
                    if reader:
                        reader.stop()
                        self.camera_manager.running_cameras.discard(cam_id)

                return jsonify({"status": 200, "message": f"摄像头 {cam_id} 已 {'启用' if enabled else '禁用'}"})
            except Exception:
                logger.exception("切换摄像头状态失败")
                return jsonify({"status": 500, "message": "内部错误"}), 500

        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": 200, "message": "服务运行正常"})

    def run(self, host='0.0.0.0', port=8080):
        """启动Flask服务"""
        self.app.run(host=host, port=port, threaded=True)


# ============ 主程序 ============
def main():
    """主函数"""

    # 配置（请根据实际修改）
    MODEL_PATH = r"C:\Users\26601\Desktop\phone.pt"
    SAVE_DIR = r"C:\Users\26601\Desktop\detection_results"
    API_HOST = "0.0.0.0"
    API_PORT = 8080

    # 1. 创建核心组件
    frame_buffer = FrameBuffer(buffer_size=2)
    camera_manager = CameraManager(frame_buffer, max_workers=50)
    detector = Detector(MODEL_PATH, SAVE_DIR, warmup=True)
    scheduler = DetectionScheduler(camera_manager, detector, max_workers=3, queue_size=500)

    # 2. 启动检测调度器
    scheduler.start(num_schedule_threads=1, num_detect_workers=3)

    # 3. 启动API服务
    api_server = APIServer(camera_manager)

    # 信号处理，优雅退出
    def _signal_handler(signum, frame):
        logger.info(f"接收到信号 {signum}，准备停止服务...")
        try:
            scheduler.stop(wait_for_queue=True)
        except Exception:
            pass
        try:
            camera_manager.stop_all(wait=True)
        except Exception:
            pass
        logger.info("系统已关闭")
        os._exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("=" * 60)
    logger.info("系统启动完成!")
    logger.info(f"API服务: http://{API_HOST}:{API_PORT}")
    logger.info("添加摄像头: POST /addVideo")
    logger.info("删除摄像头: POST /deleteVideo")
    logger.info("查询状态: GET /getVideos")
    logger.info("切换状态: POST /toggleCamera")
    logger.info("健康检查: GET /health")
    logger.info("=" * 60)

    try:
        api_server.run(host=API_HOST, port=API_PORT)
    except Exception:
        logger.exception("API 服务异常退出")
    finally:
        logger.info("主程序结束，开始清理...")
        try:
            scheduler.stop(wait_for_queue=True)
        except Exception:
            pass
        try:
            camera_manager.stop_all(wait=True)
        except Exception:
            pass
        logger.info("系统已完全停止")


if __name__ == "__main__":
    main()
"""
优化的多路RTSP视频流检测系统
支持FFmpeg和OpenCV双模式,自动回退
"""

import subprocess
import threading
import queue
import time
import datetime
import os
import shutil
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List
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
    format="%(asctime)s [%(levelname)s] [%(threadName)-10s] %(message)s",
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


# ============ 检测FFmpeg是否可用 ============
def check_ffmpeg_available() -> bool:
    """检测系统是否安装FFmpeg"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False


# ============ FFmpeg帧读取器 ============
class FFmpegFrameReader:
    """基于FFmpeg的高性能帧读取器"""

    def __init__(self, config: CameraConfig, use_hw_decode: bool = False):
        self.config = config
        self.use_hw_decode = use_hw_decode
        self.process = None
        self.is_running = False
        self.last_frame_time = 0
        self.frame_interval = 1.0 / config.sample_rate
        self.consecutive_failures = 0
        self.max_failures = 5

    def _build_ffmpeg_command(self) -> List[str]:
        """构建FFmpeg命令"""
        cmd = ['ffmpeg']

        # RTSP优化参数
        cmd.extend([
            '-rtsp_transport', 'tcp',
            '-stimeout', '5000000',
            '-max_delay', '500000',
            '-reorder_queue_size', '0',
        ])

        # 硬件解码(可选)
        if self.use_hw_decode:
            try:
                cmd.extend(['-hwaccel', 'cuda'])
            except:
                pass

        # 输入源
        cmd.extend(['-i', self.config.rtsp_url])

        # 帧率和分辨率控制
        cmd.extend([
            '-r', str(self.config.sample_rate),
            '-s', f'{self.config.width}x{self.config.height}'
        ])

        # 输出格式
        cmd.extend([
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-an',
            'pipe:1'
        ])

        return cmd

    def start(self) -> bool:
        """启动FFmpeg进程"""
        try:
            cmd = self._build_ffmpeg_command()
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10 ** 8
            )
            self.is_running = True
            self.consecutive_failures = 0
            logger.info(f"Camera {self.config.camera_id} FFmpeg模式启动成功")
            return True
        except Exception as e:
            logger.error(f"Camera {self.config.camera_id} FFmpeg启动失败: {e}")
            self.is_running = False
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧"""
        if not self.is_running or not self.process:
            return None

        try:
            frame_size = self.config.width * self.config.height * 3
            raw_frame = self.process.stdout.read(frame_size)

            if len(raw_frame) != frame_size:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    logger.warning(f"Camera {self.config.camera_id} 连续失败{self.max_failures}次")
                return None

            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((self.config.height, self.config.width, 3))

            self.last_frame_time = time.time()
            self.consecutive_failures = 0
            return frame

        except Exception as e:
            logger.error(f"Camera {self.config.camera_id} 读取错误: {e}")
            self.stop()
            return None

    def stop(self):
        """停止FFmpeg进程"""
        self.is_running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            finally:
                self.process = None

    def is_healthy(self) -> bool:
        """检查连接健康状态"""
        if not self.is_running:
            return False
        if time.time() - self.last_frame_time > self.frame_interval * 3:
            return False
        return self.process and self.process.poll() is None


# ============ OpenCV帧读取器(备用) ============
class OpenCVFrameReader:
    """基于OpenCV的帧读取器(FFmpeg不可用时的备用方案)"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = None
        self.is_running = False
        self.last_frame_time = 0
        self.frame_interval = 1.0 / config.sample_rate
        self.consecutive_failures = 0
        self.max_failures = 5

    def start(self) -> bool:
        """启动OpenCV读取"""
        try:
            self.cap = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)

            # 设置参数
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲
            self.cap.set(cv2.CAP_PROP_FPS, self.config.sample_rate)

            if self.cap.isOpened():
                self.is_running = True
                self.consecutive_failures = 0
                logger.info(f"Camera {self.config.camera_id} OpenCV模式启动成功")
                return True
            else:
                logger.error(f"Camera {self.config.camera_id} OpenCV无法打开流")
                return False

        except Exception as e:
            logger.error(f"Camera {self.config.camera_id} OpenCV启动失败: {e}")
            self.is_running = False
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧"""
        if not self.is_running or not self.cap:
            return None

        try:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    logger.warning(f"Camera {self.config.camera_id} 连续失败{self.max_failures}次")
                return None

            # 调整分辨率
            if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
                frame = cv2.resize(frame, (self.config.width, self.config.height))

            self.last_frame_time = time.time()
            self.consecutive_failures = 0
            return frame

        except Exception as e:
            logger.error(f"Camera {self.config.camera_id} 读取错误: {e}")
            self.stop()
            return None

    def stop(self):
        """停止读取"""
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            finally:
                self.cap = None

    def is_healthy(self) -> bool:
        """检查连接健康状态"""
        if not self.is_running:
            return False
        if time.time() - self.last_frame_time > self.frame_interval * 3:
            return False
        return self.cap and self.cap.isOpened()


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

    def get_latest(self, camera_id: str) -> Optional[Dict]:
        """获取最新帧"""
        with self.lock:
            if camera_id not in self.buffers or not self.buffers[camera_id]:
                return None
            return self.buffers[camera_id][-1]

    def clear(self, camera_id: str):
        """清空缓存"""
        with self.lock:
            if camera_id in self.buffers:
                self.buffers[camera_id].clear()


# ============ 摄像头管理器 ============
class CameraManager:
    """统一管理所有摄像头的连接和帧读取"""

    def __init__(self, frame_buffer: FrameBuffer, max_workers: int = 100,
                 prefer_ffmpeg: bool = True):
        self.frame_buffer = frame_buffer
        self.configs: Dict[str, CameraConfig] = {}
        self.readers: Dict[str, object] = {}  # FFmpegFrameReader 或 OpenCVFrameReader
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="CamWorker")
        self.running_cameras = set()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        # 检测FFmpeg
        self.ffmpeg_available = check_ffmpeg_available() if prefer_ffmpeg else False
        if self.ffmpeg_available:
            logger.info("✓ FFmpeg可用,将使用FFmpeg模式(高性能)")
        else:
            logger.warning("✗ FFmpeg不可用,将使用OpenCV模式(兼容模式)")
            if prefer_ffmpeg:
                logger.warning("提示: 安装FFmpeg可获得更好的性能和稳定性")
                logger.warning("下载地址: https://ffmpeg.org/download.html")

    def add_camera(self, config: CameraConfig) -> bool:
        """添加摄像头"""
        with self.lock:
            if config.camera_id in self.configs:
                logger.warning(f"Camera {config.camera_id} 已存在")
                return False

            self.configs[config.camera_id] = config

            # 根据FFmpeg可用性选择读取器
            if self.ffmpeg_available:
                reader = FFmpegFrameReader(config, use_hw_decode=False)
            else:
                reader = OpenCVFrameReader(config)

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
                self.readers[camera_id].stop()
                del self.readers[camera_id]

            # 清理配置和缓存
            del self.configs[camera_id]
            self.frame_buffer.clear(camera_id)
            self.running_cameras.discard(camera_id)

            logger.info(f"Camera {camera_id} 已移除")
            return True

    def _camera_worker(self, camera_id: str):
        """摄像头工作线程"""
        while not self.stop_event.is_set():
            if camera_id not in self.readers:
                break

            reader = self.readers[camera_id]
            config = self.configs.get(camera_id)

            if not config or not config.enabled:
                time.sleep(1)
                continue

            # 启动连接
            if not reader.is_running:
                if reader.start():
                    time.sleep(1)
                else:
                    time.sleep(config.reconnect_interval)
                continue

            # 读取帧
            frame = reader.read_frame()

            if frame is not None:
                self.frame_buffer.put(camera_id, frame, time.time())
                self.running_cameras.add(camera_id)
            else:
                # 检查健康状态
                if not reader.is_healthy():
                    logger.warning(f"Camera {camera_id} 不健康,准备重连...")
                    reader.stop()
                    self.running_cameras.discard(camera_id)
                    time.sleep(config.reconnect_interval)

            # 控制采样率
            time.sleep(1.0 / config.sample_rate)

        # 清理
        if camera_id in self.readers:
            self.readers[camera_id].stop()
        self.running_cameras.discard(camera_id)

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """获取最新帧"""
        data = self.frame_buffer.get_latest(camera_id)
        return data['frame'].copy() if data else None

    def get_running_cameras(self) -> List[str]:
        """获取运行中的摄像头列表"""
        return list(self.running_cameras)

    def get_config(self, camera_id: str) -> Optional[CameraConfig]:
        """获取摄像头配置"""
        return self.configs.get(camera_id)

    def stop_all(self):
        """停止所有摄像头"""
        self.stop_event.set()
        for reader in self.readers.values():
            reader.stop()
        self.executor.shutdown(wait=True)
        logger.info("所有摄像头已停止")


# ============ 检测器 ============
class Detector:
    """YOLO检测器封装"""

    def __init__(self, model_path: str, save_dir: str = "./detection_results"):
        self.model = YOLO(model_path)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "alarm"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "original"), exist_ok=True)
        logger.info(f"检测模型加载完成: {model_path}")

    def detect(self, frame: np.ndarray, config: CameraConfig) -> tuple:
        """
        执行检测
        返回: (是否有检测目标, 标注后的图像, 原始图像)
        """
        try:
            results = self.model.predict(frame, conf=config.conf_threshold, verbose=False)
            if not results or len(results) == 0:
                return False, None, frame

            r = results[0]
            boxes = r.boxes

            if len(boxes) == 0:
                return False, None, frame

            # 绘制检测框
            img_copy = r.orig_img.copy()
            names = r.names

            for box in boxes:
                conf = box.conf.item()
                if conf > config.conf_threshold:
                    x1, y1, x2, y2 = [int(round(x)) for x in box.xyxy[0].cpu().numpy()]
                    cls_id = int(box.cls.item())
                    class_name = names.get(cls_id, 'unknown')

                    # 绘制边框
                    color = (0, 255, 0)
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

                    # 绘制标签
                    label = f"{class_name}: {conf:.2f}"
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(img_copy, (x1, y1 - text_h - baseline),
                                  (x1 + text_w, y1), color, -1)
                    cv2.putText(img_copy, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            return True, img_copy, r.orig_img

        except Exception as e:
            logger.error(f"检测错误: {e}")
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
        except Exception as e:
            logger.error(f"保存结果失败: {e}")


# ============ 检测调度器 ============
class DetectionScheduler:
    """检测任务调度器"""

    def __init__(self, camera_manager: CameraManager, detector: Detector,
                 max_workers: int = 5):
        self.camera_manager = camera_manager
        self.detector = detector
        self.detection_queue = queue.Queue(maxsize=100)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="DetWorker")
        self.stop_event = threading.Event()

    def start(self):
        """启动调度器"""
        self.executor.submit(self._schedule_worker)
        for _ in range(3):  # 启动3个检测工作线程
            self.executor.submit(self._detection_worker)
        logger.info("检测调度器已启动")

    def _schedule_worker(self):
        """调度线程:循环获取帧并提交检测任务"""
        while not self.stop_event.is_set():
            running_cameras = self.camera_manager.get_running_cameras()

            if not running_cameras:
                time.sleep(2)
                continue

            for camera_id in running_cameras:
                frame = self.camera_manager.get_frame(camera_id)
                config = self.camera_manager.get_config(camera_id)

                if frame is not None and config and config.enabled:
                    try:
                        self.detection_queue.put_nowait({
                            'camera_id': camera_id,
                            'frame': frame,
                            'config': config,
                            'timestamp': time.time()
                        })
                    except queue.Full:
                        pass  # 队列满时跳过

            # 动态调整间隔
            sleep_time = max(0.5, 2.0 / len(running_cameras))
            time.sleep(sleep_time)

    def _detection_worker(self):
        """检测工作线程"""
        while not self.stop_event.is_set():
            try:
                task = self.detection_queue.get(timeout=1)
                self._process_detection(task)
            except queue.Empty:
                continue

    def _process_detection(self, task: Dict):
        """处理检测任务"""
        camera_id = task['camera_id']
        frame = task['frame']
        config = task['config']

        # 执行检测
        has_target, annotated_img, original_img = self.detector.detect(frame, config)

        if has_target and config.can_alarm():
            # 保存结果
            timestamp = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
            self.detector.save_result(camera_id, annotated_img, original_img, timestamp)

            # 更新报警时间
            config.update_alarm_time()
            logger.info(f"[ALARM] Camera {camera_id} 检测到目标 @ {timestamp}")

    def stop(self):
        """停止调度器"""
        self.stop_event.set()
        self.executor.shutdown(wait=True)
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
                    return jsonify({"status": 400, "message": "缺少rtsp_url参数"})

                rtsp_url = data['rtsp_url']
                camera_id = data.get('camera_id', f"cam_{int(time.time())}")

                config = CameraConfig(
                    camera_id=camera_id,
                    rtsp_url=rtsp_url,
                    sample_rate=data.get('sample_rate', 5),
                    conf_threshold=data.get('conf_threshold', 0.5),
                    alarm_interval=data.get('alarm_interval', 300)
                )

                if self.camera_manager.add_camera(config):
                    return jsonify({"status": 200, "message": f"摄像头 {camera_id} 添加成功"})
                else:
                    return jsonify({"status": 400, "message": "添加失败,摄像头已存在"})

            except Exception as e:
                logger.error(f"添加摄像头失败: {e}")
                return jsonify({"status": 500, "message": str(e)})

        @self.app.route('/deleteVideo', methods=['POST'])
        def delete_video():
            try:
                data = request.json
                if not data or 'camera_id' not in data:
                    return jsonify({"status": 400, "message": "缺少camera_id参数"})

                camera_id = data['camera_id']

                if self.camera_manager.remove_camera(camera_id):
                    return jsonify({"status": 200, "message": f"摄像头 {camera_id} 删除成功"})
                else:
                    return jsonify({"status": 400, "message": "删除失败,摄像头不存在"})

            except Exception as e:
                logger.error(f"删除摄像头失败: {e}")
                return jsonify({"status": 500, "message": str(e)})

        @self.app.route('/getVideos', methods=['GET'])
        def get_videos():
            running = self.camera_manager.get_running_cameras()
            all_cameras = list(self.camera_manager.configs.keys())
            return jsonify({
                "status": 200,
                "total": len(all_cameras),
                "running": len(running),
                "cameras": all_cameras,
                "running_cameras": running
            })

        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": 200,
                "message": "服务运行正常",
                "mode": "FFmpeg" if self.camera_manager.ffmpeg_available else "OpenCV"
            })

    def run(self, host='0.0.0.0', port=8080):
        """启动Flask服务"""
        self.app.run(host=host, port=port, threaded=True)


# ============ 主程序 ============
def main():
    """主函数"""

    # 配置
    MODEL_PATH = r"C:\Users\26601\Desktop\phone.pt"  # 修改为你的模型路径
    SAVE_DIR = r"C:\Users\26601\Desktop/detection_results"
    API_HOST = "0.0.0.0"
    API_PORT = 8080

    logger.info("=" * 60)
    logger.info("多路RTSP视频流检测系统启动中...")
    logger.info("=" * 60)

    # 1. 创建核心组件
    frame_buffer = FrameBuffer(buffer_size=2)
    camera_manager = CameraManager(frame_buffer, max_workers=100, prefer_ffmpeg=True)
    detector = Detector(MODEL_PATH, SAVE_DIR)
    scheduler = DetectionScheduler(camera_manager, detector, max_workers=5)

    # 2. 启动检测调度器
    scheduler.start()

    # 3. 启动API服务
    api_server = APIServer(camera_manager)

    logger.info("=" * 60)
    logger.info("✓ 系统启动完成!")
    logger.info(f"✓ API服务: http://{API_HOST}:{API_PORT}")
    logger.info(f"✓ 运行模式: {'FFmpeg(高性能)' if camera_manager.ffmpeg_available else 'OpenCV(兼容)'}")
    logger.info("=" * 60)
    logger.info("API接口:")
    logger.info("  POST /addVideo     - 添加摄像头")
    logger.info("  POST /deleteVideo  - 删除摄像头")
    logger.info("  GET  /getVideos    - 查询状态")
    logger.info("  GET  /health       - 健康检查")
    logger.info("=" * 60)

    try:
        api_server.run(host=API_HOST, port=API_PORT)
    except KeyboardInterrupt:
        logger.info("\n接收到停止信号...")
    finally:
        scheduler.stop()
        camera_manager.stop_all()
        logger.info("系统已关闭")


if __name__ == "__main__":
    main()
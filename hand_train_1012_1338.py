import os
from ultralytics import YOLO

def train_driver_hand():
    # ========= 1. 基础配置 =========
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU
    data_path = r"/path/to/your/data.yaml"   # 你的数据集配置文件
    model_path = "yolov8n.pt"                # 预训练模型
    project_name = "driver_hand"             # 输出项目名

    # ========= 2. 创建模型 =========
    model = YOLO(model_path)

    # ========= 3. 启动训练 =========
    model.train(
        data=data_path,
        epochs=150,              # 训练轮次（充足但不过拟合）
        imgsz=640,               # 输入分辨率
        batch=32,                # 批次大小，若显存不足可调小
        device=0,                # GPU id
        workers=8,               # 加速加载数据

        # ========= 优化参数 =========
        lr0=0.001,               # 初始学习率（小场景检测建议小一点）
        lrf=0.01,                # 最终学习率（cosine衰减终点）
        momentum=0.937,          # 动量
        weight_decay=0.0005,     # 权重衰减
        warmup_epochs=3.0,       # 预热阶段
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # ========= 数据增强 =========
        hsv_h=0.015,             # 色调增强
        hsv_s=0.7,               # 饱和度增强
        hsv_v=0.4,               # 明亮度增强（夜间仿真）
        degrees=5.0,             # 小角度旋转
        translate=0.1,           # 平移
        scale=0.9,               # 缩放比例
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,              # 上下翻转（不常见，可关闭）
        fliplr=0.5,              # 左右翻转（手对称，增强有效）
        mosaic=1.0,              # Mosaic 拼图增强
        mixup=0.15,              # MixUp混合增强
        copy_paste=0.1,          # 复制粘贴增强（对小目标手有帮助）

        # ========= 正则化与稳定性 =========
        label_smoothing=0.05,
        patience=30,             # EarlyStopping 容忍次数
        cos_lr=True,             # 使用Cosine学习率调度
        optimizer="SGD",         # 优化器可尝试 "AdamW"

        # ========= 输出与日志 =========
        project=project_name,
        name="yolov8n_driverhand_exp1",
        exist_ok=True,
        verbose=True
    )

if __name__ == "__main__":
    train_driver_hand()

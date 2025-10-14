import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from ultralytics import YOLO

model = YOLO('./yolov8n.pt')


def train_driver_hand():
    # ========= 1. GPU 设置 =========
    # 指定可见 GPU3
    torch.cuda.empty_cache()  # 清空残余显存

    # ========= 2. 基础配置 =========
    data_path = r"/data/clearingvehicle/eating/hand_traindata_v1/data.yaml"
    project_name = "hand_model"

    # ========= 4. 启动训练 =========
    model.train(
        data=data_path,
        epochs=150,
        imgsz=640,
        batch=32,
        device=0,  # 对应上面CUDA_VISIBLE_DEVICES=3
        workers=8,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # ========= 数据增强 =========
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.9,
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,

        # ========= 优化与稳定性 =========
        patience=30,
        cos_lr=True,
        optimizer="SGD",
        amp=False,  # ✅ 关闭AMP混合精度检查（防CUBLAS错误）
        deterministic=True,  # ✅ 固定性增强
        label_smoothing=0.0,  # ✅ 最新版已弃用此参数，建议去掉或设为0

        # ========= 输出与日志 =========
        project=project_name,
        name="run1",
        exist_ok=True,
        verbose=True
    )


if __name__ == "__main__":
    train_driver_hand()


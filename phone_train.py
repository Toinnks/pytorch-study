
from ultralytics import YOLO
import torch
import os

device = torch.device("cuda:3")

def train_phone_detector():
    data_yaml = "phone-dataset-v6-1030/data.yaml"
    model = YOLO('yolov8n.pt')

    results = model.train(
        # ====== 数据配置 ======
        data=data_yaml,
        epochs=350,                # 多训一些轮次
        imgsz=640,
        batch=16,
        device=device.index,
        workers=2,
        cache=True,

        # ====== 优化器配置 ======
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        momentum=0.937,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # ====== 几何增强（核心部分）======
        degrees=0,       # 扩大旋转角度范围（多角度手机姿态）
        translate=0.15,     # 增强平移范围
        scale=0.5,          # 缩放范围 (0.5 ~ 1.5)
        shear=8.0,          # 剪切变换增强角度多样性
        perspective=0.0005, # 轻微透视
        fliplr=0.5,         # 左右翻转
        flipud=0.0,         # 不上下翻转（保留方向性）

        # ====== 夜间场景专用颜色增强 ======
        hsv_h=0.02,         # 色调轻微扰动（夜间光源色偏）
        hsv_s=0.6,          # 饱和度扰动
        hsv_v=0.5,          # 亮度扰动增强夜视鲁棒性

        # ====== Mosaic 与混合增强 ======
        mosaic=0.9,         # 夜间目标复杂背景下保留Mosaic
        mixup=0.1,          # 减少MixUp（避免光照混叠）
        copy_paste=0.1,

        # ====== 小目标检测优化 ======
        close_mosaic=15,    # 后期关闭Mosaic帮助模型收敛
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # ====== 训练策略 ======
        patience=60,
        save=True,
        save_period=10,
        project='runs',
        name='phone_v6_1030',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=False,
        val=True,
        plots=True,
        label_smoothing=0.0,
        iou=0.7,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        amp=True,
    )

    print("\n" + "=" * 50)
    print(" 训练完成！")
    print("=" * 50)
    print(f"最佳模型路径: {results.save_dir}/weights/best.pt")
    print(f"最终模型路径: {results.save_dir}/weights/last.pt")

    return results


# ====== 主入口 ======
if __name__ == '__main__':
    print(" 启动手机检测训练（夜间+多角度增强）...")
    train_phone_detector()

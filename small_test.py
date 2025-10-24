from ultralytics import YOLO
import torch


def finetune_phone_driver_detector():
    """
    模型2：基于模型1的微调版本
    目标：检测司机开车使用手机/打电话场景（含夜间）
    """
    model = YOLO("runs/phone_v4_1024/weights/best.pt")  # 模型1路径（请替换为实际路径）

    data_yaml = "/phone-v5-1024/data.yaml"  # 替换为你新数据集的配置文件路径

    results = model.train(
        data=data_yaml,
        epochs=200,  # 微调不需要太长，200轮足够
        imgsz=640,  # 保持输入尺寸一致
        batch=16,  # 视显存而定
        pretrained=True,  # 保留原有权重

        # ---------- 优化器与学习率 ----------
        optimizer='AdamW',
        lr0=0.0005,  # 微调阶段学习率更低
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,

        # ==========================================
        # 数据增强策略（针对夜间驾驶+光照变化）
        # ==========================================
        degrees=8.0,  # 驾驶场景中手机角度变化有限
        translate=0.08,  # 适度平移
        scale=0.4,  # 增强远近距离变化
        shear=4.0,
        perspective=0.0001,
        flipud=0.0,  # 禁止上下翻转（车内场景固定）
        fliplr=0.4,  # 仍允许水平翻转（驾驶员左右手使用）

        # ---------- 颜色增强（重点处理夜间和黑白图） ----------
        hsv_h=0.02,  # 轻微色调变化
        hsv_s=0.5,  # 降低饱和度增强
        hsv_v=0.5,  # 增强亮度变化（应对黑白/夜间）
        # 新增自定义夜间模拟：
        # 可考虑在训练前用 albumentations 预处理一部分样本，加上随机Gamma/亮度降低模拟低照环境

        # ---------- Mosaic / MixUp ----------
        mosaic=0.8,  # 略微降低 Mosaic 比例（真实场景一致性更强）
        mixup=0.1,  # MixUp 保留少量
        copy_paste=0.1,

        # ---------- 小目标检测优化 ----------
        close_mosaic=10,
        box=7.0,
        cls=0.5,
        dfl=1.5,

        # ---------- 训练策略 ----------
        patience=50,
        save=True,
        save_period=10,
        cache=True,
        device=device.index if hasattr(device, 'index') else 0,
        workers=8,
        project='runs',
        name='/phone-v5-1024',
        exist_ok=True,
        verbose=True,
        seed=42,
        deterministic=False,

        # ---------- 验证 ----------
        val=True,
        plots=True,

        # ---------- 其他 ----------
        label_smoothing=0.0,
        iou=0.7,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        amp=True,
    )

    # ==========================================
    # 训练结果输出
    # ==========================================
    print("\n" + "=" * 60)
    print("微调训练完成！")
    print("=" * 60)
    print(f"最佳模型路径: {results.save_dir}/weights/best.pt")
    print(f"最终模型路径: {results.save_dir}/weights/last.pt")

    return results


if __name__ == "__main__":
    finetune_phone_driver_detector()

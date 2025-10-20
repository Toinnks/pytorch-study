from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import random


def generate_colors(num_classes):
    """为每个类别生成不同的颜色"""
    colors = []
    for i in range(num_classes):
        # 生成鲜艳且区分度高的颜色
        hue = int(180 * i / num_classes)  # 使用HSV空间的Hue值
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        color = [int(c) for c in color]
        colors.append(color)

    # 打乱颜色顺序，使相邻类别颜色差异更大
    random.shuffle(colors)
    return colors


def draw_detections(image, results, colors, class_names, conf_threshold=0.25):
    """在图像上绘制检测结果"""
    result_image = image.copy()

    # 遍历所有检测结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 获取边界框坐标、置信度和类别
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())

            if conf < conf_threshold:
                continue

            # 获取类别颜色和名称
            color = colors[cls % len(colors)]
            class_name = class_names[cls] if class_names else f"Class {cls}"

            # 绘制边界框
            cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # 绘制标签背景
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (int(x1), int(y1) - label_size[1] - 10),
                          (int(x1) + label_size[0], int(y1)), color, -1)

            # 绘制标签文本
            cv2.putText(result_image, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return result_image


def process_images_with_yolo(model, input_dir, output_dir, class_names=None, conf_threshold=0.25):
    """
    使用YOLO模型处理文件夹中的所有图片

    参数:
        model: YOLO模型
        input_dir: 输入图片文件夹路径
        output_dir: 输出图片文件夹路径
        class_names: 类别名称列表
        conf_threshold: 置信度阈值
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取类别数量并生成颜色
    num_classes = len(model.names) if hasattr(model, 'names') else (len(class_names) if class_names else 80)
    colors = generate_colors(num_classes)

    # 如果未提供类别名称，使用模型自带的名称
    if class_names is None and hasattr(model, 'names'):
        class_names = [model.names[i] for i in range(len(model.names))]

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # 获取所有图片文件
    image_files = []
    for file_path in Path(input_dir).iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)

    print(f"找到 {len(image_files)} 张图片需要处理")

    # 处理每张图片
    for i, image_path in enumerate(image_files):
        print(f"处理图片 {i + 1}/{len(image_files)}: {image_path.name}")

        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}")
            continue

        # 使用模型进行推理
        results = model(image)

        # 绘制检测结果
        result_image = draw_detections(image, results, colors, class_names, conf_threshold)

        # 保存结果图片
        output_path = os.path.join(output_dir, f"detected_{image_path.name}")
        cv2.imwrite(output_path, result_image)

        print(f"已保存: {output_path}")

    print(f"所有图片处理完成！结果保存在: {output_dir}")


# 使用方法示例
if __name__ == "__main__":

    # 1. 加载YOLO模型（请替换为你的模型路径）
    model_path = "best.pt"  # 或者 .onnx, .engine 等
    model = YOLO(model_path)

    # 2. 设置路径
    input_directory = r"D:\dingdingDownloads\phone_data_own\val\images" # 测试图片文件夹
    output_directory = r"D:\dingdingDownloads\output3"  # 输出结果文件夹

    custom_class_names = ["phone"]

    # 4. 处理图片
    process_images_with_yolo(
        model=model,
        input_dir=input_directory,
        output_dir=output_directory,
        class_names=custom_class_names,
        conf_threshold=0.1  # 置信度阈值，可根据需要调整
    )
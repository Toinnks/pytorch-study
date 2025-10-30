import cv2
import numpy as np
import os
from pathlib import Path


def rotate_yolo_data(images_folder, labels_folder, augment_set, gray=False):
    """
    对YOLO数据集进行旋转增强，并可选生成灰度图

    参数:
        images_folder: 图片文件夹路径
        labels_folder: 标签文件夹路径
        augment_set: 旋转角度列表，如 [60, 90, 120]
        gray: 是否生成灰度图（默认为False）
    """
    images_folder = Path(images_folder)
    labels_folder = Path(labels_folder)

    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_folder.glob(f'*{ext}')))
        image_files.extend(list(images_folder.glob(f'*{ext.upper()}')))

    print(f"找到 {len(image_files)} 张图片")

    for img_path in image_files:
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告: 无法读取图片 {img_path}")
            continue

        h, w = img.shape[:2]

        # 查找对应的标签文件
        label_path = labels_folder / f"{img_path.stem}.txt"

        # 读取标签（如果存在）
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = parts[0]
                        x_center, y_center, width, height = map(float, parts[1:5])
                        boxes.append([class_id, x_center, y_center, width, height])

        # 对每个旋转角度进行增强
        for angle in augment_set:
            # 旋转图片
            rotated_img, M = rotate_image(img, angle)

            # 生成新文件名
            new_img_name = f"{img_path.stem}-{angle}{img_path.suffix}"
            new_img_path = images_folder / new_img_name

            # 保存旋转后的图片
            cv2.imwrite(str(new_img_path), rotated_img)

            # 如果有标签，旋转标签
            if boxes:
                new_boxes = rotate_yolo_boxes(boxes, M, w, h, rotated_img.shape[1], rotated_img.shape[0])

                # 保存新标签
                new_label_name = f"{img_path.stem}-{angle}.txt"
                new_label_path = labels_folder / new_label_name

                with open(new_label_path, 'w') as f:
                    for box in new_boxes:
                        f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

            print(f"已生成: {new_img_name}")

        # --------- 灰度图部分 ---------
        if gray:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_name = f"{img_path.stem}-gray{img_path.suffix}"
            gray_path = images_folder / gray_name
            cv2.imwrite(str(gray_path), gray_img)

            # 标签不变，直接复制
            if label_path.exists():
                gray_label = labels_folder / f"{img_path.stem}-gray.txt"
                gray_label.write_text(label_path.read_text(), encoding="utf-8")
            print(f"🩶 已生成灰度图: {gray_name}")

    print("数据增强完成！")


def rotate_image(image, angle):
    """
    旋转图片

    参数:
        image: 输入图片
        angle: 旋转角度（顺时针）

    返回:
        rotated_image: 旋转后的图片
        M: 旋转矩阵
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # 计算旋转后图片的尺寸
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵以适应新尺寸
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 执行旋转
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(114, 114, 114))

    return rotated, M


def rotate_yolo_boxes(boxes, M, orig_w, orig_h, new_w, new_h):
    """
    使用旋转矩阵 M 旋转 YOLO 格式的边界框坐标
    """
    new_boxes = []

    for box in boxes:
        class_id = box[0]
        x_center_norm, y_center_norm, width_norm, height_norm = box[1:5]

        # 转为绝对坐标
        x_center = x_center_norm * orig_w
        y_center = y_center_norm * orig_h
        width = width_norm * orig_w
        height = height_norm * orig_h

        # 四个角点
        corners = np.array([
            [x_center - width / 2, y_center - height / 2],
            [x_center + width / 2, y_center - height / 2],
            [x_center + width / 2, y_center + height / 2],
            [x_center - width / 2, y_center + height / 2]
        ])

        # 添加一列1用于矩阵乘法
        ones = np.ones((4, 1))
        corners_hom = np.hstack([corners, ones])

        # 应用旋转矩阵 M
        rotated_corners = np.dot(M, corners_hom.T).T

        # 计算新的框
        x_min = np.min(rotated_corners[:, 0])
        x_max = np.max(rotated_corners[:, 0])
        y_min = np.min(rotated_corners[:, 1])
        y_max = np.max(rotated_corners[:, 1])

        # 边界限制
        x_min = np.clip(x_min, 0, new_w)
        x_max = np.clip(x_max, 0, new_w)
        y_min = np.clip(y_min, 0, new_h)
        y_max = np.clip(y_max, 0, new_h)

        # 转回YOLO格式
        new_x_center = ((x_min + x_max) / 2) / new_w
        new_y_center = ((y_min + y_max) / 2) / new_h
        new_width = ((x_max - x_min) / new_w)*0.92
        new_height = ((y_max - y_min) / new_h)*0.92

        # 过滤过小框
        if new_width > 0.01 and new_height > 0.01:
            new_boxes.append([class_id, new_x_center, new_y_center, new_width, new_height])

    return new_boxes


# 使用示例
if __name__ == "__main__":
    images_folder = r"D:\dataset\phone-628\images"
    labels_folder = r"D:\dataset\phone-628\labels"
    augment_set = [72, 90, 105]

    # 执行数据增强，生成灰度图
    rotate_yolo_data(images_folder, labels_folder, augment_set, gray=True)

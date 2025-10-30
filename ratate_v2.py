import cv2
import numpy as np
from pathlib import Path


def rotate_yolo_data(images_folder, labels_folder, augment_set, gray=False):
    """
    对YOLO数据集进行旋转增强（优化版：更精确的框旋转 + 可选灰度图生成）

    参数:
        images_folder: 图片文件夹路径
        labels_folder: 标签文件夹路径
        augment_set: 旋转角度列表，如 [60, 90, 120]
        gray: 是否生成原图灰度版本（默认False）
    """
    images_folder = Path(images_folder)
    labels_folder = Path(labels_folder)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_folder.glob(f'*{ext}'))
        image_files.extend(images_folder.glob(f'*{ext.upper()}'))

    print(f"📂 找到 {len(image_files)} 张图片")

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ 无法读取图片: {img_path}")
            continue

        h, w = img.shape[:2]

        # 读取标签
        label_path = labels_folder / f"{img_path.stem}.txt"
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = parts[0]
                        x, y, bw, bh = map(float, parts[1:5])
                        boxes.append([cls, x, y, bw, bh])

        # --------- 旋转增强部分 ---------
        for angle in augment_set:
            rotated_img, M, new_w, new_h = rotate_image(img, angle)
            new_img_name = f"{img_path.stem}-{angle}{img_path.suffix}"
            new_img_path = images_folder / new_img_name

            cv2.imwrite(str(new_img_path), rotated_img)

            if boxes:
                new_boxes = rotate_yolo_boxes_precise(boxes, M, w, h, new_w, new_h)
                new_label_path = labels_folder / f"{img_path.stem}-{angle}.txt"
                with open(new_label_path, "w") as f:
                    for b in new_boxes:
                        f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

            print(f"✅ 已生成旋转增强图: {new_img_name}")

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

    print("🎉 数据增强完成！")


def rotate_image(image, angle):
    """旋转图片并返回旋转矩阵与新尺寸"""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)

    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(114, 114, 114))
    return rotated, M, new_w, new_h


def rotate_yolo_boxes_precise(boxes, M, orig_w, orig_h, new_w, new_h):
    """更精确的 YOLO 框旋转（使用最小外接矩形重新计算）"""
    new_boxes = []

    for cls, x, y, bw, bh in boxes:
        # 转为绝对坐标
        x *= orig_w
        y *= orig_h
        bw *= orig_w
        bh *= orig_h

        # 四角坐标
        pts = np.array([
            [x - bw / 2, y - bh / 2],
            [x + bw / 2, y - bh / 2],
            [x + bw / 2, y + bh / 2],
            [x - bw / 2, y + bh / 2]
        ])

        # 旋转变换
        pts = np.hstack([pts, np.ones((4, 1))])
        rotated_pts = np.dot(M, pts.T).T

        # 使用最小外接矩形提高准确度
        rect = cv2.minAreaRect(rotated_pts.astype(np.float32))
        (cx, cy), (rw, rh), _ = rect

        # 归一化并裁剪
        nx = np.clip(cx / new_w, 0, 1)
        ny = np.clip(cy / new_h, 0, 1)
        nw = np.clip(rw / new_w, 0, 1)
        nh = np.clip(rh / new_h, 0, 1)

        # 过滤掉太小的框
        if nw > 0.01 and nh > 0.01:
            new_boxes.append([cls, nx, ny, nw, nh])

    return new_boxes


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    images_folder = r"D:\dataset\phone-628\images"
    labels_folder = r"D:\dataset\phone-628\labels"
    augment_set = [60, 90, 120]
    rotate_yolo_data(images_folder, labels_folder, augment_set, gray=True)

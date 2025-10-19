import os
import json
import glob

# 路径配置
json_dir = '/data/solutions/cloth/newcloth/newdata1/jsons'   # LabelMe 导出的 JSON 文件夹
image_dir = '/data/solutions/cloth/newcloth/newdata1/images'   # 可选：仅用于检查图片是否存在
output_label_dir = '/data/solutions/cloth/newcloth/newdata1/labels'
os.makedirs(output_label_dir, exist_ok=True)

# 类别映射（按你的标签名来）
class_map = {
    'clothes': 0,
}

def yolo_line_from_points(points, img_w, img_h):
    """将任意点集转成 YOLO 的 (x_center, y_center, w, h)，已做归一化与边界裁剪。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = max(0.0, min(xs)), min(float(img_w), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(img_h), max(ys))

    # 防止标注点越界或相等导致零宽/零高
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    bbox_w   = bw / img_w
    bbox_h   = bh / img_h

    # 再次裁剪到 [0,1]
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    bbox_w   = min(max(bbox_w, 0.0), 1.0)
    bbox_h   = min(max(bbox_h, 0.0), 1.0)
    return x_center, y_center, bbox_w, bbox_h

# 获取所有 JSON 文件
json_files = glob.glob(os.path.join(json_dir, '*.json'))

for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1) 优先使用 LabelMe JSON 自带的尺寸，避免依赖 cv2
    img_w = data.get('imageWidth', None)
    img_h = data.get('imageHeight', None)

    # 兜底：如果 JSON 里没有尺寸，再尝试从图片探测（可选，默认不做）
    if img_w is None or img_h is None:
        # 如果确实需要兜底，可以用 PIL 代替 cv2：
        try:
            from PIL import Image
            image_filename = data['imagePath']
            image_path = os.path.join(image_dir, image_filename)
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except Exception as e:
            print(f"❌ 无法获取图像尺寸（{json_file}）：{e}")
            continue

    label_lines = []
    unknown_labels = set()

    for shape in data.get('shapes', []):
        label = shape.get('label')
        points = shape.get('points', [])

        if label not in class_map:
            unknown_labels.add(label)
            continue  # 忽略未知标签

        # LabelMe 既可能是 polygon，也可能是 rectangle 的两个点，这里统一按外接框处理
        if not points or len(points) < 2:
            # 点数太少，跳过该标注
            continue

        x_center, y_center, bbox_w, bbox_h = yolo_line_from_points(points, img_w, img_h)
        class_id = class_map[label]
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"
        label_lines.append(line)

    # 保存 .txt（与图片同名，如果 imagePath 缺失，则用 json 文件名）
    image_filename = data.get('imagePath', os.path.basename(json_file).replace('.json', ''))
    base_name = os.path.splitext(os.path.basename(image_filename))[0]
    output_txt = os.path.join(output_label_dir, base_name + '.txt')

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(label_lines))

    msg = f"✅ 转换完成：{output_txt}"
    if unknown_labels:
        msg += f"（忽略未知标签：{sorted(list(unknown_labels))}）"
    print(msg)

print("🎉 所有标签已转换为 YOLO 格式")

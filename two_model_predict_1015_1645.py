#!/usr/bin/env python3
# coding: utf-8

"""
ensemble_detect.py

功能：
 - 用 model1.pt 和 model2.pt 对 test_images 中图片逐张推理（conf=0.0001）
 - 对同类框用 IoU 阈值做合并（按 score 做 greedy NMS），避免重复
 - 输出 COCO-style JSON（在 result_sample.json 基础上添加 annotations 字段）
 - 在图片上用绿色框绘制合并后的检测结果并保存
 - 为每张图片生成裁剪（局部放大）以便人工检查
 - 生成 result_report.txt（每张图片一行，按要求格式）

注意：请把 model1.pt、model2.pt 和 result_sample.json 放在脚本同目录或调整路径变量
"""

import os
import json
import math
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

# -----------------------
# Config
# -----------------------
MODEL1 = "/app/competition/20251009_0942/aerial_finetune/weights/best.pt"
MODEL2 = "/app/competition/map50_optimized_detection_20250925_0125/yolov8x_map50_optimized_20250925_0125/weights/best.pt"
RESULT_SAMPLE_JSON = "result_sample.json"   # 比赛给的样板（images, categories 已有）
TEST_IMAGE_DIR = "test_images"
OUTPUT_JSON = "result_final.json"
REPORT_TXT = "result_report.txt"
OUTPUT_DIR = Path("output")
MERGED_BOX_IMG_DIR = OUTPUT_DIR / "merged_boxes"
CROPS_DIR = OUTPUT_DIR / "crops"

CONF_THRESH = 0.0001            # 比赛要求
IOU_MERGE_THRESH = 0.5          # 合并同类框时 IoU 阈值（可调整）
MIN_BOX_AREA = 1                # 过滤极小框
MAX_CROP_WH = 800               # 裁剪保存时最大宽/高（如果需要缩放）

# 重要：模型类别到比赛类别的映射（如果模型内类别是 0..3 且顺序 ship, people, car, motor）
# 如果你的模型类别顺序不同，请修改这个字典，例如模型类 0 对应比赛类别 2 等
MODEL_CLASS_TO_COCO = {
    0: 1,  # model class 0 -> coco category id 1 (ship)
    1: 2,  # -> people
    2: 3,  # -> car
    3: 4,  # -> motor
}

# 类别 id 到名称（和 result_sample.json 中 categories 一致）
COCO_ID_TO_NAME = {
    1: "ship",
    2: "people",
    3: "car",
    4: "motor",
}

# -----------------------
# Helpers: IoU, NMS, bbox conversions
# -----------------------
def xyxy_to_xywh(xyxy):
    # xyxy = [x1, y1, x2, y2]
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [x1, y1, w, h]

def area_xywh(xywh):
    return xywh[2] * xywh[3]

def compute_iou(boxA, boxB):
    # boxes in xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))

    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom

def greedy_nms(boxes_xyxy, scores, iou_thresh):
    """
    Greedy NMS for a single class.
    boxes_xyxy: Nx4 numpy
    scores: N
    returns indices kept
    """
    if len(boxes_xyxy) == 0:
        return []

    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = np.array([compute_iou(boxes_xyxy[current], boxes_xyxy[i]) for i in rest])
        idxs = rest[ious <= iou_thresh]
    return keep

# -----------------------
# Create output dirs
# -----------------------
os.makedirs(MERGED_BOX_IMG_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Load models
# -----------------------
print("Loading models...")
model1 = YOLO(MODEL1_PATH)
model2 = YOLO(MODEL2_PATH)

# -----------------------
# Load sample json and iterate images
# -----------------------
with open(RESULT_SAMPLE_JSON, "r", encoding="utf-8") as f:
    sample = json.load(f)

images_list = sample.get("images", [])
categories = sample.get("categories", [])
if not images_list:
    raise RuntimeError("result_sample.json 中没有 images 字段或为空，请检查路径和文件内容。")

annotations = []
ann_id = 1  # coco annotation id start

report_lines = []

# For per-image counting of model outputs (before merge)
for img_info in tqdm(images_list, desc="Images"):
    file_name = img_info["file_name"]
    image_id = img_info["id"]
    image_path = os.path.join(TEST_IMAGE_DIR, file_name)
    if not os.path.exists(image_path):
        print(f"[WARN] image missing: {image_path}, 跳过")
        # still append empty line to report
        report_lines.append(f"图片{file_name}:model1的检测结果{{ship：0，people：0，car：0，motor：0}}，model2的检测结果{{ship：0，people：0，car：0，motor：0}}，填入result_sample.json的结果是{{ship：0，people：0，car：0，motor：0}}")
        continue

    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]

    # -----------------------
    # run model1 and model2
    # -----------------------
    # 使用 ultralytics 的 predict 接口，设置 conf
    preds1 = model1.predict(source=image_path, conf=CONF_THRESH, verbose=False)[0]
    preds2 = model2.predict(source=image_path, conf=CONF_THRESH, verbose=False)[0]

    # preds.boxes.xyxy, preds.boxes.cls, preds.boxes.conf
    def extract(pred):
        boxes = []
        if hasattr(pred, "boxes") and len(pred.boxes) > 0:
            boxes_xyxy = pred.boxes.xyxy.cpu().numpy()  # shape Nx4
            scores = pred.boxes.conf.cpu().numpy()
            classes = pred.boxes.cls.cpu().numpy().astype(int)
            for b, s, c in zip(boxes_xyxy, scores, classes):
                # map class
                coco_cid = MODEL_CLASS_TO_COCO.get(int(c), None)
                if coco_cid is None:
                    # skip unknown class
                    continue
                # clamp coords
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                x1 = max(0.0, min(x1, w_img-1))
                x2 = max(0.0, min(x2, w_img-1))
                y1 = max(0.0, min(y1, h_img-1))
                y2 = max(0.0, min(y2, h_img-1))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append({
                    "xyxy": [x1, y1, x2, y2],
                    "score": float(s),
                    "category_id": int(coco_cid)
                })
        return boxes

    boxes1 = extract(preds1)
    boxes2 = extract(preds2)

    # count per class for report
    def count_by_class(boxes):
        counts = {1:0,2:0,3:0,4:0}
        for b in boxes:
            counts[b["category_id"]] += 1
        return counts
    c1 = count_by_class(boxes1)
    c2 = count_by_class(boxes2)

    # -----------------------
    # Merge: for each category separately, combine boxes from both models and NMS them
    # -----------------------
    merged_boxes = []  # list of dict: xyxy, score, category_id
    for cat_id in [1,2,3,4]:
        # collect boxes of this cat from both models
        cat_boxes = [b for b in boxes1 if b["category_id"]==cat_id] + [b for b in boxes2 if b["category_id"]==cat_id]
        if len(cat_boxes) == 0:
            continue
        boxes_xyxy = np.array([b["xyxy"] for b in cat_boxes])
        scores = np.array([b["score"] for b in cat_boxes])
        keep_idx = greedy_nms(boxes_xyxy, scores, IOU_MERGE_THRESH)
        for i in keep_idx:
            merged_boxes.append({
                "xyxy": cat_boxes[i]["xyxy"],
                "score": float(cat_boxes[i]["score"]),
                "category_id": cat_id
            })

    # -----------------------
    # Filter tiny boxes
    # -----------------------
    filtered = []
    for b in merged_boxes:
        xywh = xyxy_to_xywh(b["xyxy"])
        if area_xywh(xywh) < MIN_BOX_AREA:
            continue
        filtered.append(b)
    merged_boxes = filtered

    # -----------------------
    # Convert merged boxes to coco annotations and append
    # -----------------------
    for b in merged_boxes:
        xyxy = b["xyxy"]
        xywh = xyxy_to_xywh(xyxy)
        area = area_xywh(xywh)
        ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": int(b["category_id"]),
            "bbox": [float("{:.2f}".format(x)) for x in xywh],  # keep 2 decimals
            "area": float("{:.2f}".format(area)),
            "iscrowd": 0,
            "score": float("{:.4f}".format(b["score"]))
        }
        annotations.append(ann)
        ann_id += 1

    # -----------------------
    # Draw merged boxes on image and save
    # -----------------------
    out_img = img.copy()
    for b in merged_boxes:
        x1, y1, x2, y2 = [int(round(x)) for x in b["xyxy"]]
        # green color (BGR)
        color = (0, 255, 0)
        thickness = max(2, int(round(0.002 * max(w_img, h_img))))  # scale thickness
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, thickness)
        label = f"{COCO_ID_TO_NAME[b['category_id']]}:{b['score']:.2f}"
        # put label
        ((w_label, h_label), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out_img, (x1, y1 - h_label - 6), (x1 + w_label + 6, y1), color, -1)
        cv2.putText(out_img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

    save_img_path = MERGED_BOX_IMG_DIR / file_name
    cv2.imwrite(str(save_img_path), out_img)

    # -----------------------
    # Save crops (局部放大) 便于仔细检查
    # -----------------------
    for idx, b in enumerate(merged_boxes, start=1):
        x1, y1, x2, y2 = [int(round(x)) for x in b["xyxy"]]
        pad_w = int(0.1 * (x2 - x1) + 10)
        pad_h = int(0.1 * (y2 - y1) + 10)
        xa = max(0, x1 - pad_w)
        ya = max(0, y1 - pad_h)
        xb = min(w_img-1, x2 + pad_w)
        yb = min(h_img-1, y2 + pad_h)
        crop = img[ya:yb, xa:xb]
        # resize if too large
        ch, cw = crop.shape[:2]
        scale = 1.0
        if max(ch, cw) > MAX_CROP_WH:
            scale = MAX_CROP_WH / max(ch, cw)
            crop = cv2.resize(crop, (int(cw*scale), int(ch*scale)))
        crop_path = CROPS_DIR / f"{Path(file_name).stem}_ann{idx}_cat{b['category_id']}_score{b['score']:.3f}.jpg"
        cv2.imwrite(str(crop_path), crop)

    # -----------------------
    # Build report line
    # -----------------------
    def counts_to_brace(counts):
        return f"{{ship：{counts[1]}，people：{counts[2]}，car：{counts[3]}，motor：{counts[4]}}}"

    merged_counts = {1:0,2:0,3:0,4:0}
    for b in merged_boxes:
        merged_counts[b["category_id"]] += 1

    line = f"图片{file_name}:model1的检测结果{counts_to_brace(c1)}，model2的检测结果{counts_to_brace(c2)}，填入result_sample.json的结果是{counts_to_brace(merged_counts)}"
    report_lines.append(line)

# -----------------------
# Write final result json
# -----------------------
out = {
    "categories": categories,
    "images": images_list,
    "annotations": annotations
}
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# -----------------------
# Write report txt
# -----------------------
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    for l in report_lines:
        f.write(l + "\n")

print("Done.")
print(f"Result json: {OUTPUT_JSON}")
print(f"Report: {REPORT_TXT}")
print(f"Merged images saved to: {MERGED_BOX_IMG_DIR}")
print(f"Crops saved to: {CROPS_DIR}")

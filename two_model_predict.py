#!/usr/bin/env python3
"""
ensemble_infer.py

功能：
- 加载 model1.pt, model2.pt（Ultralytics YOLO 格式）
- 对 test_images 文件夹中每张图片推理（conf=0.0001）
- 按类别融合两模型的检测（IoU 聚类 + 分数加权平均）
- 输出 COCO 格式 JSON（基于 result_sample.json，填入 annotations 字段）
- 输出 result_report.txt（每张图片一行：model1 counts, model2 counts, merged counts）
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import math

# 引入 ultralytics YOLO（YOLOv8）
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("需要安装 ultralytics（pip install ultralytics）来加载 .pt 模型。") from e

# torchvision 用于计算 IoU 的工具（如果可用）
try:
    import torch
    from torchvision.ops import box_iou
except Exception:
    # 如果没有 torchvision 的 box_iou，用 numpy 自实现（速度可能慢）
    torch = None
    box_iou = None

# ---------- 配置 ----------
MODEL1 = "/app/competition/20251009_0942/aerial_finetune/weights/best.pt"
MODEL2 = "/app/competition/map50_optimized_detection_20250925_0125/yolov8x_map50_optimized_20250925_0125/weights/best.pt"
TEST_IMAGES_DIR = "test_images"
RESULT_SAMPLE_JSON = "result_sample.json"  # 输入：含 images & categories
OUTPUT_JSON = "result_1015_1532.json"                # 输出提交文件（含 annotations）
OUTPUT_REPORT = "result_report_1015_1532.txt"
CONF_THRESHOLD = 0.0001
# 合并同类框时 IoU 阈值（若 >= 合并）
MERGE_IOU_THRESH = 0.85
# 类别映射（COCO 要求严格：1-ship，2-people，3-car，4-motor）
CAT_ID_TO_NAME = {1: "ship", 2: "people", 3: "car", 4: "motor"}
# -------------------------

def ensure_models_exist():
    for p in (MODEL1, MODEL2):
        if not os.path.exists(p):
            raise FileNotFoundError(f"模型文件未找到: {p}")

def load_models():
    print("Loading models...")
    m1 = YOLO(MODEL1)
    m2 = YOLO(MODEL2)
    print("Models loaded.")
    return m1, m2

def predict_one(model, img_path, conf_th=CONF_THRESHOLD):
    """
    使用 ultralytics YOLO 推理，返回 list of dicts:
    [{'xyxy': [x1,y1,x2,y2], 'cls': int_class_id, 'conf': float}, ...]
    class ids expected: 0.. (we will map to 1..4 by +1 if needed)
    """
    # 使用原始图片尺寸推理（保持大小）
    results = model.predict(source=str(img_path), conf=conf_th, verbose=False, imgsz=1984, device=0)  # device None -> 自动
    # results 是 list，取第0
    if len(results) == 0:
        return []
    r = results[0]
    # r.boxes: ultralytics Boxes object
    boxes = []
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        for b in r.boxes:
            # 获取 numpy 数组。 ultralytics Boxes -> .xyxy, .conf, .cls
            xyxy = b.xyxy.numpy().cpu().tolist()[0] if hasattr(b.xyxy, "numpy") else b.xyxy.tolist()
            conf = float(b.conf.cpu().tolist()[0]) if hasattr(b.conf, "tolist") else float(b.conf)
            cls = int(b.cls.cpu().tolist()[0]) if hasattr(b.cls, "tolist") else int(b.cls)
            boxes.append({'xyxy': xyxy, 'cls': cls, 'conf': conf})
    return boxes

def xyxy_to_xywh(xyxy):
    x1,y1,x2,y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [float(x1), float(y1), float(w), float(h)]

def compute_iou_np(boxes1, boxes2):
    # boxes: Nx4 in xyxy, numpy
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=float)
    xA = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    yA = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    xB = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    yB = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    interW = np.maximum(0.0, xB - xA)
    interH = np.maximum(0.0, yB - yA)
    interArea = interW * interH
    area1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
    area2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])
    union = area1[:,None] + area2[None,:] - interArea
    iou = np.zeros_like(interArea)
    valid = union > 0
    iou[valid] = interArea[valid] / union[valid]
    return iou

def merge_per_class(boxes_list, scores_list, iou_thresh=MERGE_IOU_THRESH):
    """
    boxes_list: list of (N_i x 4) arrays in xyxy (float)
    scores_list: list of (N_i,) arrays
    Merge all boxes in boxes_list into clusters by IoU >= iou_thresh, do weighted average by score.
    Returns merged_boxes (M x 4) and merged_scores (M,)
    """
    # concatenate
    if len(boxes_list) == 0:
        return np.zeros((0,4)), np.zeros((0,))
    boxes = np.vstack(boxes_list) if len(boxes_list)>0 else np.zeros((0,4))
    scores = np.hstack(scores_list) if len(scores_list)>0 else np.zeros((0,))
    if boxes.shape[0] == 0:
        return boxes, scores

    # We'll do greedy clustering: pick highest score, find all boxes with IoU >= thresh, merge them
    remaining_idx = list(range(boxes.shape[0]))
    merged_boxes = []
    merged_scores = []

    # Precompute iou matrix (numpy)
    iou_mat = compute_iou_np(boxes, boxes)

    while remaining_idx:
        # pick index with highest score among remaining
        rem_scores = scores[remaining_idx]
        idx_in_rem = int(np.argmax(rem_scores))
        base_idx = remaining_idx[idx_in_rem]
        # find cluster: those remaining j where IoU(base_idx, j) >= thresh
        cluster = [j for j in remaining_idx if iou_mat[base_idx, j] >= iou_thresh]
        # weighted average coordinates by score
        cluster_scores = scores[cluster]
        if cluster_scores.sum() <= 0:
            weights = np.ones_like(cluster_scores) / len(cluster_scores)
        else:
            weights = cluster_scores / cluster_scores.sum()
        pts = boxes[cluster]  # K x 4
        # compute weighted mean of [x1,y1,x2,y2]
        merged = (weights[:,None] * pts).sum(axis=0)
        merged_score = float(cluster_scores.max())  # preserve top score (or use average: cluster_scores.mean())
        merged_boxes.append(merged.tolist())
        merged_scores.append(merged_score)
        # remove cluster indices from remaining
        remaining_idx = [r for r in remaining_idx if r not in cluster]

    return np.array(merged_boxes, dtype=float), np.array(merged_scores, dtype=float)


def merge_detections(dets1, dets2, merge_iou=MERGE_IOU_THRESH):
    """
    dets1/dets2: list of dicts {'xyxy': [x1,y1,x2,y2], 'cls': int, 'conf': float}
    cls in these dicts may start from 0, map to 1..4 later if needed.
    返回 merged list 同样格式，cls 保持为目标类别（1..4）
    """
    # Group by class (we'll map classes to 1..4)
    all_cls = {}
    for det in dets1:
        cls = int(det['cls'])
        all_cls.setdefault(cls, []).append(det)
    for det in dets2:
        cls = int(det['cls'])
        all_cls.setdefault(cls, []).append(det)

    merged = []
    for cls, dets in all_cls.items():
        boxes = np.array([d['xyxy'] for d in dets], dtype=float)
        scores = np.array([d['conf'] for d in dets], dtype=float)
        # To align with our merge_per_class API, pass as two lists (from two models)
        # We'll just pass all as single list
        mb, ms = merge_per_class([boxes], [scores], iou_thresh=merge_iou)
        for b, s in zip(mb, ms):
            merged.append({'xyxy': b.tolist(), 'cls': int(cls), 'conf': float(s)})
    return merged


def preds_to_counts(dets):
    counts = {1:0,2:0,3:0,4:0}
    for d in dets:
        cls = int(d['cls'])
        # some models output 0-based classes: try to auto-handle (if class 0-3 then +1)
        if cls in (0,1,2,3) and cls not in counts:
            # convert 0->1,1->2,...
            cls = cls + 1
        if cls in counts:
            counts[cls] += 1
    return counts

def convert_cls_to_submit_cls(cls):
    """
    If model outputs 0..3, convert to 1..4. If already 1..4 keep.
    """
    cls = int(cls)
    if cls in (0,1,2,3):
        return cls + 1
    return cls

def main():
    ensure_models_exist()
    m1, m2 = load_models()

    # load sample json to preserve images & categories
    if not os.path.exists(RESULT_SAMPLE_JSON):
        raise FileNotFoundError(f"找不到 {RESULT_SAMPLE_JSON}，请将比赛给出的 sample json 放到当前目录并命名为 {RESULT_SAMPLE_JSON}")
    with open(RESULT_SAMPLE_JSON, 'r', encoding='utf-8') as f:
        sample = json.load(f)
    images_info = {img['file_name']: img for img in sample.get('images', [])}
    # We'll iterate over sample images to keep order and use their IDs
    image_items = sample.get('images', [])

    annotations = []
    ann_id = 1
    report_lines = []

    for img_info in tqdm(image_items, desc="Images"):
        file_name = img_info['file_name']
        img_path = os.path.join(TEST_IMAGES_DIR, file_name)
        if not os.path.exists(img_path):
            # 如果图片不存在，跳过（也可以警告）
            print(f"警告：图片不存在 {img_path}，跳过")
            continue

        # predict with both models
        dets1 = predict_one(m1, img_path, conf_th=CONF_THRESHOLD)  # cls likely 0..3
        dets2 = predict_one(m2, img_path, conf_th=CONF_THRESHOLD)

        # normalize class ids to 1..4 for counts and merging: but keep original for merging function which handles 0-based
        # Create copies with cls normalized for final output
        # Count each model's raw predictions mapping to 1..4
        counts1 = preds_to_counts(dets1)
        counts2 = preds_to_counts(dets2)

        # Merge detections
        merged = merge_detections(dets1, dets2, merge_iou=MERGE_IOU_THRESH)
        merged_counts = preds_to_counts(merged)

        # prepare annotations entries from merged results
        # Use image id from sample json
        image_id = img_info['id']
        for d in merged:
            cls_out = convert_cls_to_submit_cls(d['cls'])  # ensure 1..4
            xyxy = d['xyxy']
            xywh = xyxy_to_xywh(xyxy)
            area = float(xywh[2] * xywh[3])
            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(cls_out),
                "bbox": [float(round(v, 3)) for v in xywh],
                "area": float(round(area,3)),
                "iscrowd": 0,
                "score": float(round(float(d['conf']), 6))
            }
            annotations.append(ann)
            ann_id += 1

        # write report line for this image
        def counts_str(cdict):
            return f"{{ship:{cdict.get(1,0)},people:{cdict.get(2,0)},car:{cdict.get(3,0)},motor:{cdict.get(4,0)}}}"
        line = f"{file_name}: model1的检测结果{counts_str(counts1)}, model2的检测结果{counts_str(counts2)}, 填入result_sample.json的结果是{counts_str(merged_counts)}"
        report_lines.append(line)

    # assemble final json: keep images & categories from sample
    out_json = {
        "images": sample.get('images', []),
        "annotations": annotations,
        "categories": sample.get('categories', [])
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"已保存提交文件: {OUTPUT_JSON}，annotations 数量: {len(annotations)}")

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"已保存报告: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()

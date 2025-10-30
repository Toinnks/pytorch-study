# phone_detect_refine.py
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import random
from typing import List, Tuple

# -------------------------
# Helper utilities
# -------------------------
def generate_colors(num_classes):
    colors = []
    for i in range(num_classes):
        hue = int(180 * i / max(1, num_classes))
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        color = [int(c) for c in color]
        colors.append(color)
    random.shuffle(colors)
    return colors

def xyxy_to_xywh(box: List[float]) -> Tuple[float,float,float,float]:
    x1,y1,x2,y2 = box
    w = x2-x1
    h = y2-y1
    return (x1 + w/2, y1 + h/2, w, h)

def box_area(box: List[float]) -> float:
    x1,y1,x2,y2 = box
    return max(0, x2-x1) * max(0, y2-y1)

def clip_box(box: List[float], w:int, h:int) -> List[float]:
    x1,y1,x2,y2 = box
    x1 = max(0, min(w-1, x1))
    y1 = max(0, min(h-1, y1))
    x2 = max(0, min(w-1, x2))
    y2 = max(0, min(h-1, y2))
    return [x1,y1,x2,y2]

def iou(boxA: List[float], boxB: List[float]) -> float:
    # boxes in xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = box_area(boxA)
    areaB = box_area(boxB)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def map_box_from_crop_to_image(crop_box_xyxy, det_box_xyxy_in_crop, orig_img_shape, crop_size):
    cx1, cy1, cx2, cy2 = crop_box_xyxy
    crop_w = int(cx2 - cx1)
    crop_h = int(cy2 - cy1)
    input_w, input_h = crop_size

    # scaling from model input back to crop original size
    sx = crop_w / input_w
    sy = crop_h / input_h

    dx1, dy1, dx2, dy2 = det_box_xyxy_in_crop
    # map to original image coordinates
    ox1 = dx1 * sx + cx1
    oy1 = dy1 * sy + cy1
    ox2 = dx2 * sx + cx1
    oy2 = dy2 * sy + cy1

    # clip
    H, W = orig_img_shape[:2]
    return clip_box([ox1, oy1, ox2, oy2], W, H)

# -------------------------
# Core: refine detections by local zoom-in
# -------------------------
def refine_detections_on_image(model: YOLO,
                               image: np.ndarray,
                               results,
                               *,
                               conf_threshold=0.1,
                               refine_conf_threshold=0.2,
                               small_area_threshold=0.02,   # 相对于图像面积的阈值，面积小于该值被认为“小目标”
                               enlarge_ratio=2.0,           # 对框外扩的倍数（相对于短边）
                               input_size=(640,640),
                               max_refine_per_image=50,
                               iou_replace_thresh=0.5):
    """
    对初始检测结果中的小目标或低置信度目标进行局部放大复检并合并结果。

    参数:
      model: 已加载的 YOLO 模型
      image: 原图 (H, W, C)
      results: model(image) 得到的结果对象列表（通常是只包含一个元素）
      conf_threshold: 初筛置信度阈值（绘图时使用）
      refine_conf_threshold: 复检中保留的最小置信度
      small_area_threshold: 相对图像面积阈值 (0~1)，小于它的box会被复检
      enlarge_ratio: 放大倍数（例如 2.0 则外扩至2倍短边）
      input_size: 复检时 resize 的尺寸 (w,h)
      max_refine_per_image: 单图最大复检次数，防止太慢
      iou_replace_thresh: 若复检box与原box IoU > 此阈值并且置信度更高则替换
    返回:
      merged_boxes: list of dicts: {'xyxy': [x1,y1,x2,y2], 'conf':float, 'cls':int}
    """
    H, W = image.shape[:2]
    img_area = H * W

    # flatten results (we assume single result)
    dets = []
    for res in results:
        for b in res.boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            conf = float(b.conf[0].cpu().numpy())
            cls = int(b.cls[0].cpu().numpy())
            dets.append({'xyxy': xyxy, 'conf': conf, 'cls': cls})

    # sort detections by confidence asc so low conf small boxes get refined first (optional)
    dets_sorted = sorted(dets, key=lambda x: x['conf'])

    merged = dets[:]  # start with initial detections
    refine_count = 0

    for det in dets_sorted:
        if refine_count >= max_refine_per_image:
            break

        xyxy = det['xyxy']
        conf = det['conf']
        cls = det['cls']
        area = box_area(xyxy)
        rel_area = area / img_area

        # 条件：小目标 或 置信度处于中间区间（例如 [0.05, 0.4]）
        needs_refine = (rel_area < small_area_threshold) or (conf < 0.4 and conf > 0.05)

        if not needs_refine:
            continue

        # 计算外扩 crop 区域（以短边为基准）
        x1,y1,x2,y2 = xyxy
        bw = x2 - x1
        bh = y2 - y1
        short = max(1.0, min(bw, bh))
        pad = (enlarge_ratio - 1.0) * short / 2.0  # 两边总共外扩 (enlarge_ratio-1)*short
        cx1 = int(max(0, x1 - pad))
        cy1 = int(max(0, y1 - pad))
        cx2 = int(min(W-1, x2 + pad))
        cy2 = int(min(H-1, y2 + pad))

        # 若 crop 极小或已超出图像，则跳过
        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            continue

        crop = image[cy1:cy2, cx1:cx2]

        # resize crop to input_size while keeping aspect - but we'll just resize to fixed for simplicity
        resized_crop = cv2.resize(crop, input_size)

        # model inference on crop
        crop_results = model(resized_crop)

        # gather crop detections same class
        crop_dets = []
        for cres in crop_results:
            for cb in cres.boxes:
                c_xyxy = cb.xyxy[0].cpu().numpy().tolist()  # in resized_crop coords
                c_conf = float(cb.conf[0].cpu().numpy())
                c_cls = int(cb.cls[0].cpu().numpy())
                if c_cls != cls:
                    continue
                if c_conf < refine_conf_threshold:
                    continue
                # map to original image coords
                mapped = map_box_from_crop_to_image([cx1,cy1,cx2,cy2], c_xyxy, image.shape, input_size)
                crop_dets.append({'xyxy': mapped, 'conf': c_conf, 'cls': c_cls})

        # if we got refined detections, pick the best (highest conf)
        if crop_dets:
            best = max(crop_dets, key=lambda x: x['conf'])
            # compare with original: 如果 IoU 高且置信度更好，替换；否则如果 IoU 低（表示定位改进），也可以替换
            orig = det
            iou_val = iou(orig['xyxy'], best['xyxy'])
            replace = False
            if best['conf'] > orig['conf'] and iou_val >= iou_replace_thresh:
                replace = True
            elif iou_val < iou_replace_thresh and best['conf'] >= orig['conf'] * 0.9:
                # 定位明显不同且置信度接近则也替换
                replace = True

            if replace:
                # replace in merged list (find by IoU or exact match)
                replaced = False
                for m in merged:
                    if m['cls'] == orig['cls'] and iou(m['xyxy'], orig['xyxy']) > 0.9:
                        m.update(best)
                        replaced = True
                        break
                if not replaced:
                    merged.append(best)

            refine_count += 1

    # 最后做一次简单的 NMS（按 class），以减少重复框
    final = []
    for cls_id in set([m['cls'] for m in merged]):
        cls_boxes = [m for m in merged if m['cls'] == cls_id]
        # convert to arrays for nms
        if not cls_boxes:
            continue
        boxes = np.array([b['xyxy'] for b in cls_boxes])
        scores = np.array([b['conf'] for b in cls_boxes])
        # cv2.dnn.NMSBoxes expects x,y,w,h
        xywh = []
        for x1,y1,x2,y2 in boxes:
            xywh.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
        if len(xywh) == 0:
            continue
        indices = cv2.dnn.NMSBoxes(xywh, scores.tolist(), score_threshold=0.01, nms_threshold=0.5)
        if len(indices) == 0:
            # indices can be an array of shape (N,1) or tuple
            for idx in range(len(cls_boxes)):
                final.append(cls_boxes[idx])
        else:
            # indices may be nested, flatten
            if isinstance(indices, (list, tuple)):
                idxs = [i[0] if isinstance(i, (list, tuple, np.ndarray)) and len(i)>0 else i for i in indices]
            else:
                idxs = indices.flatten().tolist()
            for ii in idxs:
                ii = int(ii)
                if ii < len(cls_boxes):
                    final.append(cls_boxes[ii])

    return final

# -------------------------
# Drawing routine (keeps your style)
# -------------------------
def draw_detections(image, merged_boxes, colors, class_names, conf_threshold=0.25):
    img = image.copy()
    for det in merged_boxes:
        x1,y1,x2,y2 = map(int, det['xyxy'])
        conf = det['conf']
        cls = det['cls']
        if conf < conf_threshold:
            continue
        color = colors[cls % len(colors)]
        label = f"{class_names[cls] if class_names else cls}: {conf:.2f}"
        # bounding box
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        # label background
        tsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(img, (x1, y1 - tsize[1] - 6), (x1 + tsize[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return img

# -------------------------
# Main processing pipeline
# -------------------------
def process_images_with_refinement(model, input_dir, output_dir, class_names=None, conf_threshold=0.1):
    os.makedirs(output_dir, exist_ok=True)
    num_classes = len(model.names) if hasattr(model, 'names') else (len(class_names) if class_names else 1)
    colors = generate_colors(num_classes)
    if class_names is None and hasattr(model, 'names'):
        class_names = [model.names[i] for i in range(len(model.names))]

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in image_exts]
    print(f"Found {len(image_files)} images")

    for i, p in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Processing {p.name}")
        img = cv2.imread(str(p))
        if img is None:
            print("  cannot read image, skip")
            continue

        # 全图推理
        results = model(img)

        # 局部放大复检并合并
        merged = refine_detections_on_image(model, img, results,
                                            conf_threshold=conf_threshold,
                                            refine_conf_threshold=0.15,
                                            small_area_threshold=0.012,  # 小目标阈值 (可调)
                                            enlarge_ratio=3.0,           # 放大到 3x 短边
                                            input_size=(640,640),
                                            max_refine_per_image=40,
                                            iou_replace_thresh=0.45)

        # 绘制并保存
        out_img = draw_detections(img, merged, colors, class_names, conf_threshold=conf_threshold)
        out_path = os.path.join(output_dir, f"detected_{p.name}")
        cv2.imwrite(out_path, out_img)
        print("  saved ->", out_path)

    print("All done. Results saved in:", output_dir)


if __name__ == "__main__":
    model_path = r"C:\Users\26601\Desktop\train-9.pt"  # 替换为你的模型路径
    model = YOLO(model_path)

    input_directory = r"D:\dataset\output_images_gray"
    output_directory = r"D:\dataset\trained-9_de_smoke"
    custom_class_names = ['Open Eye','Closed Eye','Cigarette','Phone','Seatbelt']

    process_images_with_refinement(model, input_directory, output_directory,
                                   class_names=custom_class_names,
                                   conf_threshold=0.6)

import cv2
import numpy as np
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")

# 加载模型
phone_model = YOLO('/data/clearingvehicle/phone/runs/detect/train6/weights/best.pt' )
head_model = YOLO("/data/all/model/head.pt")
conf = 0.6  # 置信度阈值


def box_overlap_area(a, b):
    """计算两个框的重叠区域"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0
    return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)


def is_phone_in_head(phone_box, head_box, threshold=0.4):
    """检查手机是否在头部区域内"""
    phone_area = (phone_box[2] - phone_box[0]) * (phone_box[3] - phone_box[1])
    inter_area = box_overlap_area(phone_box, head_box)
    return (phone_area > 0) and (inter_area / phone_area > threshold)


def test_image(image_path, output_path="test_result.jpg"):
    """
    测试单张图片

    参数:
        image_path: 输入图片路径
        output_path: 输出结果图片路径（带检测框）
    """
    print(f"正在读取图片: {image_path}")

    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ 无法读取图片: {image_path}")
        return

    print(f"✅ 图片读取成功，尺寸: {frame.shape}")

    # 推理检测
    print("正在进行目标检测...")
    results_head_all = head_model.predict(frame, conf=conf, verbose=True)
    print(results_head_all)
    results_head=results_head_all[0]
    results_phone = phone_model.predict(frame, conf=conf, verbose=True)[0]

    # 提取检测框
    heads = []
    phones = []
    if hasattr(results_head, "boxes") and results_head.boxes is not None:
        heads = results_head.boxes.xyxy.cpu().numpy().tolist()
    if hasattr(results_phone, "boxes") and results_phone.boxes is not None:
        phones = results_phone.boxes.xyxy.cpu().numpy().tolist()

    print(f"检测结果: 人头={len(heads)}个, 手机={len(phones)}个")

    # 在图片上绘制所有检测框（绿色）
    frame_display = frame.copy()
    for head_box in heads:
        hx1, hy1, hx2, hy2 = map(int, head_box)
        cv2.rectangle(frame_display, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
        cv2.putText(frame_display, "HEAD", (hx1, max(0, hy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for phone_box in phones:
        px1, py1, px2, py2 = map(int, phone_box)
        cv2.rectangle(frame_display, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(frame_display, "PHONE", (px1, max(0, py1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 判定逻辑
    calling_detected = False
    alarm_count = 0

    for idx_h, head_box in enumerate(heads):
        hx1, hy1, hx2, hy2 = head_box
        head_w = hx2 - hx1
        head_h = hy2 - hy1
        head_area = head_w * head_h

        # 定义耳朵和脸部区域
        ry1 = hy1 - head_h * 0.1
        ry2 = hy2 + head_h * 0.2
        rx1 = hx2 - head_w * 0.2
        rx2 = hx2 + head_w * 0.6
        lx1 = hx1 - head_w * 0.6
        lx2 = hx1 + head_w * 0.2
        ly1 = hy1 + head_h * 0.2
        ly2 = hy2 + head_h * 0.4

        for phone_box in phones:
            px1, py1, px2, py2 = phone_box
            cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0

            # 计算手机框面积
            phone_w = px2 - px1
            phone_h = py2 - py1
            phone_area = phone_w * phone_h

            print(f"\n--- 人头{idx_h + 1} vs 手机 ---")
            print(f"  人头框面积: {head_area:.1f} 像素²")
            print(f"  手机框面积: {phone_area:.1f} 像素²")

            # ⭐ 新增判断：人头框必须大于手机框
            if head_area <= phone_area:
                print(f"  ❌ 人头框 <= 手机框，跳过")
                continue
            else:
                print(f"  ✅ 人头框 > 手机框，继续判断")

            # 计算重叠比例
            overlap_ratio = box_overlap_area(
                (px1, py1, px2, py2), (hx1, hy1, hx2, hy2)
            ) / (max((px2 - px1) * (py2 - py1), 1e-6))

            print(f"  重叠比例: {overlap_ratio:.2%}")

            # 判断是否在耳朵区域
            in_ear = (
                    ((rx1 <= cx <= rx2 and ry1 <= cy <= ry2) or
                     (lx1 <= cx <= lx2 and ly1 <= cy <= ly2)) and
                    overlap_ratio > 0.2 and
                    cy < hy2
            )

            # 判断是否在脸部区域
            in_face = is_phone_in_head(
                (px1, py1, px2, py2), (hx1, hy1, hx2, hy2)
            ) and cy < hy2 + head_h * 0.2

            print(f"  在耳朵区域: {in_ear}")
            print(f"  在脸部区域: {in_face}")

            if in_ear or in_face:
                print(f"  🚨 检测到打电话行为！")
                calling_detected = True
                alarm_count += 1

                # 在结果图上用红色标记报警的手机框
                cv2.rectangle(frame_display, (int(px1), int(py1)),
                              (int(px2), int(py2)), (0, 0, 255), 3)
                cv2.putText(frame_display, "CALLING!", (int(px1), max(0, int(py1) - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                break

    # 输出最终结果
    print("\n" + "=" * 50)
    if calling_detected:
        print(f"🚨 检测结果: 发现打电话行为! (共{alarm_count}次)")
        # 在图片顶部添加警告文字
        cv2.putText(frame_display, f"ALARM: Phone Calling Detected! ({alarm_count})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        print("✅ 检测结果: 未发现打电话行为")
        cv2.putText(frame_display, "No Calling Detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print("=" * 50)

    # 保存结果图片
    cv2.imwrite(output_path, frame_display)
    print(f"\n✅ 结果已保存到: {output_path}")

    return calling_detected


if __name__ == "__main__":
    image_path = "A9U358-d89380.png"
    output_path="test_result.jpg"
    test_image(image_path, output_path)

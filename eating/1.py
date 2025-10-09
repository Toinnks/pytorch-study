import cv2
import os
from ultralytics import YOLO

print("当前运行目录是：",os.getcwd())

# 文件路径
input_folder = "/data/clearingvehicle/eating/test/images"  # 输入图片的文件夹路径
output_folder = "/data/clearingvehicle/eating/test/output"  # 输出文件夹路径

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # 处理图片文件
        image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, f"processed_{filename}")

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法打开图片文件: {image_path}")
            continue
        # 加载模型
        model = YOLO("./best.pt")
        # for class_id, class_name in model.names.items():
        #     print(f"类别编号: {class_id}, 类别名称: {class_name}")

        # 进行预测
        results = model.predict(source=image, save=False, conf=0.4, verbose=False)

        # 处理每个检测结果
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                print("    ➖ 没有检测到目标。")
                continue

            print(f"    ✅ 检测到 {len(boxes)} 个目标：")
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"

                print(f"    🔸 目标 {i+1}: 类别={model.names[cls]}, 置信度={conf:.2f}, 坐标=({x1},{y1})-({x2},{y2})")

                # 绘制边框和标签
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 255), -1)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 保存处理后的图片
        cv2.imwrite(output_image_path, image)
        print(f"处理完成，图片已保存: {output_image_path}")

print("\n所有图片处理完毕！")
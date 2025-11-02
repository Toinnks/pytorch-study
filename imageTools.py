import os
import random
import shutil
from tqdm import tqdm
from ultralytics.models import YOLO

"""
remove_image_from_include_str(self, key: str):
    若在folder_path文件夹下的图片名中有key字符，则移除该图片
    
copy_folder_to_src(self, src_folder_path: str, need_copy_folder: str)
     # 将need_copy_folder文件夹中的文件复制到self.folder_path文件夹中

clean_imgdir_and_labeldir(self, label_dir: str, img_dir: str,clean_rule:list=[0,0,1]):
    清理图片和标注文件：
    clean_rule清理规则，[0,0,1]，分别对应下面三条，1为选中
    0. 删除空的标注文件 (.txt 大小为 0KB)，并删除对应图片
    1. 删除没有对应标注文件的图片
    2. 删除没有对应图片的标注文件
    
auto_label(self, model_path: str, image_dir: str, output_label_dir: str, conf_thresh: float = 0.5)
    output_label_dir输出txt标签的文件夹路径
    conf_thresh: float, 置信度阈值,默认0.5,（过滤低置信度框）
    
label_change_class(self,label_dir: str, change_dict: dict)
    改变标签的类别，映射关系为change_dict={'0':'2','1':'4'}
    会读取文件夹下的每一个txt文件的每行，对于类别0，会改为2……
"""


class DatasetProcess(object):
    image_end = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, folder_path=None):
        self.folder_path = folder_path

    def remove_image_from_include_str(self, key: str):
        # 若在folder_path文件夹下的图片名中有key字符，则移除该图片
        folder_path = self.folder_path
        for filename in os.listdir(folder_path):
            if key in filename:
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
                print(f"已删除: {filename}")
        print(f"{folder_path}清理完成！")

    def copy_folder_to_src(self, src_folder_path: str, need_copy_folder: str):

        # 将need_copy_folder文件夹中的文件复制到self.folder_path文件夹中

        for item in os.listdir(need_copy_folder):
            src_path = os.path.join(src_folder_path, item)
            dst_path = os.path.join(need_copy_folder, item)

            if os.path.isdir(dst_path):
                # 如果是文件夹，递归复制
                shutil.copytree(dst_path, src_path, dirs_exist_ok=True)
            else:
                # 如果是文件，直接复制
                shutil.copy2(dst_path, src_path)

    def clean_imgdir_and_labeldir(self, label_dir: str, img_dir: str, clean_rule: list = [0, 0, 1]):
        """
           清理图片和标注文件的无效对应关系：
           0. 删除空的标注文件 (.txt 大小为 0KB)，并删除对应图片
           1. 删除没有对应标注文件的图片
           2. 删除没有对应图片的标注文件
       """
        img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in self.image_end]
        label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

        img_basenames = {os.path.splitext(f)[0] for f in img_files}
        label_basenames = {os.path.splitext(f)[0] for f in label_files}
        if clean_rule[0]:
            # --- Step 1: 删除空标注文件及其对应图片 ---
            for label_file in label_files:
                label_path = os.path.join(label_dir, label_file)
                base_name = os.path.splitext(label_file)[0]

                if os.path.getsize(label_path) == 0:
                    print(f" 删除空标注文件: {label_path}")
                    os.remove(label_path)

                    # 删除对应图片（匹配任意后缀）
                    for ext in self.image_end:
                        img_path = os.path.join(img_dir, base_name + ext)
                        if os.path.exists(img_path):
                            print(f"️ 删除对应图片: {img_path}")
                            os.remove(img_path)
                            break

        # 重新获取有效文件列表（避免上一步删除后仍处理）
        img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in self.image_end]
        label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        img_basenames = {os.path.splitext(f)[0] for f in img_files}
        label_basenames = {os.path.splitext(f)[0] for f in label_files}
        if clean_rule[1]:
            # --- Step 2: 删除没有标注文件的图片 ---
            for base_name in img_basenames - label_basenames:
                for ext in self.image_end:
                    img_path = os.path.join(img_dir, base_name + ext)
                    if os.path.exists(img_path):
                        print(f" 删除无标注图片: {img_path}")
                        os.remove(img_path)
                        break
        if clean_rule[2]:
            # --- Step 3: 删除没有图片的标注文件 ---
            for base_name in label_basenames - img_basenames:
                label_path = os.path.join(label_dir, base_name + ".txt")
                if os.path.exists(label_path):
                    print(f"️ 删除无图片标注文件: {label_path}")
                    os.remove(label_path)

        print(" 清理完成。")

    def split_to_train_valid_test(self, img_dir: str, label_dir: str, output_dir: str,
                                  split_list: list[float] = None, seed: int = 42
                                  ):
        """
        将图片与标签分割为 train/val/test 三个集合。
        自动保持图片和标签一一对应。

        参数：
            img_dir: 图片文件夹路径
            label_dir: 标签文件夹路径
            output_dir: 输出数据集根路径
            split_list: 划分比例 [train, val, test]
            seed: 随机种子，保证可复现
        """

        if split_list is None:
            split_list = [0.7, 0.2, 0.1]
        assert abs(sum(split_list) - 1.0) < 1e-6, "split_list之和必须为1"

        # 检查输入路径
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"图片目录不存在: {img_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"标签目录不存在: {label_dir}")

        # 创建输出目录结构
        subdirs = ["train/images", "train/labels",
                   "val/images", "val/labels",
                   "test/images", "test/labels"]
        for sub in subdirs:
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

        dirs = {
            'train': (os.path.join(output_dir, "train/images"),
                      os.path.join(output_dir, "train/labels")),
            'val': (os.path.join(output_dir, "val/images"),
                    os.path.join(output_dir, "val/labels")),
            'test': (os.path.join(output_dir, "test/images"),
                     os.path.join(output_dir, "test/labels")),
        }

        # 收集所有有效图片（必须有对应标签）
        all_images = []
        for f in os.listdir(img_dir):
            if f.lower().endswith(tuple(self.image_end)):
                name = os.path.splitext(f)[0]
                label_path = os.path.join(label_dir, name + ".txt")
                if os.path.exists(label_path):
                    all_images.append(os.path.join(img_dir, f))

        if not all_images:
            raise RuntimeError("未找到任何有对应标签的图片")

        # 随机打乱
        random.seed(seed)
        random.shuffle(all_images)

        n_total = len(all_images)
        n_train = int(split_list[0] * n_total)
        n_val = int(split_list[1] * n_total)
        splits = {
            "train": all_images[:n_train],
            "val": all_images[n_train:n_train + n_val],
            "test": all_images[n_train + n_val:]
        }

        # 定义复制函数
        def copy_files(img_list: list[str], img_dest: str, lbl_dest: str, phase: str):
            img_copied = 0
            lbl_copied = 0
            lbl_missing = 0

            for img_path in tqdm(img_list, desc=f"{phase:5s}", ncols=80):
                try:
                    shutil.copy2(img_path, img_dest)
                    img_copied += 1
                except Exception as e:
                    print(f"[{phase}] 复制图片失败: {img_path} -> {e}")
                    continue

                # 标签
                name = os.path.splitext(os.path.basename(img_path))[0]
                lbl_path = os.path.join(label_dir, name + ".txt")
                if os.path.exists(lbl_path):
                    try:
                        shutil.copy2(lbl_path, lbl_dest)
                        lbl_copied += 1
                    except Exception as e:
                        print(f"[{phase}] 复制标签失败: {lbl_path} -> {e}")
                else:
                    lbl_missing += 1
                    print(f"[{phase}] 警告：缺少标签文件 -> {lbl_path}")

            return img_copied, lbl_copied, lbl_missing

        # 执行分割
        stats = {}
        for phase, img_list in splits.items():
            img_dest, lbl_dest = dirs[phase]
            stats[phase] = copy_files(img_list, img_dest, lbl_dest, phase)

        print("\n=== 数据集分割完成 ===")
        for phase, (n_img, n_lbl, n_miss) in stats.items():
            print(f"{phase:5s}: 图片 {n_img}, 标签 {n_lbl}, 缺失标签 {n_miss}")

        print(f"输出路径: {os.path.abspath(output_dir)}\n")

    def auto_label(self, model_path: str, image_dir: str, output_label_dir: str, conf_thresh: float = 0.5):
        os.makedirs(output_label_dir, exist_ok=True)
        model = YOLO(model_path)

        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self.image_end)
        ]

        print(f" 共检测到 {len(image_files)} 张图片，开始自动标注...")
        first = True
        for img_name in tqdm(image_files, desc="自动标注中"):
            img_path = os.path.join(image_dir, img_name)
            results = model.predict(img_path, conf=conf_thresh, verbose=False)
            result = results[0]
            boxes = result.boxes.xywhn
            classes = result.boxes.cls
            names = result.names

            if first:
                with open(os.path.join(output_label_dir, "classes.txt"), "w") as f:
                    for key, value in names.items():
                        f.write(f"{value}\n")
                first = False

            label_path = os.path.join(
                output_label_dir,
                os.path.splitext(img_name)[0] + ".txt"
            )

            with open(label_path, "w") as f:
                for box, cls in zip(boxes, classes):
                    f.write(f"{int(cls)} {box[0]:.6f} {box[1]:.6f} "
                            f"{box[2]:.6f} {box[3]:.6f}\n")

        print(" 自动标注完成！")
        print(f" 生成标签目录: {output_label_dir}")

    def label_change_class(self, label_dir: str, change_dict: dict):
        label_txt_list = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        for i, filename in enumerate(label_txt_list):
            file_path = os.path.join(label_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"读取文件 {file_path} 失败，错误：{e}")
                continue

            new_lines = []
            changed = False

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                ori_class_name = parts[0]
                rest_of_line = parts[1] if len(parts) > 1 else ""

                new_class_name = change_dict.get(ori_class_name)
                if new_class_name is None:
                    print(f"[警告] 类映射错误：'{ori_class_name}' 未在 change_dict 中找到（文件: {filename}）")
                    new_lines.append(line)
                    continue

                if new_class_name != ori_class_name:
                    changed = True

                new_lines.append(f"{new_class_name} {rest_of_line}")

            if changed:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(new_lines) + "\n")
                    print(f"已更新: {filename}")
                except Exception as e:
                    print(f"写入文件失败: {file_path}, 错误: {e}")

    def rotate_yolo_data(self,images_folder, labels_folder, augment_set, gray=False):
        def rotate_image(image, angle):
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
                new_width = ((x_max - x_min) / new_w) * 0.92
                new_height = ((y_max - y_min) / new_h) * 0.92

                # 过滤过小框
                if new_width > 0.01 and new_height > 0.01:
                    new_boxes.append([class_id, new_x_center, new_y_center, new_width, new_height])

            return new_boxes

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

s1 = DatasetProcess()
# s1.label_change_class(label_dir=r"D:\dataset\smoke\labels", change_dict={"3":"2"})
# s1.auto_label(model_path=r"C:\Users\26601\Desktop\best.pt",image_dir=r"D:\dataset\train\0_phone",output_label_dir=r"D:\dataset\train\labels")
# s1.clean_imgdir_and_labeldir(img_dir=r"D:\dataset\phone-545\images", label_dir=r"D:\dataset\phone-545\labels",clean_rule=[1,1,1])
s1.split_to_train_valid_test(img_dir=r"D:\dataset\phone-3135\images", label_dir=r"D:\dataset\phone-3135\labels",
                             output_dir=r"D:\dataset\phone-dataset-v7-1030", split_list=[0.7, 0.2, 0.1])

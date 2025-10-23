import os
import random
import shutil
from tqdm import tqdm

"""train_folder_path = None, test_folder_path = None, valid_folder_path = None,
train_images_path = None, test_images_path = None, valid_images_path = None, train_labels_path = None,
test_labels_path = None, valid_labels_path = None"""


class TrainDatasetProcess(object):
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

    def clean_imgdir_and_labeldir(self, label_dir: str, img_dir: str):
        """
           清理图片和标注文件的无效对应关系：
           1. 删除空的标注文件 (.txt 大小为 0KB)，并删除对应图片
           2. 删除没有对应标注文件的图片
           3. 删除没有对应图片的标注文件
       """
        # 获取所有图片和标注文件名（去除扩展名）
        img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in self.image_end]
        label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

        img_basenames = {os.path.splitext(f)[0] for f in img_files}
        label_basenames = {os.path.splitext(f)[0] for f in label_files}

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

        # --- Step 2: 删除没有标注文件的图片 ---
        for base_name in img_basenames - label_basenames:
            for ext in self.image_end:
                img_path = os.path.join(img_dir, base_name + ext)
                if os.path.exists(img_path):
                    print(f" 删除无标注图片: {img_path}")
                    os.remove(img_path)
                    break

        # --- Step 3: 删除没有图片的标注文件 ---
        for base_name in label_basenames - img_basenames:
            label_path = os.path.join(label_dir, base_name + ".txt")
            if os.path.exists(label_path):
                print(f"️ 删除无图片标注文件: {label_path}")
                os.remove(label_path)

        print(" 清理完成。")

    def split_to_train_valid_test(self, img_dir: str, label_dir: str, output_dir: str,
                                  split_list: list[float] = None, seed: int = 42
                                  ) -> dict[str, tuple[int, int, int]]:
        """
        将图片与标签分割为 train/val/test 三个集合。
        自动保持图片和标签一一对应。

        参数：
            img_dir: 图片文件夹路径
            label_dir: 标签文件夹路径
            output_dir: 输出数据集根路径
            split_list: 划分比例 [train, val, test]
            seed: 随机种子，保证可复现

        返回：
            各子集的统计信息 {phase: (图片数, 标签数, 缺失标签数)}
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
        return stats

    def auto_label(self, model_path: str, image_dir: str, output_label_dir: str, conf_thresh: float = 0.5):
        """
        使用YOLOv8模型为未标注图片自动生成伪标签(txt格式)。
        参数:
            model_path: str
                模型路径 (例如 'best.pt')
            image_dir: str
                未标注图片所在文件夹路径
            output_label_dir: str
                输出txt标签的文件夹路径
            conf_thresh: float, 默认0.5
                置信度阈值（过滤低置信度框）
        """
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


s1 = TrainDatasetProcess()
s1.clean_imgdir_and_labeldir(img_dir=r"D:\dataset\train-label\images", label_dir=r"D:\dataset\train-label\labels")
# s1.split_to_train_valid_test(img_dir=r"D:\dataset\phone-data-all\images",
#                              label_dir=r"D:\dataset\phone-data-all\labels",
#                              output_dir=r"D:\dataset\phone-v3-1022",split_list=[0.7, 0.2, 0.1])

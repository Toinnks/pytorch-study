import os
import random
import shutil

from torch.utils.data import Dataset


class TrainDatasetProcess(object):
    image_end = ['png', 'jpg', 'jpeg']
    """train_folder_path = None, test_folder_path = None, valid_folder_path = None,
    train_images_path = None, test_images_path = None, valid_images_path = None, train_labels_path = None,
    test_labels_path = None, valid_labels_path = None"""

    def __init__(self, src_folder_path):
        self.src_folder_path = src_folder_path

    def remove_image_from_include_str(self, key: str):
        # 若在父文件夹下的图片名中有key字符，则移除该图片
        folder_path = self.src_folder_path
        for filename in os.listdir(folder_path):
            if key in filename:
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
                print(f"已删除: {filename}")
        print(f"{folder_path}清理完成！")

    def judge_folder(self, is_train: bool, is_test: bool, is_valid: bool):
        train_child_folder = None
        test_child_folder = None
        valid_child_folder = None
        child_folder = os.listdir(self.src_folder_path)
        for i in child_folder:
            if 'train' in i:
                train_child_folder = os.path.join(self.src_folder_path, i)
            if 'test' in i:
                test_child_folder = os.path.join(self.src_folder_path, i)
            if 'valid' in i:
                valid_child_folder = os.path.join(self.src_folder_path, i)
        if is_test and test_child_folder is None:
            print("test文件夹不存在")

        if is_train and train_child_folder is None:
            print("train文件夹不存在")

        if is_valid and valid_child_folder is None:
            print("valid文件夹不存在")

    def copy_folder_to_src(self, need_copy_folder: str):

        # 将need_copy_folder文件夹中的文件复制到self.src_folder_path文件夹中

        src = self.src_folder_path
        for item in os.listdir(need_copy_folder):
            src_path = os.path.join(src, item)
            dst_path = os.path.join(need_copy_folder, item)

            if os.path.isdir(dst_path):
                # 如果是文件夹，递归复制
                shutil.copytree(dst_path, src_path, dirs_exist_ok=True)
            else:
                # 如果是文件，直接复制
                shutil.copy2(dst_path, src_path)

    def clean_imgdir_by_labeldir(self, label_dir: str):
        """
        通过label_dir文件夹中的标注文件txt，来删除img_dir = self.src_folder_path中的无效图片
        判断条件：
        1、txt文件大小为0kb为无效标注文件txt，会删除txt文件的同时删除对应图片
        2、图片没有对应txt标注文件的话会删除
        """
        img_dir = self.src_folder_path
        # 1. 遍历标注文件夹
        for label_file in os.listdir(label_dir):
            if not label_file.endswith(".txt"):
                continue
            label_path = os.path.join(label_dir, label_file)
            base_name = os.path.splitext(label_file)[0]
            # 1.1 如果标注文件大小为0 → 删除标注和对应图片
            if os.path.getsize(label_path) == 0:
                print(f"删除无效标注文件: {label_path}")
                os.remove(label_path)

                for ext in self.image_end:
                    img_path = os.path.join(img_dir, base_name + ext)
                    if os.path.exists(img_path):
                        print(f"删除对应无效图片: {img_path}")
                        os.remove(img_path)
                        break
        # 2. 遍历图片文件夹，检查是否有对应标注
        for img_file in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_file)
            base_name, ext = os.path.splitext(img_file)
            if ext.lower() not in self.image_end:
                continue
            label_path = os.path.join(label_dir, base_name + ".txt")
            # 2.1 如果没有对应标注文件 → 删除图片
            if not os.path.exists(label_path):
                print(f"删除无标注的图片: {img_path}")
                os.remove(img_path)
            # 2.2 如果标注文件存在但是已经被删除(0B情况处理过) → 图片也删掉
            elif not os.path.exists(img_path):
                print(f"图片已被删除，无需处理: {img_file}")
                continue

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
            if f.lower().endswith(self.image_end):
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
        def copy_files(img_list: List[str], img_dest: str, lbl_dest: str, phase: str):
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

        print("\n=== ✅ 数据集分割完成 ===")
        for phase, (n_img, n_lbl, n_miss) in stats.items():
            print(f"{phase:5s}: 图片 {n_img}, 标签 {n_lbl}, 缺失标签 {n_miss}")

        print(f"输出路径: {os.path.abspath(output_dir)}\n")
        return stats

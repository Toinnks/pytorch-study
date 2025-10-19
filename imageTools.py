import os
import shutil

from torch.utils.data import Dataset


class TrainDatasetProcess(object):
    image_end = ['png', 'jpg', 'jpeg']
    """train_folder_path = None, test_folder_path = None, valid_folder_path = None,
    train_images_path = None, test_images_path = None, valid_images_path = None, train_labels_path = None,
    test_labels_path = None, valid_labels_path = None"""

    def __init__(self, src_folder_path):
        self.src_folder_path = src_folder_path

    def get_child_from_father(self):
        pass

    def remove_image_from_include_str(self, key: str):
        # 在父文件夹下的图片名中有key字符，则移除该图片
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


s1=TrainDatasetProcess(src_folder_path=r"D:\edgeDownloads\Mobile detection.v1i.yolov8\valid\labels")
s1.copy_folder_to_src(need_copy_folder=r"D:\edgeDownloads\phone.v1i.yolov8\valid\labels")

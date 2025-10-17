import os

from torch.utils.data import Dataset


class TrainDatasetProcess(object):
    image_end = ['png','jpg','jpeg']
    def __init__(self, father_folder_path=None, train_folder_path=None, test_folder_path=None, valid_folder_path=None,
                 train_images_path=None, test_images_path=None, valid_images_path=None, train_labels_path=None,
                 test_labels_path=None, valid_labels_path=None):
        self.father_folder_path = father_folder_path
        self.train_folder_path = train_folder_path
        self.test_folder_path = test_folder_path
        self.valid_folder_path = valid_folder_path
        self.train_images_path = train_images_path
        self.test_images_path = test_images_path
        self.valid_images_path = valid_images_path
        self.train_labels_path = train_labels_path
        self.test_labels_path = test_labels_path
        self.valid_labels_path = valid_labels_path

    def get_child_from_father(self):
        pass

    def remove_image_from_include_str(self, key: str):
        # 在父文件夹下的图片名中有key字符，则移除该图片
        folder_path = self.father_folder_path
        for filename in os.listdir(folder_path):
            if  key in filename:
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
                print(f"已删除: {filename}")
        print(f"{folder_path}清理完成！")

print()
s1 = TrainDatasetProcess()
s1.father_folder_path = r"D:\else\image"
s1.remove_image_from_include_str("33")
import os

from torch.utils.data import Dataset

folder_path = r"D:\dingdingDownloads\person_detection"  # 当前目录，可以修改为您的文件夹路径

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") and ".ori." not in filename:
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
        print(f"已删除: {filename}")

print("清理完成！")


class TrainDatasetProcess(object):
    def __init__(self, father_image_path=None, train_image_path=None, test_image_path=None, valid_image_path=None,
                 train_label_path=None, test_label_path=None, valid_label_path=None):
        self.father_image_path = father_image_path
        self.train_image_path = train_image_path
        self.test_image_path = test_image_path
        self.valid_image_path = valid_image_path
        self.train_label_path = train_label_path
        self.test_label_path = test_label_path
        self.valid_label_path = valid_label_path

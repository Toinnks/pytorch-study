import os
import shutil

def separate_files(src_folder, img_folder, json_folder):
    # 创建目标文件夹（如果不存在）
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_folder):
        filepath = os.path.join(src_folder, filename)

        if os.path.isfile(filepath):
            # 判断文件类型并移动
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')):
                shutil.copy(filepath, os.path.join(img_folder, filename))
                print(f"已复制图片: {filename}")
            elif filename.lower().endswith('.json'):
                shutil.copy(filepath, os.path.join(json_folder, filename))
                print(f"已复制JSON: {filename}")

if __name__ == "__main__":
    # 设置源文件夹和目标文件夹路径
    source_folder = r"./newdata"       # 原始文件夹
    image_folder = r"./newdata1/images" # 图片保存路径
    json_folder = r"./newdata1/jsons"   # JSON 保存路径

    separate_files(source_folder, image_folder, json_folder)
    print("文件分类完成！")

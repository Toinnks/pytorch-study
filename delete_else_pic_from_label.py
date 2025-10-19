import os

def clean_dataset(img_dir, label_dir):
    """
    清理无效标注文件和无效图片
    :param img_dir: 图片所在文件夹（文件夹1）
    :param label_dir: 标注文件所在文件夹（文件夹2）
    """
    # 支持的图片格式
    img_exts = [".jpg", ".jpeg", ".png", ".bmp"]

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

            for ext in img_exts:
                img_path = os.path.join(img_dir, base_name + ext)
                if os.path.exists(img_path):
                    print(f"删除对应无效图片: {img_path}")
                    os.remove(img_path)
                    break

    # 2. 遍历图片文件夹，检查是否有对应标注
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        base_name, ext = os.path.splitext(img_file)

        if ext.lower() not in img_exts:
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

if __name__ == "__main__":
    folder1 = r"C:\Users\26601\Desktop\alarmpic"  # 文件夹1
    folder2 = r"D:\else\target"  # 文件夹2

    clean_dataset(folder1, folder2)

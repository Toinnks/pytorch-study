import os

def modify_image_name_if_include_str(folder_path: str, key: str):
    renamed_count = 0  # 记录重命名的文件数量

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if key in filename:
            old_file_path = os.path.join(folder_path, filename)
            # 将文件名中的 key 替换为空字符串，即删除 key
            new_filename = filename.replace(key, "")
            new_file_path = os.path.join(folder_path, new_filename)

            # 检查新文件名是否已存在，避免覆盖
            if os.path.exists(new_file_path):
                print(f"跳过重命名: {filename} -> {new_filename} (目标文件已存在)")
                continue

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"已重命名: {filename} -> {new_filename}")
            renamed_count += 1

    print(f"{folder_path} 重命名完成！共处理 {renamed_count} 个文件。")
if __name__ == '__main__':
    modify_image_name_if_include_str(r'D:\dataset\phone-dataset-v8-1104\val\images', '浙')
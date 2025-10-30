import os
from PIL import Image, ImageEnhance
import glob

def process_images_to_nightvision(images_folder, output_folder, num=0):
    """
    将文件夹中的图片转换为灰度夜视效果（增强对比度）
    参数:
        images_folder: 源图片文件夹路径
        output_folder: 输出文件夹路径
        num: 处理的数量，0表示全部，其他数字表示处理的图片数量
    """
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']

    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, ext)))
        image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))

    image_files = sorted(list(set(image_files)))
    if not image_files:
        print(f"错误: 在 '{images_folder}' 文件夹中没有找到图片文件")
        return

    total_images = len(image_files)
    print(f"找到 {total_images} 张图片")

    # 处理数量
    process_count = total_images if num == 0 or num >= total_images else num
    print(f"将处理 {process_count} 张图片")

    os.makedirs(output_folder, exist_ok=True)
    print(f"输出文件夹: {output_folder}")

    # 开始处理
    success_count = 0
    for i, image_path in enumerate(image_files[:process_count]):
        try:
            # 打开图片
            img = Image.open(image_path)

            # 转换为灰度
            gray_img = img.convert('L')

            # 提升对比度，让夜视灰度更清晰
            enhancer = ImageEnhance.Contrast(gray_img)
            nightvision_img = enhancer.enhance(1.5)  # 1.5为增强系数，可调节

            # 保存为灰度图像
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_folder, filename)
            nightvision_img.save(output_path)

            success_count += 1
            print(f"[{i + 1}/{process_count}] 已处理: {filename}")

        except Exception as e:
            print(f"[{i + 1}/{process_count}] 处理失败 {os.path.basename(image_path)}: {e}")

    print(f"\n处理完成！成功处理 {success_count}/{process_count} 张图片")
    print(f"输出位置: {output_folder}")


# 示例运行
if __name__ == "__main__":
    process_images_to_nightvision(
        r"D:\edgeDownloads\pp_smoke\images",
        r"D:\dataset\output_images_gray",
        100
    )

"""
    数据集分割 ———— train / val / test  (7:2:1)
    YOLO 标签（.txt）
"""

import os
import shutil
import random
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")

def ensure_dirs(base):
    subdirs = [
        "train/images", "train/labels",
        "val/images",   "val/labels",
        "test/images",  "test/labels",
    ]
    for sd in subdirs:
        os.makedirs(os.path.join(base, sd), exist_ok=True)

def to_label_path(img_path, label_root):
    # 14.png -> 14.txt ；14.JPG -> 14.txt
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(label_root, stem + ".txt")

def _copy(from_path, to_dir):
    os.makedirs(to_dir, exist_ok=True)
    to_path = os.path.join(to_dir, os.path.basename(from_path))
    shutil.copy2(from_path, to_path)

def split_img(img_path, label_path, split_list):
    out_root = "./traindatas"
    ensure_dirs(out_root)

    train_img_dir = os.path.join(out_root, "train/images")
    val_img_dir   = os.path.join(out_root, "val/images")
    test_img_dir  = os.path.join(out_root, "test/images")

    train_label_dir = os.path.join(out_root, "train/labels")
    val_label_dir   = os.path.join(out_root, "val/labels")
    test_label_dir  = os.path.join(out_root, "test/labels")

    train, val, test = split_list

    # 只收集图片文件
    all_imgs = [
        os.path.join(img_path, f)
        for f in os.listdir(img_path)
        if os.path.isfile(os.path.join(img_path, f)) and f.lower().endswith(IMAGE_EXTS)
    ]

    if not all_imgs:
        print("未在图片目录中找到图片文件。")
        return

    random.shuffle(all_imgs)  # 打乱后再按比例切分
    n = len(all_imgs)
    n_train = int(train * n)
    n_val   = int(val * n)
    # 剩余给 test
    n_test  = n - n_train - n_val

    train_imgs = all_imgs[:n_train]
    val_imgs   = all_imgs[n_train:n_train + n_val]
    test_imgs  = all_imgs[n_train + n_val:]

    def copy_pair(img_list, img_dir, lbl_dir, phase):
        img_ok = 0
        lbl_ok = 0
        missing_lbl = 0
        for img in tqdm(img_list, desc=f"{phase:5s}", ncols=80, unit="img"):
            try:
                _copy(img, img_dir)
                img_ok += 1
            except Exception as e:
                print(f"[{phase}] 复制图片失败: {img} -> {e}")

            lbl = to_label_path(img, label_path)
            if os.path.isfile(lbl):
                try:
                    _copy(lbl, lbl_dir)
                    lbl_ok += 1
                except Exception as e:
                    print(f"[{phase}] 复制标签失败: {lbl} -> {e}")
            else:
                missing_lbl += 1
                # 这里仅告警，不中断流程
                print(f"[{phase}] 警告：缺少标签文件 -> {lbl}")

        return img_ok, lbl_ok, missing_lbl

    ti, tl, tm = copy_pair(train_imgs, train_img_dir, train_label_dir, "train")
    vi, vl, vm = copy_pair(val_imgs,   val_img_dir,   val_label_dir,   "val")
    si, sl, sm = copy_pair(test_imgs,  test_img_dir,  test_label_dir,  "test")

    print("\n=== 拷贝统计 ===")
    print(f"train: images {ti}/{len(train_imgs)}, labels {tl}/{len(train_imgs)}, 缺少标签 {tm}")
    print(f"val  : images {vi}/{len(val_imgs)}, labels {vl}/{len(val_imgs)}, 缺少标签 {vm}")
    print(f"test : images {si}/{len(test_imgs)}, labels {sl}/{len(test_imgs)}, 缺少标签 {sm}")
    print("完成。输出目录：", os.path.abspath(out_root))

if __name__ == '__main__':
    img_path   = r"C:\Users\26601\Desktop\alarmpic"   # 图片目录
    label_path = r"D:\else\target"   # 标签目录（.txt）
    split_list = [0.7, 0.2, 0.1]  # [train, val, test]
    split_img(img_path, label_path, split_list)

import os
import random
import shutil
# from sklearn.model_selection import train_test_split
import streamlit as st  

def split_file(paired_files, test_size):
    split_index=int(len(paired_files)*test_size)
    train_pairs=paired_files[:split_index]
    val_pairs=paired_files[split_index:]
    return train_pairs, val_pairs


def random_shuffle(train_dir_path, val_dir_path, upload_dir, train_ratio):
    # 画像ファイルと対応するラベルファイルのリストを取得
    image_files = [f for f in os.listdir(upload_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    label_files = [f for f in os.listdir(upload_dir) if f.endswith('.txt')]

    # 画像とラベルのファイル名が一致することを確認
    paired_files = [(img, img.replace(os.path.splitext(img)[1], ".txt")) for img in image_files if img.replace(os.path.splitext(img)[1], ".txt") in label_files]

    if len(paired_files) == 0:
        st.error("画像とラベルのファイル名が一致しません。")
    else:
        # データセットをランダムにシャッフル
        random.shuffle(paired_files)
        
        # データセットを７対３で分割
        # train_files, val_files = train_test_split(paired_files, test_size=0.7)
        train_files, val_files = split_file(paired_files, test_size=train_ratio)

        # データセットをトレーニングセットと検証セットに分割
        os.makedirs(os.path.join(train_dir_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(train_dir_path, "labels"), exist_ok=True)
        os.makedirs(os.path.join(val_dir_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(val_dir_path, "labels"), exist_ok=True)

        # ファイルを各ディレクトリに移動
        for img, label in train_files:
            shutil.move(os.path.join(upload_dir, img), os.path.join(train_dir_path, "images", img))
            shutil.move(os.path.join(upload_dir, label), os.path.join(train_dir_path, "labels", label))

        for img, label in val_files:
            shutil.move(os.path.join(upload_dir, img), os.path.join(val_dir_path, "images", img))
            shutil.move(os.path.join(upload_dir, label), os.path.join(val_dir_path, "labels", label))
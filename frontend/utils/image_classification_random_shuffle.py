import os
import random
import shutil
import pandas
# from sklearn.model_selection import train_test_split
import streamlit as st  

def extract_image_name(full_path):
    parts = full_path.split('-')
    return '-'.join(parts[1:])

def split_file(paired_files, test_size):
    split_index=int(len(paired_files)*test_size)
    train_pairs=paired_files[:split_index]
    # st.write(f"train_pairs=paired_files[:split_index]:{train_pairs}")
    val_pairs=paired_files[split_index:]
    # st.write(f"val_pairs=paired_files[split_index:]:{val_pairs}")
    
    return train_pairs, val_pairs

def image_classification_random_shuffle(train_dir_path, val_dir_path, csv_path, upload_dir, train_ratio):
    # 画像ファイルと対応するラベルファイルのリストを取得
    df = pandas.read_csv(csv_path)
    csv_data = df["image"].to_list()
    csv_files = [os.path.basename(path) for path in csv_data]
    # st.write(csv_files)
    images_names = []
    for path in csv_files:
        # images_names.append(extract_image_name(path))
        images_names.append(path)
    csv_label = df["choice"].unique().tolist()
    tuple_list = list(zip(df["choice"], images_names))
    image_files = [f for f in os.listdir(upload_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 画像とラベルのファイル名が一致することを確認
    paired_files = [img for img in image_files if any(img in path for path in images_names)]

    if len(paired_files) == 0:
        st.error("画像とクラスの情報がありません。")
        st.error("csvファイルを入れ直してください")
    else:
        # データセットをランダムにシャッフル
        random.shuffle(tuple_list)
        
        # データセットを７対３で分割
        train_files, val_files = split_file(tuple_list, test_size=train_ratio)



        # データセットをトレーニングセットと検証セットに分割
        for choice in train_files:
            os.makedirs(os.path.join(train_dir_path, choice[0]), exist_ok=True)
        for val in val_files:
            os.makedirs(os.path.join(val_dir_path, val[0]), exist_ok=True)

        # ファイルを各ディレクトリに移動
        for img in train_files:
            shutil.move(os.path.join(upload_dir, img[1]), os.path.join(train_dir_path, img[0], img[1]))

        for val_img in val_files:
            shutil.move(os.path.join(upload_dir, val_img[1]), os.path.join(val_dir_path, val_img[0], val_img[1]))

    return csv_label
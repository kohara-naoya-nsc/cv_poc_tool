import streamlit as st  
# import cv2
import os
import io
import zipfile
import shutil
import time
import threading
import requests
import re
import yaml
import numpy as np
import pandas 
from pathlib import Path
# from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw
import numpy as pd
import cv2
import os
from datetime import datetime
from tempfile import NamedTemporaryFile
from tempfile import TemporaryDirectory
# from ultralytics import YOLO
from utils.dtc_create_yaml import dtc_create_yaml
from utils.random_shuffle import random_shuffle

# 学習コールバック関数
# def on_train_epoch_end(trainer): paired_files = [(img, img.replace(os.path.splitext(img)[1], ".txt")) for img in image_files if img.replace(os.path.splitext(img)[1], ".txt") in label_files]
#     epoch = trainer.epoch + 1
#     total_epochs = trainer.epochs
#     progress = (epoch / total_epochs) * 100
#     log_line = f"Epoch {epoch}/{total_epochs} - Progress: {progress:.2f}%"

#     st.session_state.logs.append(log_line)
#     st.session_state.progress = progress

def on_download_click(parent_directory,run_id):
    delete_path=os.path.join(parent_directory,run_id)
    shutil.rmtree(delete_path)
    st.session_state.view=False

def para_change():
    # print("初期化開始")
    st.session_state.clear()
    st.session_state['lang'] = "en"
    st.session_state['use_angle_cls'] = False

    # print("初期化終了")

# def train_model(model,parent_directory,current_datetime,directory_name_format,epochs,batch,save,cache, device, workers, hsv_h, hsv_s, hsv_v, degrees, flipud, fliplr, use_model):
#     model.train(data=f"{parent_directory}/yaml_{current_datetime.strftime(directory_name_format)}/data.yaml", epochs=epochs, batch=batch, save=save, cache=cache, device=device, workers=workers, hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v, degrees=degrees, flipud=flipud, fliplr=fliplr, name=f"{current_datetime.strftime(directory_name_format)}_{os.path.basename(use_model)}_学習結果")

def paddleocr_pre():

    st.title("【推論】文字認識")
    if'zip_data' not in st.session_state:
        st.session_state.zip_data=None
    if'zip_filename' not in st.session_state:
        st.session_state.zip_filename=None
    if'view' not in st.session_state:
        st.session_state.view=True
    if'run_id' not in st.session_state:
        st.session_state.run_id=None

    # # Streamlitセッション状態の初期化
    if 'degrees' not in st.session_state:
        st.session_state.degrees = 0.0

    # 任意の場所(親ディレクトリ)を指定
    #parent_directory = os.path.abspath(__file__)
    #print(parent_directory)
    parent_directory = "shared/predict/"
    # model_dir_path = "/home/itp0139/work/streamlit-app/app/model"
    # 現在の日付と時刻を取得
    # current_datetime = datetime.now()
    # # ディレクトリ名に使用するフォーマット(yyyyMMddhhmmss)
    # directory_name_format = "%Y%m%d_%H%M%S"
    # run_id=f"{current_datetime.strftime(directory_name_format)}"
    # # 作成したいディレクトリのパスを作成
    # train_directory_name = f"{current_datetime.strftime(directory_name_format)}/pre_data"
    # # 完全なディレクトリパスを作成
    # train_dir_path = os.path.join(parent_directory, train_directory_name)

    

    # 画像をアップロードする
    # 注釈を挿入
    # img_options = ["動画", "画像"]
    # img_toggle = st.pills("推論するものをアップロードしてください。", img_options, selection_mode="single")

    # model_dir_path = "/home/itp0139/work/streamlit-app/app/model"
    # # 画像をアップロードする
    # # 注釈を挿入
    # if img_toggle == "動画":
    #     st.markdown("###### 動画をアップロードしてください")
    #     multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
    #     *対応する形式：.avi, .mp4
    #     '''
    #     st.markdown(multi_img)
    #     file_path = st.file_uploader('import images(動画をアップロードしてください)', type=['avi', 'mp4'], accept_multiple_files=True, label_visibility="hidden")
    #     st.write("\n\n\n")
    # elif img_toggle == "画像":
    st.markdown("###### 画像をアップロードしてください")
    multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
    *対応する形式：.jpg, .png, .jpeg
    '''
    st.markdown(multi_img)
    file_path = st.file_uploader('import images(画像をアップロードしてください)', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, label_visibility="hidden")
    st.write("\n\n\n\n\n")

     # ハイパーパラメータを取得
    # オプション
    with st.expander("オプション"):
        lang_options = ["en", "ja"]
        lang = st.selectbox("言語選択", lang_options, index=lang_options.index("en"), key="lang")
    
    with st.expander("高度なオプション"):
        st.info("必要に応じて変更を加えてください。")
        # st.write("# 特別な指示がない限り変更しないでください")
        use_angle_cls = st.toggle("use_angle_cls:画角自動補正", True, key="use_angle_cls")


    # ２つの列を作成
    col1, col2 = st.columns(2)

    with col2:
        st.button("パラメータ初期化", on_click=para_change)

    with col1:
        start_button = st.button('Start')

    if start_button:
        current_datetime = datetime.now()
        # ディレクトリ名に使用するフォーマット(yyyyMMddhhmmss)
        directory_name_format = "%Y%m%d_%H%M%S"
        run_id=f"{current_datetime.strftime(directory_name_format)}"
        st.session_state.run_id=run_id
        # 作成したいディレクトリのパスを作成
        train_directory_name = f"{current_datetime.strftime(directory_name_format)}/pre_data"
        # 完全なディレクトリパスを作成
        train_dir_path = os.path.join(parent_directory, train_directory_name)
            # 画像をtrainとvalに分割する
        # train_ratio = div_ratio/100
        os.makedirs(train_dir_path, exist_ok=True)
        os.chmod(f'{parent_directory}/{run_id}', 0o777)

        data_config={
            "lang": lang,
            "use_angle_cls": use_angle_cls,
        }
        data_yaml_path=os.path.join(train_dir_path,"data.yaml")
        with open(data_yaml_path,"w") as f:
            yaml.dump(data_config,f)


        if file_path is not None:
            for file in file_path:
                save_path=os.path.join(train_dir_path,file.name)

                with open(save_path,"wb") as f:
                    f.write(file.getbuffer())
                                            

        result_name = f"{current_datetime.strftime(directory_name_format)}_推論結果"
        with st.spinner("文字列を抽出しています..."):
            data = { "run_id": f"{run_id}"}
            response=requests.post("http://paddleocr:83/predict/",data=data)

        if response.status_code == 200:
            st.write("推論完了")
            #if img_toggle == "画像" :
            folder_path = f"{parent_directory }{run_id}/result_data"
            # st.write(f"folder_path")
            zip_path = f"{current_datetime.strftime(directory_name_format)}_推論結果.zip"
            shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
            # フォルダパスをPathオブジェクトで指定
            image_dir = Path(folder_path)

                # 画像拡張子を含むファイルを取得
            # image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
                # または、正規表現で拡張子を指定（より柔軟）
            csv_file = list(image_dir.glob("*.csv"))
            if len(csv_file) >0:
                st.session_state.csv_file=csv_file
            image_files = [f for f in image_dir.iterdir() if f.is_file() and re.match(r".*\.(jpg|jpeg|png|gif|bmp|tiff)$", f.name, re.IGNORECASE)]

            image_files.sort()

            # 画像リストと初期選択インデックスをセッション状態に保存
            # st.session_state.csv_file = csv_file
            st.session_state.image_files = image_files
            st.session_state.selected_index = 0
            with open(zip_path, 'rb') as f:
                zip_data = f.read()
                st.session_state.zip_data=zip_data
                st.session_state.zip_filename=zip_path
            os.remove(zip_path)
            # for idx,img in enumerate(image_files):
            #     st.image(str(img), caption='文字認識結果')
        elif response.status_code == 503:
            st.warning("""
            メモリ不足のためアプリが実行できません。
            手順書に従って立ち下げ、立ち上げ操作を行ってください。
            """)
        else:
            st.error("予期せぬエラーが発生しました。")
            st.error("表示されているエラーメッセージをコピーのうえ、サポート窓口までご連絡ください。")
        
    # 画像モードかつ画像リストがセッション状態にある場合のみ表示
    #if img_toggle == "画像" and "image_files" in st.session_state:
    if "image_files" in st.session_state and st.session_state.view:
        image_files = st.session_state.image_files
        image_options = [img.name for img in image_files]
        # csv_file = st.session_state.csv_file
        # df=pandas.read_csv(csv_file[0])

        selected_index = st.selectbox(
            "表示する画像を選択してください",
            options=range(len(image_options)),
            format_func=lambda x: image_options[x],
            index=st.session_state.selected_index
        )

        # 選択変更時にセッション状態を更新
        st.session_state.selected_index = selected_index

        # 選択された画像を表示
        selected_image = image_files[selected_index]
        # st.write(f"selected_image:{selected_image}")
        st.image(str(selected_image), caption='文字認識結果')
        if "csv_file" in st.session_state:
            csv_file=st.session_state.csv_file
            df=pandas.read_csv(csv_file[0])
            filtered_df=df[df["ImageName"] == os.path.basename(selected_image)][["BoundingBoxNo","Text"]]
            # st.write(f"filtered_df:{filtered_df}")
            # st.table(filtered_df.reset_index(drop=True))
            # if filtered_df is not None:
            st.write(filtered_df)
        # st.table(df)


    #if img_toggle=="画像" and st.session_state.zip_data and st.session_state.zip_filename:
    if st.session_state.zip_data and st.session_state.zip_filename and st.session_state.view and st.session_state.run_id:
        run_id=st.session_state.run_id
        st.download_button(
                        label="文字認識の推論結果をダウンロード",
                        data=st.session_state.zip_data,
                        file_name=st.session_state.zip_filename,
                        mime='application/zip',
                        disabled=False,
                        on_click=lambda: on_download_click(parent_directory,run_id)
        )

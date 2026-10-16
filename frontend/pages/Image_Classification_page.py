import streamlit as st  
# import cv2
import os
import shutil
import time
import io
import zipfile
import requests
import numpy as np
import random
from datetime import datetime
from tempfile import NamedTemporaryFile
from tempfile import TemporaryDirectory
# from ultralytics import YOLO
from utils.create_imgclass_yaml import create_imgclass_yaml
from utils.image_classification_random_shuffle import image_classification_random_shuffle

# 物体検出の時のimagesフォルダ内の画像とcsvファイルに書かれている画像のファイル名を対応させる。
# 画像をアップロードするようにする。
# imagesフォルダ内の画像とcsvファイルに書かれている画像のファイル名が一致することを確認してからランダムシャッフルする

def on_download_click(parent_directory,run_id):
    """学習完了後の後処理"""
    delete_path=os.path.join(parent_directory,run_id)
    shutil.rmtree(delete_path)
    st.session_state.view=False


def split_file(paired_files, test_size):
    split_index=int(len(paired_files)*test_size)
    train_pairs=paired_files[:split_index]
    val_pairs=paired_files[split_index:]
    
    return train_pairs, val_pairs

def random_shuffle(train_dir_path, val_dir_path, upload_dir, train_ratio):
    # 画像ファイルと対応するラベルファイルのリストを取得
    tuple_list=[]
    for subdir in os.listdir(upload_dir):
        subdir_path=os.path.join(upload_dir,subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                file_path=os.path.join(subdir_path,file)
                if os.path.isfile(file_path):
                    tuple_list.append((subdir,file))
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
        shutil.move(os.path.join(upload_dir, img[0], img[1]), os.path.join(train_dir_path, img[0], img[1]))

    for val_img in val_files:
        shutil.move(os.path.join(upload_dir, val_img[0], val_img[1]), os.path.join(val_dir_path, val_img[0], val_img[1]))


def para_change():
    """パラメータ初期化"""
    st.session_state['selection_device'] = "GPU"
    st.session_state['device'] = "0"
    st.session_state['degrees'] = 0.0
    st.session_state['flipud_toggle'] = False
    st.session_state['fliplr_toggle'] = False
    st.session_state['div_ratio'] = 70
    st.session_state['epochs'] = 500
    st.session_state['batch_toggle'] = True
    st.session_state['batch'] = 16
    st.session_state['save'] = True
    st.session_state['cache'] = False
    st.session_state['workers'] = 8
    st.session_state['hsv_h'] = 0.0
    st.session_state['hsv_s'] = 0.0
    st.session_state['hsv_v'] = 0.0
    st.session_state['translate'] = 0.1
    st.session_state['scale'] = 0.5
    st.session_state['shear'] = 0.0
    st.session_state['perspective'] = 0.0
    st.session_state['mosaic'] = 1.0
    st.session_state['mixup'] = 0.0
    st.session_state['copy_paste'] = 0.0


def image_classification():
    """画像分類：学習ページ"""

    # 共有フォルダ
    parent_directory = "shared/train/"
    st.title("【学習】画像分類")

    if'zip_data' not in st.session_state:
        st.session_state.zip_data=None
    if'zip_filename' not in st.session_state:
        st.session_state.zip_filename=None
    if'view' not in st.session_state:
        st.session_state.view=True
    if'run_id' not in st.session_state:
        st.session_state.run_id=None

    ###########################
    # データセットのアップロード
    ###########################
    img_options = ["LabelStadioFormat", "YOLO Format"]
    img_toggle = st.pills("推論するものをアップロードしてください。", img_options, selection_mode="single")
    
    if img_toggle == "LabelStadioFormat":
        st.markdown("###### import zipfile(zipファイルをアップロードしてください)")
        multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
        *対応する形式：.zip
        '''
        st.markdown(multi_img)
        files_path = st.file_uploader('import zipfile(zipファイルをアップロードしてください)', type=['zip'], accept_multiple_files=False, label_visibility="hidden", key="labelstudio")
        st.write("\n\n\n\n\n")

        # アノテーションラベルをアップロードする
        st.markdown("###### import csv(アノテーションラベルをアップロードしてください)")
        multi_csv = '''*注：日本語を含むファイルはアップロードしないでください。
                *対応する形式：.csv
        '''
        st.markdown(multi_csv)
        csv_path = st.file_uploader('import csv(アノテーションラベルをアップロードしてください)', type=['csv'], label_visibility="hidden", key="csv_file")
        st.write("\n\n\n\n\n")
    
    elif img_toggle == "YOLO Format":
        st.markdown("###### 画像をアップロードしてください")
        multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
        *対応する形式：.zip
        '''
        st.markdown(multi_img)
        zip_path = st.file_uploader('import zipfile(zipファイルをアップロードしてください)', type=['zip'], label_visibility="hidden", key="hot_zip")        
        st.write("\n\n\n\n")


    ###########################
    # 学習方法の選択
    ###########################
    model_options = ["再学習", "１から学習"]
    selection_model = st.pills("学習方法選択", model_options, index=0, selection_mode="single")

    # 再学習の場合
    if selection_model == "再学習" :
        resume = True

        st.markdown("###### import model(モデル名.ptをアップロードしてください)")
        multi_mdl = '''*注：日本語を含むファイルはアップロードしないでください。
             *対応する形式：.pt
        '''
        st.markdown(multi_mdl)

        model_path = st.file_uploader(
            'import model(モデル名.ptをアップロードしてください)',
            type=['pt'],
            label_visibility="hidden"
            )
        if model_path :
            use_model = f"{model_path.name}"
    
    # １から学習の場合
    elif selection_model == "１から学習":
        resume = False

        select_model = st.selectbox(
            "デフォルトのモデルを選択してください。",
                (
                    "yolov8n-cls.pt(5.3MB)",
                    "yolov8s-cls.pt(12.3MB)",
                    "yolov8m-cls.pt(32.7MB)",
                    "yolov8l-cls.pt(71.7MB)",
                    "yolov8x-cls.pt(110.0MB)"
                )
            )
        use_model = f"{select_model.rsplit('(',1)[0]}"
        # path=f"models/{use_model}"
        
    st.write("\n\n\n")
    
    
    #############################
    # パラメータ設定
    #############################
    with st.expander("オプション"):
        device_options = ["CPU", "GPU"]
        selection_device = st.selectbox("デバイス選択", device_options, index=device_options.index("GPU"), key="selection_device")
        if selection_device == "CPU" :
            device = selection_device
        elif selection_device == "GPU" :
            gpu_options = {0:"0", 1:"1", 2:"2", 3:"3"}
            device = st.multiselect("使用するGPUの番号を選択してください。", gpu_options, default=0, key="device")
        
        st.info("以下の項目は学習データセット水増しオプション")
        degrees = st.slider("degrees:最大回転範囲", min_value=0.0, max_value=45.0, value=0.0, key="degrees")
        flipud_toggle = st.toggle("flipud:垂直方向の反転", False, key="flipud_toggle")
        if flipud_toggle :
            flipud = 0.5
        else :
            flipud = 0.0
        fliplr_toggle = st.toggle("fliplr:水平方向の反転", False, key="fliplr_toggle")
        if fliplr_toggle :
            fliplr = 0.5
        else :
            fliplr = 0.0
    
    with st.expander("高度なオプション"):
        st.info("必要に応じて変更を加えてください。")
        div_ratio = st.slider("ファイル分割率%", min_value=0, max_value=100, value=70, key="div_ratio")
        epochs = st.slider("epochs", min_value=0, max_value=1000, value=500, key="epochs")
        batch_toggle = st.toggle("batch", value=True, key="batch_toggle")
        if batch_toggle :
            batch = -1
        else :
            batch = st.slider("batch:同時に処理される画像の数", min_value=-1, max_value=1024, value=16)
        save = st.toggle("save:チェックポイントと重みを保存", True, key="save")
        cache = st.toggle("cache:メモリ上に一時保存", False, key="cache")
        workers = st.slider("workers:データロードのためのワーカースレッド数", min_value=0, max_value=100, value=8, key="workers")
        hsv_h = st.slider("hsv_h:HSV空間の色相拡張範囲", min_value=0.0, max_value=1.0, value=0.0, key="hsv_h")
        hsv_s = st.slider("hsv_s:HSV空間の彩度増加範囲", min_value=0.0, max_value=0.9, value=0.0, key="hsv_s")
        hsv_v = st.slider("hsv_v:輝度増加範囲", min_value=0.0, max_value=0.9, value=0.0, key="hsv_v")
        translate = st.slider("translate:画像の移動範囲(画像サイズに対する比率)", min_value=0.0, max_value=1.0, value=0.1, key="translate")
        scale = st.slider("scale:画像のスケーリング範囲(倍率)", min_value=0.0, max_value=10.0, value=0.5, key="scale")
        shear = st.slider("shear:画像のシアー変換範囲", min_value=-90.0, max_value=90.0, value=0.0, key="shear")
        perspective = st.slider("perspective:透視投影変換の強さ", min_value=0.0, max_value=1.0, value=0.0, key="perspective")
        mosaic = st.slider("mosaic:Mosaic augmentationの適応確率", min_value=0.0, max_value=1.0, value=1.0, key="mosaic")
        mixup = st.slider("mixup:Mixup augmentationの適応確率", min_value=0.0, max_value=1.0, value=0.0, key="mixup")
        copy_paste  = st.slider("copy_paste:Copy-paste augmentationの適応確率", min_value=0.0, max_value=1.0, value=0.0, key="copy_paste")
        st.markdown(
            f'<a href="https://docs.ultralytics.com/modes/train/#train-settings" target="_blank" style="color: blue; text-decoration: underline;">Ultralytics公式ドキュメント</a>',
            unsafe_allow_html=True
        )


    ######################
    # オプションボタンの作成
    ######################
    col1, col2 = st.columns(2)

    with col2:
        st.button("パラメータ初期化", on_click=para_change)

    with col1:
        start_button = st.button('Start')

    ######################
    # 学習API の呼び出し
    ######################
    if start_button:
        if img_toggle == "LabelStadioFormat":
            current_datetime = datetime.now()
            
            # ディレクトリ名に使用するフォーマット(yyyyMMddhhmmss)
            directory_name_format = "%Y%m%d_%H%M%S"
            run_id=f"{current_datetime.strftime(directory_name_format)}"
            st.session_state.run_id=run_id
            
            # 画像分類ディレクトリの作成
            class_directory_name = f"{current_datetime.strftime(directory_name_format)}/pre_data"
            class_dir_path=os.path.join(parent_directory, class_directory_name)
            
            # 作成したいディレクトリのパスを作成
            train_directory_name = f"train"
            val_directory_name = f"val"
            yaml_directory_name = f"{current_datetime.strftime(directory_name_format)}/pre_data"
            
            # 完全なディレクトリパスを作成
            train_dir_path = os.path.join(class_dir_path, train_directory_name)
            val_dir_path = os.path.join(class_dir_path, val_directory_name)
            yaml_dir_path = os.path.join(parent_directory, yaml_directory_name)
            os.makedirs(class_dir_path, exist_ok=True)
            os.makedirs(train_dir_path, exist_ok=True)
            os.makedirs(val_dir_path, exist_ok=True)
            os.chmod(f'{parent_directory}/{run_id}', 0o777)
            save_path=os.path.join(class_dir_path,use_model)

            if selection_model == "再学習" :
                with open(save_path,"wb") as f:
                    f.write(model_path.getbuffer())
            # elif selection_model == "１から学習":
            #     shutil.copy(path,class_dir_path)
            

            ###################################################        
            # データセットを学習用（train）と検証用（val）に分割する
            ###################################################
            train_ratio = div_ratio/100

            if files_path is not None:
                with TemporaryDirectory() as temp_dir:
                    with io.BytesIO(files_path.read()) as zip_file:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            file_list = zip_ref.namelist()
                            if file_list:
                                for file_name in file_list:
                                    if not file_name.endswith('/'):
                                        if file_name.startswith("images"):
                                            with zip_ref.open(file_name) as file, open(os.path.join(temp_dir, os.path.basename(file_name)), 'wb') as target_file:
                                                target_file.write(file.read())
                    
                    # ランダムシャッフルして分割する
                    image_classification_random_shuffle(train_dir_path, val_dir_path, csv_path, temp_dir, train_ratio)
        
        elif img_toggle == "YOLO Format":
            current_datetime = datetime.now()
            
            # ディレクトリ名に使用するフォーマット(yyyyMMddhhmmss)
            directory_name_format = "%Y%m%d_%H%M%S"
            run_id=f"{current_datetime.strftime(directory_name_format)}"
            st.session_state.run_id=run_id
            
            # 画像分類ディレクトリの作成
            class_directory_name = f"{current_datetime.strftime(directory_name_format)}/pre_data"
            class_dir_path=os.path.join(parent_directory, class_directory_name)
            
            # 作成したいディレクトリのパスを作成
            train_directory_name = f"train"
            val_directory_name = f"val"
            yaml_directory_name = f"{current_datetime.strftime(directory_name_format)}/pre_data"
            
            # 完全なディレクトリパスを作成
            train_dir_path = os.path.join(class_dir_path, train_directory_name)
            val_dir_path = os.path.join(class_dir_path, val_directory_name)
            yaml_dir_path = os.path.join(parent_directory, yaml_directory_name)
            os.makedirs(class_dir_path, exist_ok=True)
            os.makedirs(train_dir_path, exist_ok=True)
            os.makedirs(val_dir_path, exist_ok=True)
            save_path=os.path.join(class_dir_path,use_model)

            if selection_model == "再学習" :
                with open(save_path,"wb") as f:
                    f.write(model_path.getbuffer())
            # elif selection_model == "１から学習":
            #     shutil.copy(path,class_dir_path)

            ###################################################        
            # データセットを学習用（train）と検証用（val）に分割する
            ###################################################
            train_ratio = div_ratio/100

            if zip_path is not None:
                with TemporaryDirectory() as temp_dir:
                    with io.BytesIO(zip_path.read()) as zip_file:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            file_list = zip_ref.namelist()
                            if file_list:
                                for file_name in file_list:
                                    if not file_name.endswith('/'):
                                        directory=os.path.dirname(file_name)
                                        subdirs=os.path.basename(directory)
                                        temp_save_path=os.path.join(temp_dir,subdirs)
                                        os.makedirs(temp_save_path, exist_ok=True)
                                        with zip_ref.open(file_name) as file, open(os.path.join(temp_save_path, os.path.basename(file_name)), 'wb') as target_file:
                                            target_file.write(file.read())
                    random_shuffle(train_dir_path, val_dir_path, temp_dir, train_ratio)


        #################
        # FaseAPI にpost
        #################
        result_name = f"{current_datetime.strftime(directory_name_format)}_{use_model}_学習結果"
        create_imgclass_yaml(train_dir_path, val_dir_path, epochs,batch,save,cache, device, workers, hsv_h, hsv_s, hsv_v, degrees, flipud, fliplr, result_name,yaml_dir_path,translate,scale,shear,perspective,mosaic,mixup,copy_paste)
        with st.spinner("画像分類の学習を行っています..."):
            data = {
                "run_id": f"{run_id}",
                "resume": f"{resume}",
                "use_model": f"{use_model}",
                }
            response=requests.post("http://yolo:81/train_img/",data=data)


        #######################
        # 応答結果に処理の分岐
        #######################
        if response.status_code == 200:
            st.write("学習完了")

            folder_path = f"{parent_directory }{run_id}/result_data/{run_id}_{os.path.basename(use_model)}_学習結果"
            # st.write(f"folder_path:{folder_path}")
            zip_path = f"{current_datetime.strftime(directory_name_format)}_{os.path.basename(use_model)}_学習結果.zip"            
            shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
            with open(zip_path, 'rb') as f:
                zip_data = f.read()
                st.session_state.zip_data=zip_data
                st.session_state.zip_filename=zip_path
            os.remove(zip_path)

        elif response.status_code == 503:
            st.warning("""
            メモリ不足のためアプリが実行できません。
            手順書に従って立ち下げ、立ち上げ操作を行ってください。
            """)
        else:

            st.error("予期せぬエラーが発生しました。")
            st.error("表示されているエラーメッセージをコピーのうえ、サポート窓口までご連絡ください。")

    #######################
    # 学習結果のダウンロード
    #######################
    if st.session_state.zip_data and st.session_state.zip_filename and st.session_state.view and st.session_state.run_id:
        run_id=st.session_state.run_id
        st.download_button(
                        label="画像分類の推論結果をダウンロード",
                        data=st.session_state.zip_data,
                        file_name=st.session_state.zip_filename,
                        mime='application/zip',
                        disabled=False,
                        on_click=lambda:on_download_click(parent_directory,run_id)
        )
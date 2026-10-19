import streamlit as st  
import os
import io
import zipfile
import shutil
import time
import threading
import pandas
import yaml
import requests
import random
import numpy as np
from datetime import datetime
from tempfile import NamedTemporaryFile
from tempfile import TemporaryDirectory


def para_change():
    """パラメータ初期化"""
    st.session_state['selection_device'] = "GPU"
    st.session_state['device'] = "0"
    st.session_state['load_size'] = 224
    st.session_state['input_size'] = 224
    st.session_state['net_model'] = "wide_resnet50_2"
    st.session_state['repo'] = "pytorch/vision:v0.11.3"
    st.session_state['sampling_ratio'] = 0.01
    st.session_state['n_neighbors'] = 3
    st.session_state['batch_toggle'] = True
    st.session_state['batch'] = 16
    st.session_state['coef_sigma'] = 5
    st.session_state['contour'] = None
    st.session_state['workers'] = 8
    st.session_state['num_workers'] = 1


def on_download_click(parent_directory,run_id):
    """後処理"""
    delete_path=os.path.join(parent_directory,run_id)
    shutil.rmtree(delete_path)
    st.session_state.view=False


def random_shuffle(train_dir_path, df, upload_dir, annotation):
    """画像ファイルと対応するラベルファイルのリストを取得"""
    csv_data = df["image"].to_list()
    csv_files = [os.path.basename(path) for path in csv_data]
    images_names = []
    for path in csv_files:
        images_names.append(path)
    tuple_list = list(zip(df["choice"], images_names))
    image_files = [f for f in os.listdir(upload_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 画像とラベルのファイル名が一致することを確認
    paired_files = [img for img in image_files if any(img in path for path in images_names)]

        # ファイルを各ディレクトリに移動
    for img in tuple_list:
        if img[0] ==annotation:
            shutil.move(os.path.join(upload_dir, img[1]), os.path.join(train_dir_path, img[1]))

def move_file(train_dir_path, temp_dir):
    """画像ファイルと対応するラベルファイルのリストを取得"""
    for file in os.listdir(temp_dir):
        shutil.move(os.path.join(temp_dir, file), os.path.join(train_dir_path, file))

def patchcore():
    """異常検知：学習ページ"""

    # 共有ディレクトリ
    parent_directory = "shared/train/"

    st.title("【学習】異常検知")

    if 'degrees' not in st.session_state:
        st.session_state.degrees = 0.0
    if'zip_data' not in st.session_state:
        st.session_state.zip_data=None
    if'zip_filename' not in st.session_state:
        st.session_state.zip_filename=None
    if'view' not in st.session_state:
        st.session_state.view=True
    if'run_id' not in st.session_state:
        st.session_state.run_id=None


    ########################
    # データセットをアップロード
    ########################
    # データセットの形式を選択
    img_options = ["LabelStadioFormat", "Anotated Format"]
    img_toggle = st.pills("推論するものをアップロードしてください。", img_options, default="Anotated Format", selection_mode="single")
    
    # LabelStadioFormatの場合
    if img_toggle == "LabelStadioFormat":
        st.markdown("###### import zipfile(zipファイルをアップロードしてください)")
        
        multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
        *対応する形式：.zip
        '''
        st.markdown(multi_img)
        
        files_path = st.file_uploader('import zipfile(zipファイルをアップロードしてください)', type=['zip'], accept_multiple_files=False, label_visibility="hidden")
        st.write("\n\n\n\n\n")

        # アノテーションラベルをアップロードする
        st.markdown("###### import csv(アノテーションラベルをアップロードしてください)")
        multi_csv = '''*注：日本語を含むファイルはアップロードしないでください。
             *対応する形式：.csv
        '''
        st.markdown(multi_csv)
        
        csv_path = st.file_uploader('import csv(アノテーションラベルをアップロードしてください)', type=['csv'], label_visibility="hidden")
        st.write("\n\n\n\n\n")

        if csv_path is not None:
            df = pandas.read_csv(csv_path)
            csv_label=df["choice"].unique().tolist()

    # Anotated Formatの場合
    elif img_toggle == "Anotated Format":
        st.markdown("###### 画像をアップロードしてください")
        multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
        *対応する形式：.zip
        '''
        st.markdown(multi_img)
        
        zip_path = st.file_uploader('import zipfile(zipファイルをアップロードしてください)', type=['zip'], label_visibility="hidden", key="hot_zip")        
        st.write("\n\n\n\n")


    #####################
    # オプション設定
    #####################
    with st.expander("オプション"):
        annotat_option = [""]
        
        device_options = ["CPU", "GPU"]
        selection_device = st.selectbox(
            "デバイス選択", device_options,
            index=device_options.index("GPU"),
            key="selection_device"
            )
        if selection_device == "CPU" :
            device = selection_device
        elif selection_device == "GPU" :
            gpu_options = {0:"0", 1:"1", 2:"2", 3:"3"}
            device = st.multiselect("使用するGPUの番号を選択してください。", gpu_options, default=0, key="device")
        
        if img_toggle == "LabelStadioFormat" and csv_path is not None:
            for label in csv_label:
                annotat_option.append(label)
            annotation=st.selectbox("正常のラベルを選択してください。",annotat_option)

    with st.expander("高度なオプション"):
        st.info("必要に応じて変更を加えてください。")
        mode = "train"
        batch = st.slider("batch:同時に処理される画像の数", min_value=0, max_value=1024, value=16)
        load_size = st.slider("load_size:読み込まれた画像のサイズ", min_value=0, max_value=500, value=224, key="load_size")
        input_size = st.slider("input_size:モデルに渡される画像のサイズ", min_value=0, max_value=500, value=224, key="input_size")
        model_options = ["wide_resnet50_2"]
        net_model = st.selectbox("model:ニューラルネットワークモデル選択", model_options, index=model_options.index("wide_resnet50_2"), key="net_model")
        repo="models"
        n_neighbors = st.slider("n_neighbors:検索窓の大きさ", min_value=0, max_value=100, value=3, key="n_neighbors")
        sampling_ratio = st.slider("sampling_ratio:サンプリング率", min_value=0.00, max_value=1.00, value=0.01, key="sampling_ratio")
        coef_sigma = st.slider("coef_sigma:閾値計算時のシグマ係数", min_value=0.0, max_value=20.0, value=5.0, key="coef_sigma")
        contour_toggle = st.toggle("contour:等高線マップの可視化(OFFでヒートマップで出力)", False, key="contour_toggle")
        if contour_toggle :
            contour = st.slider("contour:等高線マップの可視化", min_value=0.0, max_value=50.0, value=None, key="contour")
        else :
            contour = None
        if selection_device == "CPU" :
            num_workers = st.slider("num_workers:使用するCPUの数を選択(GPU選択時は0にしてください)", min_value=0, max_value=20, value=1, key="num_workers")
        elif selection_device == "GPU" :
            num_workers = st.slider("num_workers:使用するCPUの数を選択(GPU選択時は0にしてください)", min_value=0, max_value=20, value=0, key="num_workers")


    ##################################
    # パラメータ初期化と開始ボタンの配置
    ##################################
    col1, col2 = st.columns(2)

    with col2:
        st.button("パラメータ初期化", on_click=para_change)

    with col1:
        start_button = st.button('Start')

    if start_button:
        # 画像をtrainとvalに分割する
        train_ratio = 70/100
        
        
        # ディレクトリ名に使用するフォーマット(yyyyMMddhhmmss)
        current_datetime = datetime.now()
        directory_name_format = "%Y%m%d_%H%M%S"
        run_id=f"{current_datetime.strftime(directory_name_format)}"
        st.session_state.run_id=run_id
        
        # 作成したいディレクトリのパスを作成
        class_directory_name = f"{current_datetime.strftime(directory_name_format)}/pre_data"
        class_dir_path=os.path.join(parent_directory, class_directory_name)
        train_directory_name = f"{current_datetime.strftime(directory_name_format)}/pre_data"

        # 完全なディレクトリパスを作成
        train_dir_path = os.path.join(parent_directory, train_directory_name)
        os.makedirs(train_dir_path, exist_ok=True)
        os.chmod(f'{parent_directory}/{run_id}', 0o777)
        if img_toggle == "LabelStadioFormat":
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
                        random_shuffle(train_dir_path, df, temp_dir,annotation)
        elif img_toggle == "Anotated Format":
            if zip_path is not None:
                with TemporaryDirectory() as temp_dir:
                    with io.BytesIO(zip_path.read()) as zip_file:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            file_list = zip_ref.namelist()
                            if file_list:
                                for file_name in file_list:
                                    if not file_name.endswith('/'):
                                        with zip_ref.open(file_name) as file, open(os.path.join(temp_dir, os.path.basename(file_name)), 'wb') as target_file:
                                            target_file.write(file.read())
                    # ランダムシャッフルして分割する
                    move_file(train_dir_path, temp_dir)

        data_config ={
            # "annotation": annotation,
            "device": device,
            "mode": mode,
            "batch": batch,
            "load_size": load_size,
            "input_size": input_size,
            "model_options": model_options,
            "net_model": net_model,
            "repo": repo,
            "n_neighbors": n_neighbors,
            "sampling_ratio": sampling_ratio,
            "coef_sigma": coef_sigma,
            "contour": contour,
            "num_workers": num_workers,
        }
        data_yaml_path=os.path.join(class_dir_path,"data.yaml")
        with open(data_yaml_path,"w") as f:
            yaml.dump(data_config,f)

        data = { "run_id": f"{run_id}"}
        with st.spinner("異常検知の学習を行っています..."):
            response=requests.post("http://patchcore:82/train/",data=data)

        if response.status_code == 200:
            st.write("学習完了")
            folder_path = f"{parent_directory }{run_id}/result_data"
            zip_path = f"{current_datetime.strftime(directory_name_format)}_学習結果.zip"
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
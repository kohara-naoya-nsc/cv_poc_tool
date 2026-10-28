import streamlit as st  
import cv2
import os
import shutil
import time
import threading
from datetime import datetime
from tempfile import NamedTemporaryFile
from tempfile import TemporaryDirectory
import argparse
# Import third-party packages.
import numpy as np
import re
import yaml
import asyncio
import requests
import pandas
import shutil
import os
from pathlib import Path

def para_change():
    # st.session_state.clear()
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
    delete_path=os.path.join(parent_directory,run_id)
    shutil.rmtree(delete_path)
    st.session_state.view=False

def patchcore_prediction():

    st.title("【推論】異常検知")
    if'zip_data' not in st.session_state:
        st.session_state.zip_data=None
    if'zip_filename' not in st.session_state:
        st.session_state.zip_filename=None
    if'view' not in st.session_state:
        st.session_state.view=True
    if'run_id' not in st.session_state:
        st.session_state.run_id=None

    
    directory_name_format = "%Y%m%d_%H%M%S"
    parent_directory = "shared/predict/"
    # current_datetime = datetime.now()
    # run_id=f"{current_datetime.strftime(directory_name_format)}"
    # train_directory_name = f"{current_datetime.strftime(directory_name_format)}/pre_data"
    # train_dir_path = os.path.join(parent_directory, train_directory_name)

    if 'degrees' not in st.session_state:
        st.session_state.degrees = 0.0


    # 画像をアップロードする
    # 注釈を挿入
    img_options = ["動画", "画像"]
    img_toggle = st.pills("推論するものをアップロードしてください。", img_options, selection_mode="single")

    model_dir_path = "/home/itp0139/work/streamlit-app/app/model"
    # 画像をアップロードする
    # 注釈を挿入
    if img_toggle == "動画":
        st.markdown("###### 動画をアップロードしてください")
        multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
        *対応する形式：.avi, .mp4
        '''
        st.markdown(multi_img)
        file_path = st.file_uploader('import images(動画をアップロードしてください)', type=['avi', 'mp4'], accept_multiple_files=True, label_visibility="hidden")
        st.write("\n\n\n")
    elif img_toggle == "画像":
        st.markdown("###### 画像をアップロードしてください")
        multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
        *対応する形式：.jpg, .png, .jpeg
        '''
        st.markdown(multi_img)
        file_path = st.file_uploader('import images(画像をアップロードしてください)', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, label_visibility="hidden")
        st.write("\n\n\n\n\n")

    # モデルをアップロードする
    # "１から学習"であればyoloのデフォルトモデルから選択する
    # "再学習"であればユーザーがモデルを選択する
    # model_options = ["再学習", "１から学習"]
    # selection_model = st.pills("学習方法選択", model_options, selection_mode="single")
    # デフォルトモデルを使用
    # if selection_model == "再学習" :
    st.markdown("###### モデルをアップロードしてください(yyyymmdd_HHMMSS_モデル名.faiss)")
    multi_mdl = '''*注：日本語を含むファイルはアップロードしないでください。
         *対応する形式：.faiss
    '''
    st.markdown(multi_mdl)
    model_path = st.file_uploader('import model(yyyymmdd_HHMMSS_モデル名.faissをアップロードしてください)', type=['faiss'], label_visibility="hidden")
    # if model_path :
    #     use_model = f"{model_dir_path}/{model_path.name}"
    # ユーザー選択モデルを使用
    # elif selection_model == "１から学習":
    #     select_model = st.selectbox("デフォルトのモデルを選択してください。",
    #                                 ("yolov8n.pt(6MB)", "yolov8s.pt(22MB)", "yolov8m.pt(52MB)", "yolov8l.pt(87MB)", "yolov8x.pt(137MB)"))
    #     use_model = f"./model/{select_model.rsplit('(',1)[0]}"
    st.write("\n\n\n")
    # multi_csv = '''*注：日本語を含むファイルはアップロードしないでください。
    #      *対応する形式：.csv
    # '''
    # st.markdown(multi_csv)
    # csv_path = st.file_uploader('import csv(アノテーションラベルをアップロードしてください)', type=['csv'], label_visibility="hidden")
    # st.write("\n\n\n\n\n")
    
    # if csv_path is not None:
    #     df = pandas.read_csv(csv_path)
    #     csv_label=df["choice"].unique().tolist()

    
    

    # ハイパーパラメータを取得
    # オプション
    with st.expander("オプション"):
        annotat_option = [""]
        device_options = ["CPU", "GPU"]
        selection_device = st.selectbox("デバイス選択", device_options, index=device_options.index("GPU"), key="selection_device")
        if selection_device == "CPU" :
            device = selection_device
        elif selection_device == "GPU" :
            gpu_options = {0:"0", 1:"1", 2:"2", 3:"3"}
            device = st.multiselect("使用するGPUの番号を選択してください。", gpu_options, default=0, key="device")
        # if csv_path is not None:
        #     for label in csv_label:
        #         annotat_option.append(label)
        #     annotation=st.selectbox("正常のラベルを選択してください。",annotat_option)

    with st.expander("高度なオプション"):
        st.info("必要に応じて変更を加えてください。")
        mode = "train"
        batch = st.slider("batch:同時に処理される画像の数", min_value=0, max_value=1024, value=16)
        # threshold = st.slider("threshold:確信度の閾値", min_value=0.0, max_value=10.0, value=7.0, key="threshold")
        load_size = st.slider("load_size:読み込まれた画像のサイズ", min_value=0, max_value=500, value=224, key="load_size")
        input_size = st.slider("input_size:モデルに渡される画像のサイズ", min_value=0, max_value=500, value=224, key="input_size")
        model_options = ["wide_resnet50_2"]
        net_model = st.selectbox("model:ニューラルネットワークモデル選択", model_options, index=model_options.index("wide_resnet50_2"), key="net_model")
        repo="models"
        n_neighbors = st.slider("n_neighbors:検索窓の大きさ", min_value=0, max_value=100, value=3, key="n_neighbors")
        sampling_ratio = st.slider("sampling_ratio:サンプリング率", min_value=0.00, max_value=1.00, value=0.01, key="sampling_ratio")
        coef_sigma = st.slider("coef_sigma:しきい値計算時のシグマ係数", min_value=0.0, max_value=20.0, value=5.0, key="coef_sigma")
        # contour = st.slider("contour:等高線マップの可視化", min_value=0.0, max_value=50.0, value=None, key="contour")
        contour_toggle = st.toggle("contour:等高線マップの可視化(OFFでヒートマップで出力)", False, key="contour_toggle")
        if contour_toggle :
            contour = st.slider("contour:等高線マップの可視化", min_value=0.0, max_value=50.0, value=None, key="contour")
        else :
            contour = None
        if selection_device == "CPU" :
            num_workers = st.slider("num_workers:使用するCPUの数を選択(GPU選択時は0にしてください)", min_value=0, max_value=20, value=1, key="num_workers")
        elif selection_device == "GPU" :
            num_workers = st.slider("num_workers:使用するCPUの数を選択(GPU選択時は0にしてください)", min_value=0, max_value=20, value=0, key="num_workers")

    # ２つの列を作成
    col1, col2 = st.columns(2)

    with col2:
        st.button("パラメータ初期化", on_click=para_change)

    with col1:
        start_button = st.button('Start')

    if start_button:
        current_datetime = datetime.now()
        run_id=f"{current_datetime.strftime(directory_name_format)}"
        st.session_state.run_id=run_id
        train_directory_name = f"{current_datetime.strftime(directory_name_format)}/pre_data"
        train_dir_path = os.path.join(parent_directory, train_directory_name)
        os.makedirs(train_dir_path, exist_ok=True)
        os.chmod(f'{parent_directory}/{run_id}', 0o777)
        if file_path is not None:
            for file in file_path:
                save_path=os.path.join(train_dir_path,file.name)

                with open(save_path,"wb") as f:
                    f.write(file.getbuffer())

        if model_path is not None:
            save_path=os.path.join(train_dir_path,model_path.name)

            with open(save_path,"wb") as f:
                f.write(model_path.getbuffer())


        data_config ={
            # "annotation": annotation,
            "device": device,
            "mode": mode,
            "batch": batch,
            # "threshold": threshold,
            "device": device,
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
        data_yaml_path=os.path.join(train_dir_path,"data.yaml")
        with open(data_yaml_path,"w") as f:
            yaml.dump(data_config,f)
        
        data = { "run_id": f"{run_id}"}
        with st.spinner("異常検知の推論を行っています..."):
            if img_toggle == "画像" :
                response=requests.post("http://patchcore:82/predict/image/",data=data)
            elif img_toggle == "動画" :
                response=requests.post("http://patchcore:82/predict/movie/",data=data)

        if response.status_code == 200:
            st.write("推論完了")
            if img_toggle == "画像" :
                folder_path = f"{parent_directory }{run_id}/result_data"
                zip_path = f"{current_datetime.strftime(directory_name_format)}_{net_model}_推論結果.zip"
                
                shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
                    # フォルダパスをPathオブジェクトで指定
                image_dir = Path(folder_path)
                    # st.write(f"image_dir:{image_dir}")
    
                    # 画像拡張子を含むファイルを取得
                # image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    
                    # または、正規表現で拡張子を指定（より柔軟）
                image_files = [f for f in image_dir.iterdir() if f.is_file() and re.match(r".*\.(jpg|jpeg|png)$", f.name, re.IGNORECASE)]
    
                    # print("取得した画像ファイル一覧:")
                image_files.sort()

            # 画像リストと初期選択インデックスをセッション状態に保存
                st.session_state.image_files = image_files
                st.session_state.selected_index = 0
                with open(zip_path, 'rb') as f:
                    zip_data = f.read()
                    st.session_state.zip_data=zip_data
                    st.session_state.zip_filename=zip_path
                os.remove(zip_path)
                # for idx,img in enumerate(image_files):
                #         # st.write(f"filename:{str(img)}")
                #     st.image(str(img), caption='異常検知結果')
    
                # with open(zip_path, 'rb') as f:
                #     zip_data = f.read()
                #     st.download_button(
                #         label="異常検知の推論結果をダウンロード",
                #         data=zip_data,
                #         file_name=zip_path,
                #         mime='application/zip',
                #         disabled=False,
                #         key="zip_download_initial"
                #     )
                # os.remove(zip_path)
            elif img_toggle == "動画" :
                folder_path = f"{parent_directory }{run_id}/result_data"
                zip_path = f"{current_datetime.strftime(directory_name_format)}_{net_model}_推論結果.zip"
                shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
                with open(zip_path, 'rb') as f:
                    zip_data=f.read()
                    st.session_state.zip_data=zip_data
                    st.session_state.zip_filename=zip_path
                os.remove(zip_path)

                    # フォルダパスをPathオブジェクトで指定
                image_dir = Path(folder_path)
                    # 正規表現で拡張子を指定（より柔軟）
                video_files = [f for f in image_dir.iterdir() if f.is_file() and re.match(r".*\.(avi|mp4)$", f.name, re.IGNORECASE)]
                for idx,video_path in enumerate(video_files):
                    stframe=st.empty()
                    stop_flag=False
                    if st.button("動画を停止"):
                        stframe.empty()
                    #     st.download_button(
                    #     label="異常検知の推論結果をダウンロード",
                    #     data=st.session_state.zip_data,
                    #     file_name=st.session_state.zip_filename,
                    #     mime='application/zip',
                    #     disabled=False,
                    #     key="zip_download_video_{idx}",
                    #     # on_click=on_download_click(parent_directory,run_id)
                    # )

                        # 動画再生スレッド
                    async def play_video(video_path, stframe, stop_flag):
                        try:
                            cap = cv2.VideoCapture(str(video_path))
                            if not cap.isOpened():
                                st.error("動画を開けませんでした。")
                                return

                            while (cap.isOpened() and not stop_flag):
                                ret, frame = cap.read()
                                if not ret:
                                    break

                                # frame = cv2.resize(frame_val, (640, 360))  # リサイズ
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                stframe.image(frame_rgb)
                                await asyncio.sleep(0.033)  # 約30fps

                            cap.release()
                            stframe.empty()
                            st.session_state.video_played = True
                            st.write("動画再生が終了しました。")
                        except Exception as e:
                            print(e)

                    # 実行処理
                    stframe.empty()
                    asyncio.run(play_video(video_path, stframe, stop_flag))

        elif response.status_code == 503:
            st.warning("""
            メモリ不足のためアプリが実行できません。
            手順書に従って立ち下げ、立ち上げ操作を行ってください。
            """)
        else:
            st.error("予期せぬエラーが発生しました。")
            st.error("表示されているエラーメッセージをコピーのうえ、サポート窓口までご連絡ください。")

    # 画像モードかつ画像リストがセッション状態にある場合のみ表示
    if img_toggle == "画像" and "image_files" in st.session_state and st.session_state.view:
        image_files = st.session_state.image_files
        image_options = [img.name for img in image_files]

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
        st.image(str(selected_image), caption='異常検知結果')

    if img_toggle=="画像" and st.session_state.zip_data and st.session_state.zip_filename and st.session_state.view and st.session_state.run_id:
        run_id=st.session_state.run_id
        st.download_button(
                        label="異常検知の推論結果をダウンロード",
                        data=st.session_state.zip_data,
                        file_name=st.session_state.zip_filename,
                        mime='application/zip',
                        disabled=False,
                        on_click=lambda: on_download_click(parent_directory,run_id)
        )
    elif img_toggle=="動画" and st.session_state.zip_data and st.session_state.zip_filename and st.session_state.view and st.session_state.run_id:
        run_id=st.session_state.run_id
        st.download_button(
                        label="異常検知の推論結果をダウンロード",
                        data=st.session_state.zip_data,
                        file_name=st.session_state.zip_filename,
                        mime='application/zip',
                        disabled=False,
                        on_click=lambda: on_download_click(parent_directory,run_id)
        )
import streamlit as st  
import cv2
import numpy as np
import time
import os
import requests
import shutil
import re
from PIL import Image
from pathlib import Path
from datetime import datetime
from tempfile import NamedTemporaryFile
from tempfile import TemporaryDirectory
# from ultralytics import YOLO
import asyncio

def on_download_click(parent_directory,run_id):
    """学習完了後の後処理"""
    delete_path=os.path.join(parent_directory,run_id)
    shutil.rmtree(delete_path)
    st.session_state.view=False


def image_classification_prediction(show_title=True):
    """画像分類：推論ページ"""
    # 共有ディレクトリ
    parent_directory = "shared/predict/"


    if show_title:
        st.title("【推論】画像分類")

    if'zip_data' not in st.session_state:
        st.session_state.zip_data=None
    if'zip_filename' not in st.session_state:
        st.session_state.zip_filename=None
    if'view' not in st.session_state:
        st.session_state.view=True
    if'run_id' not in st.session_state:
        st.session_state.run_id=None


    # 推論対象の選択
    img_options = ["動画", "画像"]
    img_toggle = st.pills("推論するものをアップロードしてください。", img_options, selection_mode="single")

    # model_dir_path = "/home/itp0139/work/streamlit-app/app/model"
    ############################
    # 推論対象のアップロード
    ############################
    # 動画の場合
    if img_toggle == "動画":
        st.markdown("###### 動画をアップロードしてください")
        multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
        *対応する形式：.avi, .mp4
        '''
        st.markdown(multi_img)
        
        file_path = st.file_uploader(
            'import images(動画をアップロードしてください)',
            type=['avi', 'mp4'],
            accept_multiple_files=True,
            label_visibility="hidden"
            )
        st.write("\n\n\n")

    # 画像の場合
    elif img_toggle == "画像":
        st.markdown("###### 画像をアップロードしてください")
        multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
        *対応する形式：.jpg, .png, .jpeg
        '''
        st.markdown(multi_img)
        
        file_path = st.file_uploader(
            'import images(画像をアップロードしてください)',
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            label_visibility="hidden"
            )
        st.write("\n\n\n\n\n")


    ########################
    # モデルのアップロード
    ########################
    st.markdown("###### モデルをアップロードしてください(yyyymmdd_HHMMSS_モデル名.pt)")
    multi_mdl = '''*注：日本語を含むファイルはアップロードしないでください。
         *対応する形式：.pt
    '''
    st.markdown(multi_mdl)
    
    model_path = st.file_uploader(
        'import model(yyyymmdd_HHMMSS_モデル名.ptをアップロードしてください)',
        type=['pt'],
        label_visibility="hidden"
        )
    st.write("\n\n\n")


    ###################
    # 推論の開始ボタン
    ###################
    start_button = st.button('Start')
    if start_button:
        # 現在の日付と時刻を取得
        current_datetime = datetime.now()
        
        # ディレクトリ名に使用するフォーマット(yyyyMMddhhmmss)
        directory_name_format = "%Y%m%d_%H%M%S"
        run_id=f"{current_datetime.strftime(directory_name_format)}"
        st.session_state.run_id=run_id
        
        # 作成したいディレクトリのパスを作成
        train_directory_name = f"{current_datetime.strftime(directory_name_format)}/pre_data"
        
        # 完全なディレクトリパスを作成
        train_dir_path = os.path.join(parent_directory, train_directory_name)
        os.makedirs(train_dir_path, exist_ok=True)
        os.chmod(f'{parent_directory}/{run_id}', 0o777)

        ######################################################
        # アップロードされた推論対象とモデルを共有ディレクトリに保存する
        ######################################################
        if file_path is not None:
            for file in file_path:
                file_save_path = os.path.join(train_dir_path, file.name)

                with open(file_save_path,"wb") as f:
                    f.write(file.getbuffer())

        if model_path is not None:
            model_save_path = os.path.join(train_dir_path, model_path.name)

            with open(model_save_path,"wb") as f:
                f.write(model_path.getbuffer())


        ###################
        # 推論API の呼び出し
        ###################
        data = {
            "run_id": f"{run_id}",
            "model_path": f"{model_save_path}",
            }
        with st.spinner("画像分類の推論を行っています..."):
            if img_toggle == "画像" :
                response=requests.post("http://yolo:81/predict_img/image/",data=data)
            elif img_toggle == "動画" :
                response=requests.post("http://yolo:81/predict_img/movie/",data=data)


        ####################
        # 応答結果に処理の分岐
        ####################
        if response.status_code == 200:
            st.write("推論完了")
            if img_toggle == "画像" :
                folder_path = f"{parent_directory }{run_id}/result_data/{run_id}_{model_path.name}_推論結果"
                zip_path = f"{current_datetime.strftime(directory_name_format)}_{model_path.name}_推論結果.zip"
                
                shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
                # フォルダパスをPathオブジェクトで指定
                image_dir = Path(folder_path)
    
                # 正規表現で拡張子を指定（より柔軟）
                image_files = [f for f in image_dir.iterdir() if f.is_file() and re.match(r".*\.(jpg|jpeg|png)$", f.name, re.IGNORECASE)]
                image_files.sort()

                # 画像リストと初期選択インデックスをセッション状態に保存
                st.session_state.image_files = image_files
                st.session_state.selected_index = 0
                with open(zip_path, 'rb') as f:
                    zip_data = f.read()
                    st.session_state.zip_data=zip_data
                    st.session_state.zip_filename=zip_path
                os.remove(zip_path)

            elif img_toggle == "動画" :
                folder_path = f"{parent_directory}{run_id}/result_data/{run_id}_{model_path.name}_推論結果"
                zip_path = f"{current_datetime.strftime(directory_name_format)}_{model_path.name}_推論結果.zip"
                shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
                
                with open(zip_path, 'rb') as f:
                    zip_data=f.read()
                    st.session_state.zip_data=zip_data
                    st.session_state.zip_filename=zip_path
                    zip_data = f.read()
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

                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                stframe.image(frame_rgb)
                                await asyncio.sleep(0.033)  # 約30fps

                            cap.release()
                            stframe.empty()
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
        st.image(str(selected_image), caption='画像分類結果')


    ########################
    # 推論結果のダウンロード
    ########################
    if img_toggle=="画像" and st.session_state.zip_data and st.session_state.zip_filename and st.session_state.view and st.session_state.run_id:
        run_id=st.session_state.run_id
        st.download_button(
                        label="画像分類の推論結果をダウンロード",
                        data=st.session_state.zip_data,
                        file_name=st.session_state.zip_filename,
                        mime='application/zip',
                        disabled=False,
                        on_click=lambda:on_download_click(parent_directory,run_id)
        )
        
    elif img_toggle=="動画" and st.session_state.zip_data and st.session_state.zip_filename and st.session_state.view and st.session_state.run_id:
        run_id=st.session_state.run_id
        st.download_button(
                        label="画像分類の推論結果をダウンロード",
                        data=st.session_state.zip_data,
                        file_name=st.session_state.zip_filename,
                        mime='application/zip',
                        disabled=False,
                        on_click=lambda: on_download_click(parent_directory,run_id)
        )
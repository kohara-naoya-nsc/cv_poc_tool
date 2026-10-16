import streamlit as st
import cv2
import random
import os
import zipfile
import shutil
from io import BytesIO

def random_frame(num_image: int, total_frame: int) -> list:
    """
    指定したフレーム数分、ランダムにフレームNoを指定する関数
    Args:
        num_image (int): 取得したいフレームの数
        total_frame (int): 動画の総フレーム数
    Returns:
        list: ランダムに選ばれたフレーム番号のリスト
    """
    random_frame = sorted(random.sample(range(total_frame), num_image))
    return random_frame

def interval_frame(interval: int, total_frame: int) -> list:
    """
    指定したフレーム数分、間隔を空けてフレームNoを指定する関数
    Args:
        interval (int): 取得したいフレームの間隔
        total_frame (int): 動画の総フレーム数
    Returns:
        list: 指定した間隔で選択されたフレーム番号のリスト
    """
    return list(range(0, total_frame, interval))


def process_video(uploaded_file, method, num_frames, interval) -> str:
    """
    アップロードされた動画から指定した数のランダムなフレームを抽出し、ZIPファイルに保存する関数
    Args:
        uploaded_file (streamlit.uploaded_file): アップロードされた動画ファイル
        num_frames (int): 取得したいフレームの数
    Returns:
        str: 生成されたZIPファイルの名前
    """
    # 一時ディレクトリを作成
    temp_dir = 'temp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # ファイルを一時的に保存
    video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    cap = cv2.VideoCapture(video_path)
    # 動画の総フレーム数を取得
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if method == 'ランダム抽出':
        # ランダムなフレーム番号を生成
        target_frames = random_frame(num_frames, total_frame)
    elif method == '指定した間隔で抽出':
        target_frames = interval_frame(interval, total_frame)
    else:
        st.error("不正な取得方法が選択されました。")
        return ""
    
    frame_cnt = 0
    saved_cnt = 0
    for i in range(total_frame):
        ret, frame_org = cap.read()
        if not ret:
            break
        if frame_cnt in target_frames:
            # フレームを保存
            frame_filename = os.path.join(temp_dir, f'frame_{frame_cnt}.png')
            cv2.imwrite(frame_filename, frame_org)
            saved_cnt += 1
        frame_cnt += 1
    
    cap.release()
    
    # ZIPファイルにフレーム画像を保存
    zip_filename = 'frames.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.png'):
                    zipf.write(os.path.join(root, file), file)
    
    # 一時ディレクトリを削除
    shutil.rmtree(temp_dir)
    
    return zip_filename

def tool_capture_app():
    # Streamlit アプリケーションの設定
    title = '<p style="font-size: 40px; font-weight: bold;">【前処理ツール】動画→画像変換</p>'
    st.markdown(title, unsafe_allow_html=True)
    st.write("アップロードした動画ファイルから任意の画像フレームをキャプチャーします。")
    
    # 動画のアップロード
    st.markdown("###### import mp4.avi(動画ファイルをアップロードしてください)")
    st.markdown('''*注：日本語を含むファイルはアップロードしないでください。
        *対応する形式：.mp4, .avi
        ''')
    uploaded_file = st.file_uploader("動画ファイルを選択してください (mp4, avi)", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        method = st.radio("フレームの取得方法を選択してください", ("ランダム抽出", "指定した間隔で抽出"))

        if method == "ランダム抽出":
            num_frames = st.number_input("取得したいフレーム数を指定してください", min_value=1, value=30)
            interval = None
        else:
            interval = st.number_input("何フレームごとに取得しますか？", min_value=1, value=30)
            num_frames = None

        
        if st.button("処理開始"):
            with st.spinner("動画から画像データを抽出しています..."):
                zip_filename = process_video(uploaded_file, method, num_frames, interval)
            
            if zip_filename:
                # ZIPファイルのダウンロードリンク
                with open(zip_filename, "rb") as f:
                    bytes_data = f.read()
                    st.download_button(label="取得した画像をダウンロード", data=BytesIO(bytes_data), file_name="frames.zip")
                # ZIPファイルを削除
                os.remove(zip_filename)
    
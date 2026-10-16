import streamlit as st  
# import cv2
import numpy as np
import os
from PIL import Image
# from ultralytics import YOLO


# フォルダ内の画像を全て取得する関数
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswhith(".png") or filename.endswhith(".jpg") or filename.endswhith(".jpeg"):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images

# フォルダ選択ダイアログを表示する関数
@st.dialog("フォルダ選択")
def select_folder():
    folder_path = st.text_input("フォルダのパスを入力してください")
    if st.button("選択"):
        st.session_state.folder_path = folder_path
        st.rerun()

def test_folder_page():

    st.title("Test Folder Page")


    if "folder_path" not in st.session_state:
        if st.button("フォルダを選択"):
            select_folder()
    else:
        folder_path = st.session_state.folder_path
        images = load_images_from_folder(folder_path)
        if images:
            for img in images:
                st.image(img, caption="アップロードされた画像", use_column_width=True)
        else:
            st.write("画像が見つかりませんでした")
        # ファイルをバイト列として読み込む

        # バイト列から画像をデコードする
        # img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # 画像を表示する
        # BGR チャンネルで表示（一般的な画像フォーマットはRGBであるため）
        # st.image(img, channels="BGR")

    #start_button = st.button('Start')

    #if start_button:

        #model = YOLO('/home/itp0139/work/streamlit-app/app/model/yolov8n.pt')
        #results = model(img)

        #result = results[0].plot()
        #st.image(result, channels="BGR")


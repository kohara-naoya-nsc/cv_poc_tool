import streamlit as st  

# 学習・推論のページ
# 画像解析Pocツールのページ
from pages.home_page import home_page
from pages.Object_Detection_page import object_detection
from pages.Image_Classification_page import image_classification
from pages.Patchcore_page import patchcore
from pages.Patchcore_Prediction_page import patchcore_prediction
from pages.Object_Prediction_page import object_prediction
from pages.Image_Classification_Prediction_page import image_classification_prediction
from pages.paddleocr_page import paddleocr_pre

# 画像認識FWのページ
from pages.Object_Detection_YOLOX_page import object_detection_yolox
from pages.Object_Prediction_YOLOX_page import object_prediction_yolox
# toolのページ
from pages.tool_reshape_app import tool_reshape_app
from pages.tool_capture_app import tool_capture_app


class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({"title": title, "function": func})

    def run(self):
        st.sidebar.markdown("## メインメニュー")
        st.sidebar.markdown("ページを選択してください:")

        # サイドバーにボタンを配置
        for app in self.apps:
            if st.sidebar.button(app["title"], key=app["title"]):
                st.session_state.current_page = app["title"]
        
        st.sidebar.markdown("---")

        st.sidebar.markdown(
            f'<a href="http://0.0.0.0:8080" target="_blank" style="color: blue; text-decoration: underline;">【領域検出】推論</a>',
            unsafe_allow_html=True
        )
        st.sidebar.markdown(
            f'<a href="http://0.0.0.0:8080" target="_blank" style="color: blue; text-decoration: underline;">【データセット作成】画像分類</a>',
            unsafe_allow_html=True
        )
        st.sidebar.markdown(
            f'<a href="http://0.0.0.0:8080" target="_blank" style="color: blue; text-decoration: underline;">【データセット作成】物体検出</a>',
            unsafe_allow_html=True
        )

        # 現在のページをセッション状態で管理
        if 'current_page' not in st.session_state:
            st.session_state.current_page = self.apps[0]["title"]

        # 選択されたページを実行
        for app in self.apps:
            if app["title"] == st.session_state.current_page:
                app["function"]()
                break


# メインアプリケーションの構築
app = MultiApp()
app.add_app("ホームページ", home_page)
app.add_app("【前処理】射影変換(画角補正)", tool_reshape_app)
app.add_app("【前処理】動画→画像変換", tool_capture_app)
app.add_app("【画像分類】学習", image_classification)
app.add_app("【画像分類】推論", image_classification_prediction)
app.add_app("【物体検出】学習", object_detection)
app.add_app("【物体検出】推論", object_prediction)
app.add_app("【異常検知】学習", patchcore)
app.add_app("【異常検知】推論", patchcore_prediction)
app.add_app("【文字認識】推論", paddleocr_pre)
app.add_app("【物体検出(YOLOX)】学習", object_detection_yolox)
app.add_app("【物体検出(YOLOX)】推論", object_prediction_yolox)

app.run()

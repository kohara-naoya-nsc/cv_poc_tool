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
from pages.tool_gzo_fw import tool_gzo_fw
from pages.Object_Detection_YOLOX_page import object_detection_yolox
from pages.Object_Prediction_YOLOX_page import object_prediction_yolox
from pages.Image_Classification_TF2_page import image_classification_tf2
from pages.Image_Classification_Prediction_TF2_page import image_classification_prediction_tf2
from pages.Semantic_Segmentation import semantic_segmentation
from pages.Semantic_Segmentation_Prediction import segmentation_prediction

# toolのページ
from pages.tool_reshape_app import tool_reshape_app
from pages.tool_capture_app import tool_capture_app


class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func, category):
        self.apps.append({"title": title, "function": func, "category": category})

    def run(self):
        st.sidebar.markdown("## メインメニュー")
        st.sidebar.markdown("ページを選択してください:")

        # 現在のページをセッション状態で管理
        if 'current_page' not in st.session_state:
            st.session_state.current_page = self.apps[0]["title"]

        categories = dict.fromkeys(app["category"] for app in self.apps)
        for category in categories:
            category_apps = [
                app for app in self.apps if app["category"] == category
            ]
            is_current_category = any(
                app["title"] == st.session_state.current_page
                for app in category_apps
            )
            with st.sidebar.expander(category, expanded=is_current_category):
                for app in category_apps:
                    if st.button(app["title"], key=app["title"]):
                        st.session_state.current_page = app["title"]

        with st.sidebar.expander("関連リンク"):
            st.markdown(
                '<a href="http://0.0.0.0:8080" target="_blank" '
                'style="color: blue; text-decoration: underline;">'
                '【領域検出】推論</a>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<a href="http://0.0.0.0:8080" target="_blank" '
                'style="color: blue; text-decoration: underline;">'
                '【データセット作成】画像分類</a>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<a href="http://0.0.0.0:8080" target="_blank" '
                'style="color: blue; text-decoration: underline;">'
                '【データセット作成】物体検出</a>',
                unsafe_allow_html=True
            )

        # 選択されたページを実行
        for app in self.apps:
            if app["title"] == st.session_state.current_page:
                app["function"]()
                break


# メインアプリケーションの構築
app = MultiApp()
app.add_app("ホームページ", home_page, "ホーム")
app.add_app("【前処理】射影変換(画角補正)", tool_reshape_app, "前処理")
app.add_app("【前処理】動画→画像変換", tool_capture_app, "前処理")
app.add_app("【画像分類】学習", image_classification, "画像分類")
app.add_app("【画像分類】推論", image_classification_prediction, "画像分類")
app.add_app("【物体検出】学習", object_detection, "物体検出")
app.add_app("【物体検出】推論", object_prediction, "物体検出")
app.add_app("【異常検知】学習", patchcore, "異常検知")
app.add_app("【異常検知】推論", patchcore_prediction, "異常検知")
app.add_app("【文字認識】推論", paddleocr_pre, "文字認識")
app.add_app("【前処理】画像認識FW", tool_gzo_fw, "前処理")
app.add_app("【物体検出(YOLOX)】学習", object_detection_yolox, "物体検出")
app.add_app("【物体検出(YOLOX)】推論", object_prediction_yolox, "物体検出")
app.add_app("【画像分類(TF2)】学習", image_classification_tf2, "画像分類")
app.add_app("【画像分類(TF2)】推論", image_classification_prediction_tf2, "画像分類")
app.add_app("【領域検知】学習", semantic_segmentation, "領域検知")
app.add_app("【領域検知】推論", segmentation_prediction, "領域検知")

app.run()

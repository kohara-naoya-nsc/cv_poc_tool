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


# toolのページ
from pages.tool_reshape_app import tool_reshape_app
from pages.tool_capture_app import tool_capture_app


class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func, group=None, category=None):
        self.apps.append({
            "id": (group, category, title),
            "title": title, "function": func, "group": group, "category": category
        })

    def select_page(self, page_id):
        st.session_state.current_page = page_id

    def run(self):
        st.sidebar.markdown("## メインメニュー")

        for app in self.apps:
            if app["group"] is None:
                st.sidebar.button(
                    app["title"], key=f"navigation:{app['id']!r}",
                    on_click=self.select_page, args=(app["id"],)
                )

        groups = dict.fromkeys(app["group"] for app in self.apps if app["group"] is not None)
        for group in groups:
            with st.sidebar.expander(group):
                group_apps = [app for app in self.apps if app["group"] == group]
                categories = dict.fromkeys(app["category"] for app in group_apps)
                for category in categories:
                    with st.expander(category):
                        for app in group_apps:
                            if app["category"] == category:
                                st.button(
                                    app["title"], key=f"navigation:{app['id']!r}",
                                    on_click=self.select_page, args=(app["id"],)
                                )
        
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
        if st.session_state.get("current_page") not in [app["id"] for app in self.apps]:
            st.session_state.current_page = self.apps[0]["id"]

        # 選択されたページを実行
        for app in self.apps:
            if app["id"] == st.session_state.current_page:
                app["function"]()
                break


# メインアプリケーションの構築
app = MultiApp()
app.add_app("ホームページ", home_page)
app.add_app("射影変換(画角補正)", tool_reshape_app, "画像解析PoCツール", "前処理")
app.add_app("動画→画像変換", tool_capture_app, "画像解析PoCツール", "前処理")
app.add_app("学習", image_classification, "画像解析PoCツール", "画像分類")
app.add_app("推論", image_classification_prediction, "画像解析PoCツール", "画像分類")
app.add_app("学習", object_detection, "画像解析PoCツール", "物体検出")
app.add_app("推論", object_prediction, "画像解析PoCツール", "物体検出")
app.add_app("学習", patchcore, "画像解析PoCツール", "異常検知")
app.add_app("推論", patchcore_prediction, "画像解析PoCツール", "異常検知")
app.add_app("推論", paddleocr_pre, "画像解析PoCツール", "文字認識")
app.add_app("データセット変換", tool_gzo_fw, "画像認識FW", "前処理")
app.add_app("学習", object_detection_yolox, "画像認識FW", "物体検出")
app.add_app("推論", object_prediction_yolox, "画像認識FW", "物体検出")
app.add_app("学習", image_classification_tf2, "画像認識FW", "画像分類")
app.add_app("推論", image_classification_prediction_tf2, "画像認識FW", "画像分類")

app.run()

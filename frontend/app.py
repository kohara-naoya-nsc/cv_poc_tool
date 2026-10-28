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

    def add_app(self, title, func, category, subcategory=None):
        self.apps.append({
            "id": f"page_{len(self.apps)}",
            "title": title,
            "function": func,
            "category": category,
            "subcategory": subcategory,
        })

    def run(self):
        st.sidebar.markdown("## メインメニュー")

        # 現在のページをセッション状態で管理
        if 'current_page' not in st.session_state:
            st.session_state.current_page = self.apps[0]["id"]
        elif not any(
            app["id"] == st.session_state.current_page for app in self.apps
        ):
            previous_page = next(
                (
                    app for app in self.apps
                    if app["title"] == st.session_state.current_page
                ),
                self.apps[0],
            )
            st.session_state.current_page = previous_page["id"]

        home_app = next(app for app in self.apps if app["category"] == "ホーム")
        if st.sidebar.button(home_app["title"], key=home_app["id"]):
            st.session_state.current_page = home_app["id"]

        categories = dict.fromkeys(
            app["category"] for app in self.apps if app["category"] != "ホーム"
        )
        for category in categories:
            category_apps = [app for app in self.apps if app["category"] == category]
            is_current_category = any(
                app["id"] == st.session_state.current_page
                for app in category_apps
            )
            with st.sidebar.expander(category, expanded=is_current_category):
                subcategories = dict.fromkeys(
                    app["subcategory"]
                    for app in category_apps
                    if app["subcategory"] is not None
                )
                if subcategories:
                    for subcategory in subcategories:
                        subcategory_apps = [
                            app for app in category_apps
                            if app["subcategory"] == subcategory
                        ]
                        is_current_subcategory = any(
                            app["id"] == st.session_state.current_page
                            for app in subcategory_apps
                        )
                        with st.expander(
                            subcategory, expanded=is_current_subcategory
                        ):
                            for app in subcategory_apps:
                                if st.button(app["title"], key=app["id"]):
                                    st.session_state.current_page = app["id"]
                else:
                    for app in category_apps:
                        if st.button(app["title"], key=app["id"]):
                            st.session_state.current_page = app["id"]

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            '<a href="http://localhost:8080" target="_blank" style="color: blue; text-decoration: underline;">アノテーション（Label Studio）</a>',
            unsafe_allow_html=True
        )
        st.sidebar.caption("*従来のデータセット作成")

        # 選択されたページを実行
        for app in self.apps:
            if app["id"] == st.session_state.current_page:
                app["function"]()
                break


# メインアプリケーションの構築
app = MultiApp()
app.add_app("ホームページ", home_page, "ホーム")
app.add_app("射影変換(画角補正)", tool_reshape_app, "前処理")
app.add_app("動画→画像変換", tool_capture_app, "前処理")
app.add_app("データ変換", tool_gzo_fw, "前処理")

app.add_app("学習", image_classification, "画像分類", "YOLOv8")
app.add_app("推論", image_classification_prediction, "画像分類", "YOLOv8")
app.add_app("学習", image_classification_tf2, "画像分類", "VGG")
app.add_app("推論", image_classification_prediction_tf2, "画像分類", "VGG")

app.add_app("学習", object_detection, "物体検出", "YOLOv8")
app.add_app("推論", object_prediction, "物体検出", "YOLOv8")
app.add_app("学習", object_detection_yolox, "物体検出", "YOLOX")
app.add_app("推論", object_prediction_yolox, "物体検出", "YOLOX")

app.add_app("学習", patchcore, "異常検知", "Wide-ResNet-50-2")
app.add_app("推論", patchcore_prediction, "異常検知", "Wide-ResNet-50-2")

app.add_app("学習", semantic_segmentation, "領域検知", "HRNet")
app.add_app("推論", segmentation_prediction, "領域検知", "HRNet")

app.add_app("推論", paddleocr_pre, "文字認識")


app.run()

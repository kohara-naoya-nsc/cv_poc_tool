import streamlit as st  


def home_page():

    st.title("画像解析PoCツール")

    # 画像解析テンプレート画像
    st.image('/app/assets/home_page_image.png')

    st.write("以下の項目をクリックすると、別タブでドキュメントが表示されます。")

    # セクションごとのリンク
    pages = {
         "ユーザ登録・プロジェクト作成": "doc_user",
         "前処理ツール(射影変換(画角補正)、動画→画像変換)": "doc_tool",
         "画像分類モデル(アノテーション、オートアノテーション、学習、推論)": "doc_classification",
         "物体検出モデル(アノテーション、オートアノテーション、学習、推論)": "doc_detection",
         "領域検出モデル(推論)": "doc_segmentation",
         "異常検知モデル(アノテーション、学習、推論)": "doc_anomaly",
         "汎用文字認識モデル(推論)": "doc_ocr",
    }

    for label, page_name in pages.items():
        st.markdown(
            f'<a href="/{page_name}" target="_blank" style="color: blue; text-decoration: underline;">{label}</a>',
            unsafe_allow_html=True
        )

    st.divider()
    st.caption(
        '[本ツールのソースコード](https://dev.azure.com/nsdig01/ISTD_ImageProcessingAI_PoC_Tools/_git/azure-cv_poc_tool?path=/cv_poc_tool/backend/yolo)'
    )
    st.caption(
        '本ツールはUltralytics YOLOv8を使用しています。'
        '[Ultralytics公式ソースコード](https://github.com/ultralytics/ultralytics)'
        ' / '
        '[AGPL-3.0ライセンス](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)'
    )



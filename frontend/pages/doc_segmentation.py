import streamlit as st  

st.title("領域検出モデル")
st.markdown('''
    学習を必要としない(ゼロショット)のモデルのため、学習画面が表示されません。
''')
st.header("推論")
st.markdown('''
    サイドバーの「アノテーション（Label Studio）」リンクをクリックしてください。\n
    次に「【領域検出】推論」プロジェクトを選択してください。
''')
st.video("assets/movie/領域検知_プロジェクト選択.mp4")

st.markdown('''
    「Go to import」ボタンをクリックしてください。\n
    領域検出を行いたい画像ファイルをアップロードしてください。
''')
st.video("assets/movie/領域検知_画像インポート.mp4")

st.markdown("### 推論作業")
st.video("assets/movie/領域検知_推論.mp4")

st.markdown("### アノテーションの編集・削除")
st.video("assets/movie/アノテーション_編集_削除.mp4")

st.markdown("### アノテーション削除（一括）")
st.video("assets/movie/アノテーション_データ削除.mp4")

st.header("エクスポート")
st.markdown('''
    エクスポート形式は「Brush labels to PNG」から選択してください。\n
    ファイルダウンロード時にzipファイルが取得できることをご確認ください。
''')
st.video("assets/movie/領域検知_ダウンロード.mp4")



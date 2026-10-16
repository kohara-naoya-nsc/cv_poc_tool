import streamlit as st  

st.title("汎用文字認識モデル")
st.markdown('''
    学習を必要としない(ゼロショット)のモデルのため、学習画面が表示されません。
''')
st.header("推論")
st.markdown("##### 推論開始")
st.write("アップローダーに画像ファイルをアップロードしてください。")
st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
st.image('assets/ocr_predict_fig1.png')

st.markdown("##### オプション設定")
st.image('assets/ocr_predict_fig2.png')
st.write("認識したい言語を一覧から選択してください。")
st.image('assets/ocr_predict_fig2.1.png')
st.write("高度なオプションについては、必要に応じて内容を変更してください。")
st.write("ファイルおよびモデルのアップロードが完了しましたら、「start」ボタンを押してください。")

st.markdown("##### 推論結果")
st.write("ドロップダウンメニューから他の画像を選択することが可能です。")
st.image('assets/ocr_predict_fig3.png')

st.markdown("##### 結果のダウンロード")
st.write("推論が完了すると「文字認識の推論結果をダウンロード」ボタンが表示されます。クリックして推論結果をダウンロードしてください。")
st.image('assets/ocr_predict_fig4.png')

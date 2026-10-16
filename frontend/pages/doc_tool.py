import streamlit as st 

st.title("前処理ツール")

section = st.radio("セクションを選択してください", [
    "射影変換(画角補正)",
    "動画→画像変換"
])

if section == "射影変換(画角補正)":
    st.header("射影変換(画角補正)")
    st.markdown("##### 画像データのアップロード")
    st.write("射影変換する画像データ（png, jpg, jpegファイル）をアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.image('assets/tool_reshape_fig1.png')
    st.markdown("##### 画像の選択")
    st.image('assets/tool_reshape_fig2.png')
    st.write("ドロップダウンメニューから他の画像を選択することが可能です。")
    st.image('assets/tool_reshape_fig3.png')
    st.markdown("##### 射影変換(画角補正)の実行")
    st.write("射影変換(画角補正)したい４点を直接画像上でクリックしてください。")
    st.write("※クリックする順番は自由です。")
    st.image('assets/tool_reshape_fig4.png')
    st.write("※クリックした座標をリセットしたい場合は「座標をリセット」ボタンを押してください。")
    st.write("任意の出力画像の高さと幅を設定したら、「射影変換を開始」ボタンを押してください。")
    st.image('assets/tool_reshape_fig5.png')
    st.write("射影変換後の画像を確認し、問題がなければ「他の画像も同じ座標で射影変換」ボタンを押してください。")
    st.image('assets/tool_reshape_fig6.png')
    st.write("射影変換が完了すると「射影変換した画像をダウンロード」ボタンが表示されます。クリックして結果画像をダウンロードしてください。")


elif section == "動画→画像変換":
    st.header("動画→画像変換")
    st.markdown("##### 動画データのアップロード")
    st.write("画像に変換する動画データ（mp4, aviファイル）をアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.image('assets/tool_capture_fig1.png')
    st.markdown("##### フレーム取得方法の選択")
    st.write("ランダム抽出：指定したフレーム数の画像ファイルをランダムに抽出します。")
    st.write("指定した間隔で抽出：指定したフレーム数ごとに１枚の画像ファイルを抽出します。")
    st.image('assets/tool_capture_fig2.png')
    st.write("「処理開始」ボタンを押すと動画から画像を抽出されます。")
    st.image('assets/tool_capture_fig3.png')
    st.write("処理が完了すると「取得した画像をダウンロード」ボタンが表示されます。クリックして処理結果をダウンロードしてください。")

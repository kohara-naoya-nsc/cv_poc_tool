import streamlit as st  

st.title("異常検知モデル")
st.write('''
画像解析は通常 \n
`アノテーション(学習データ)`→`学習`→`推論` \n
の順番で行います。  
''')
section = st.radio("セクションを選択してください", [
    "アノテーション&エクスポート",
    "学習",
    "推論"
])

if section == "アノテーション&エクスポート":
    st.header("アノテーション")
    st.markdown('''
    ラベル設定画面において、「Image Classification」を選択してください。
    ''')
    st.video("assets/movie/異常検知_ラベリング.mp4")
    st.markdown("※ラベル名は半角英数字（例: class1）で入力してください")
    st.markdown("### アノテーション作業")
    st.video("assets/movie/異常検知_アノテーション.mp4")
    st.markdown('''
    ※ラベルの選択は、マウス操作に加えてキーボードショートカットによる選択も可能です。  \n
    例: test[1] → 「1」キーで選択、test2[2] → 「2」キーで選択
    ''')
    st.markdown("### アノテーションの編集・削除")
    st.video("assets/movie/アノテーション_編集_削除.mp4")
    st.markdown("### アノテーション削除（一括）")
    st.video("assets/movie/アノテーション_データ削除.mp4")
    st.header("エクスポート")
    st.markdown('''
    エクスポート形式は「CSV」と「YOLOv8 OBB」から選択してください。  \n
    ファイルダウンロード時に、csvファイルおよびzipファイルが取得できることをご確認ください。
    ''')
    st.video("assets/movie/異常検知_エクスポート.mp4")

elif section == "学習":
    st.header("新規学習")
    st.markdown("##### 学習データのアップロード")
    st.write("Label Studioで作成した学習データ（zipファイル）をアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.image('assets/anomaly_fig1.png')
    st.markdown("##### オプション設定")
    st.write("※ドロップダウンから正常のラベルを選択してください。")
    st.image('assets/anomaly_fig2.png')
    st.write("高度なオプションについては、必要に応じて内容を変更してください。")
    st.image('assets/anomaly_fig2.1.png')
    st.image('assets/anomaly_fig2.2.png')
    st.markdown("##### 学習開始")
    st.write("「start」ボタンを押すと学習が開始されます。")
    st.image('assets/anomaly_fig3.png')
    st.markdown("##### 結果のダウンロード")
    st.write("学習が完了すると「異常検知の学習結果をダウンロード」ボタンが表示されます。クリックして、学習結果をダウンロードしてください。")
    st.image('assets/anomaly_fig4.png')

    st.header("Label Studio以外のアノテーションデータによる学習")
    st.markdown("##### 学習データのアップロード")
    st.write("正常画像を格納したzipファイルをアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.markdown('''
    ※zipファイル内は以下のような構成である必要があります。  
    ```
    /
    └── class1/ # 正常データが格納されているフォルダ
         ├── img1.jpg
         ├── img2.jpg
         └── ...
    ```
    ''')
    st.image('assets/anomaly_fig1.1.png')
    st.image('assets/anomaly_fig1.2.png')
    st.markdown("##### オプション設定")
    st.image('assets/anomaly_fig2.3.png')
    st.write("高度なオプションについては、必要に応じて内容を変更してください。")
    st.image('assets/anomaly_fig2.4.png')
    st.image('assets/anomaly_fig2.5.png')
    st.markdown("##### 学習開始")
    st.write("「start」ボタンを押すと学習が開始されます。")
    st.image('assets/anomaly_fig3.1.png')
    st.markdown("##### 結果のダウンロード")
    st.write("学習が完了すると「異常検知の学習結果をダウンロード」ボタンが表示されます。クリックして、学習結果をダウンロードしてください。")
    st.image('assets/anomaly_fig4.1.png')

elif section == "推論":
    st.header("推論")
    st.markdown("##### 推論開始")
    st.write("推論対象として「動画」または「画像」のボタンを押してください。")
    st.write("推論用の動画または画像ファイルをアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.write("※「動画」または「画像」のどちらかの推論を行った後にもう一方の推論を行うときはページのリロードを行ってください。")
    st.write("また、学習済みモデルであるfaissファイルをアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.image('assets/anomaly_predict_fig1.png')
    st.markdown("##### オプション設定")
    st.image('assets/anomaly_predict_fig2.png')
    st.write("高度なオプションについては、必要に応じて内容を変更してください。")
    st.image('assets/anomaly_predict_fig2.1.png')
    st.image('assets/anomaly_predict_fig2.2.png')
    st.write("設定が完了しましたら、「start」ボタンを押してください。")
    st.markdown("##### 推論結果")
    st.write("画像ファイルの場合、ドロップダウンメニューから他の画像を選択することが可能です。")
    st.image('assets/anomaly_predict_fig4.png')
    st.write("動画ファイルの場合、推論結果を動画として出力し、プレビュー表示が可能です。")
    st.write("「動画を停止」ボタンをクリックすると、再生中の動画を停止できます。")
    st.image('assets/anomaly_predict_fig4.1.png')
    st.markdown("##### 結果のダウンロード")
    st.write("推論が完了すると「異常検知の推論結果をダウンロード」ボタンが表示されます。クリックして、推論結果をダウンロードしてください。")

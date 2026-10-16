import streamlit as st  

st.title("画像分類モデル")
st.write('''
画像解析は通常 \n
`アノテーション(学習データ)`→`学習`→`推論` \n
の順番で行います。  
''')
section = st.radio("セクションを選択してください", [
    "アノテーション&エクスポート",
    "学習",
    "推論",
    "アノテーション(オートアノテーション)"
])

if section == "アノテーション&エクスポート":
    st.header("アノテーション")
    st.markdown('''
    ラベル設定画面で「Image Classification」を選択してください。
    ''')
    st.video("assets/movie/画像分類_ラベリング.mp4")
    st.markdown("※ラベル名は半角英数字（例: class1）でご入力ください")
    st.markdown("### アノテーション作業")
    st.video("assets/movie/画像分類_アノテーション.mp4")
    st.markdown('''
    ※ラベルの選択は、マウス操作に加えて、キーボードショートカットによる選択も可能です。  \n
    例: test[1] → 「1」キーで選択、test2[2] → 「2」キーで選択
    ''')
    st.markdown("### アノテーションの編集・削除")
    st.video("assets/movie/アノテーション_編集_削除.mp4")
    st.markdown("### アノテーション削除（一括）")
    st.video("assets/movie/アノテーション_データ削除.mp4")
    st.header("エクスポート")
    st.markdown('''
    エクスポート形式は「CSV」と「YOLOv8 OBB」から選択してください。  \n
    ダウンロード時に、csvファイルおよびzipファイルが取得できることをご確認ください。
    ''')
    st.video("assets/movie/画像分類_エクスポート.mp4")

elif section == "アノテーション(オートアノテーション)":
    st.header("オートアノテーション")
    st.write("機械学習モデルと連携することで、画像から自動的にアノテーションを行い、ラベリング作業の効率化を図ります。")
    st.markdown('''
    |モデル名称|モデル説明|
    |--|--|
    |YOLOv8|学習済みモデルを用いて画像分類を行い、自動でラベリングを行います|
    ''')
    st.markdown("### 機械学習モデルとの連携")
    st.markdown("#### モデル設定")
    st.video("assets/movie/ML連携.mp4")
    st.markdown('''
    「Backend URL」には `http://{IPアドレス}:{ポート番号}` を指定してください。  
    以下の表を参考に、モデルごとに割り当てられているポート番号をご確認ください。
    |モデル名称|ポート番号|
    |--|--|
    |YOLOv8|9093|
    ''')
    st.markdown("#### ラベル設定")
    st.video("assets/movie/ML連携_ラベル設定.mp4")
    st.write("<your model path>に`cv_annotation_tool/models`に入れているモデル名を記載する")
    st.markdown('''
    ラベル設定コード
    ```xml
    <View>
      <Image name="image" value="$image"/>
      <Choices name="choice" toName="image" model_score_threshold="0.25" model_path="<your model path>">
        <Choice value="Open"/>
        <Choice value="Close"/>
      </Choices>
    </View>
    ```
    ''')
    st.markdown("##### YOLOv8")
    st.video("assets/movie/yolov8.mp4")

elif section == "学習":
    st.header("新規学習")
    st.markdown("##### 学習データのアップロード")
    st.write("Label Studioで作成した学習データ（zipファイル）をアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.image('assets/classify_fig1.png')
    st.markdown("##### ベースモデルの選択")
    st.write("新規学習の場合、デフォルトのモデルから選択してください。")
    st.image('assets/classify_fig2.png')
    st.markdown("##### オプション設定")
    st.image('assets/classify_fig3.png')
    st.write("高度なオプションについては、必要に応じて内容を変更してください。")
    st.image('assets/classify_fig3.1.png')
    st.image('assets/classify_fig3.2.png')
    st.markdown("##### 学習開始")
    st.write("「start」ボタンを押すと学習が開始されます。")
    st.image('assets/classify_fig4.png')
    st.markdown("##### 結果のダウンロード")
    st.write("学習が完了すると「画像分類の学習結果をダウンロード」ボタンが表示されます。クリックして学習結果をダウンロードしてください。")
    st.image('assets/classify_fig5.png')

    st.header("Label Studio以外のアノテーションデータによる学習")
    st.markdown("##### 学習データのアップロード")
    st.write("クラスごとに分類された画像を含むzipファイルをアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.markdown('''
    ※zipファイル内は以下のような構成である必要があります。  
    ```
    /
    ├── class1/ # クラス名ごとにフォルダを分類
    |   ├── img1.jpg
    |   ├── img2.jpg
    |   └── ...
    └── class2/
         ├── img1.jpg
         ├── img2.jpg
         └── ...
    ```
    ''')
    st.image('assets/classify_fig1.1.png')
    st.image('assets/classify_fig1.2.png')
    st.image('assets/classify_fig1.3.png')
    st.markdown("##### ベースモデルの選択")
    st.write("新規学習の場合、デフォルトのモデルから選択してください。")
    st.image('assets/classify_fig2.png')
    st.markdown("##### オプション設定")
    st.image('assets/classify_fig3.png')
    st.write("高度なオプションについては、必要に応じて内容を変更してください。")
    st.image('assets/classify_fig3.1.png')
    st.image('assets/classify_fig3.2.png')
    st.markdown("##### 学習開始")
    st.write("「start」ボタンを押すと学習が開始されます。")
    st.image('assets/classify_fig4.png')
    st.markdown("##### 結果のダウンロード")
    st.write("学習が完了すると「画像分類の学習結果をダウンロード」ボタンが表示されます。クリックして学習結果をダウンロードしてください。")
    st.image('assets/classify_fig5.png')

    st.header("追加学習")
    st.markdown("##### 学習データのアップロード")
    st.write("Label Studioで作成した学習データ（zipファイル）をアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.image('assets/classify_fig1.png')
    st.markdown("##### 学習済モデルの選択")
    st.image('assets/classify_fig2.1.png')
    st.markdown("##### オプション設定")
    st.image('assets/classify_fig3.png')
    st.write("高度なオプションについては、必要に応じて内容を変更してください。")
    st.image('assets/classify_fig3.1.png')
    st.image('assets/classify_fig3.2.png')
    st.markdown("##### 学習開始")
    st.write("「start」ボタンを押すと学習が開始されます。")
    st.image('assets/classify_fig4.png')
    st.markdown("##### 結果のダウンロード")
    st.write("学習が完了すると「画像分類の学習結果をダウンロード」ボタンが表示されます。クリックして学習結果をダウンロードしてください。")
    st.image('assets/classify_fig5.png')

    st.header("Label Studio以外のアノテーションデータによる学習")
    st.markdown("##### 学習データのアップロード")
    st.write("クラスごとに分類された画像を含むzipファイルをアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.markdown('''
    ※zipファイル内は以下のような構成である必要があります。  
    ```
    /
    ├── class1/
    |   ├── img1.jpg
    |   ├── img2.jpg
    |   └── ...
    └── class2/
         ├── img1.jpg
         ├── img2.jpg
         └── ...
    ```
    ''')
    st.image('assets/classify_fig1.1.png')
    st.image('assets/classify_fig1.2.png')
    st.image('assets/classify_fig1.3.png')
    st.markdown("##### 学習済モデルの選択")
    st.image('assets/classify_fig2.2.png')
    st.markdown("##### オプション設定")
    st.image('assets/classify_fig3.png')
    st.write("高度なオプションについては、必要に応じて内容を変更してください。")
    st.image('assets/classify_fig3.1.png')
    st.image('assets/classify_fig3.2.png')
    st.markdown("##### 学習開始")
    st.write("「start」ボタンを押すと学習が開始されます。")
    st.image('assets/classify_fig4.png')
    st.markdown("##### 結果のダウンロード")
    st.write("学習が完了すると「画像分類の学習結果をダウンロード」ボタンが表示されます。クリックして学習結果をダウンロードしてください。")
    st.image('assets/classify_fig5.png')

elif section == "推論":
    st.header("推論")
    st.markdown("##### 推論開始")
    st.write("推論対象として「動画」または「画像」のボタンを押してください。")
    st.write("推論用の動画または画像ファイルをアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.write("※「動画」または「画像」のどちらかの推論を行った後にもう一方の推論を行うときはページのリロードを行ってください。")
    st.write("また、学習済みモデル（ptファイル）をアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.image('assets/classify_predict_fig1.png')
    st.write("ファイルおよびモデルのアップロードが完了しましたら、「start」ボタンを押してください。")
    st.markdown("##### 推論結果")
    st.write("画像ファイルの場合、ドロップダウンメニューから他の画像を選択することが可能です。")
    st.image('assets/classify_predict_fig2.png')
    st.write("動画ファイルの場合、推論結果を動画として出力し、プレビュー表示が可能です。")
    st.write("「動画を停止」ボタンをクリックすると、再生中の動画を停止できます。")
    st.image('assets/classify_predict_fig3.png')
    st.markdown("##### 結果のダウンロード")
    st.write("推論が完了すると「画像分類の推論結果をダウンロード」ボタンが表示されます。クリックして、推論結果をダウンロードしてください。")

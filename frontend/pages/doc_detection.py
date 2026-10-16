import streamlit as st  

st.title("物体検出モデル")
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
    ラベル設定画面で「Object Detection with Bounding Boxes」を選択してください。
    ''')
    st.video("assets/movie/物体検出_ラベリング.mp4")
    st.markdown("※ラベル名は半角英数字（例: class1）でご入力ください")
    st.markdown("### アノテーション作業")
    st.video("assets/movie/アノテーション.mp4")
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
    エクスポート形式は「YOLO」から選択してください。  \n
    ファイルダウンロード時にzipファイルが取得できることをご確認ください。
    ''')
    st.video("assets/movie/エクスポート.mp4")

elif section == "アノテーション(オートアノテーション)":
    st.header("オートアノテーション")
    st.write("機械学習モデルと連携することで、画像から自動的にアノテーションを行い、ラベリング作業の効率化を図ります。")
    st.markdown('''
    |モデル名称|モデル説明|
    |--|--|
    |Grounding DINO|テキストから画像内のオブジェクトをゼロショット（事前学習なし）で検出し、ラベリングを行います|
    |YOLOv8|学習済みモデルを使用して、画像内のオブジェクトを検出し、ラベリングを行います|
    ''')
    st.markdown("### 機械学習モデルとの連携")
    st.markdown("#### モデル設定")
    st.video("assets/movie/ML連携.mp4")
    st.markdown('''
    「Backend URL」には `http://{IPアドレス}:{ポート番号}` を指定してください。  
    以下の表を参考に、使用するモデルに対応するポート番号をご確認ください。
    |モデル名称|ポート番号|
    |--|--|
    |Grounding DINO|9092|
    |YOLOv8|9093|
    ''')
    st.markdown("#### ラベル設定")
    st.video("assets/movie/ML連携_ラベル設定.mp4")
    st.markdown("##### Grounding DINO")
    st.video("assets/movie/grounding-dino.mp4")
    st.markdown('''
    ラベル設定コード
    ```xml
    <View>
      <Image name="image" value="$image"/>
      <Style>
        .lsf-main-content.lsf-requesting
        .prompt::before { content: ' loading...'; color: #808080; }
      </Style>
      <View className="prompt">
        <TextArea name="prompt" toName="image" editable="true" rows="2" maxSubmissions="1" showSubmitButton="true"/>
      </View>
      <Header>Rectangle Labels</Header>
      <RectangleLabels name="label" toName="image">  
      <Label value="test" background="#FFA39E"/></RectangleLabels>
    </View>
    ```
    ''')
    st.markdown("##### YOLOv8")
    st.video("assets/movie/yolov8.mp4")
    st.markdown('''
    ラベル設定コード
    ```xml
    <View>
      <Image name="image1" value="$image"/>
      <RectangleLabels name="label1" toName="image1" model_path="torpedo_e100_best.pt" model_score_threshold="0.1">
      <Label value="p1" background="#FFA39E"/><Label value="p2" background="#D4380D"/><Label value="p3" background="#FFC069"/></RectangleLabels>
    </View>
    ```
    ''')

elif section == "学習":
    st.header("新規学習")
    st.markdown("##### 学習データのアップロード")
    st.write("Label Studioで出力されたzipファイル、またはnotes.jsonファイルを除き同じディレクトリ構造に整えたzipファイルをアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.image('assets/detect_fig.0.png')
    st.image('assets/detect_fig1.png')
    st.markdown("##### ベースモデルの選択")
    st.write("新規学習の場合は、デフォルトのモデルから選択してください。")
    st.image('assets/detect_fig2.png')
    st.markdown("##### オプション設定")
    st.image('assets/detect_fig3.png')
    st.write("高度なオプションについては、必要に応じて内容を変更してください。")
    st.image('assets/detect_fig3.1.png')
    st.markdown("##### 学習開始")
    st.write("「start」ボタンを押すと学習が開始されます。")
    st.image('assets/detect_fig4.png')
    st.markdown("##### 結果のダウンロード")
    st.write("学習が完了すると「物体検出の学習結果をダウンロード」ボタンが表示されます。クリックして学習結果をダウンロードしてください。")
    st.image('assets/detect_fig5.png')

    st.header("追加学習")
    st.markdown("##### 学習データのアップロード")
    st.write("Label Studioで出力されたzipファイル、またはnotes.jsonファイルを除き同じディレクトリ構造に整えたzipファイルをアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.image('assets/detect_fig1.png')
    st.markdown("##### 学習済モデルの選択")
    st.image('assets/detect_fig2.1.png')
    st.markdown("##### オプション設定")
    st.image('assets/detect_fig3.png')
    st.write("高度なオプションについては、必要に応じて内容を変更してください。")
    st.image('assets/detect_fig3.1.png')
    st.markdown("##### 学習開始")
    st.write("「start」ボタンを押すと学習が開始されます。")
    st.image('assets/detect_fig4.png')
    st.markdown("##### 結果のダウンロード")
    st.write("学習が完了すると「物体検出の学習結果をダウンロード」ボタンが表示されます。クリックして学習結果をダウンロードしてください。")
    st.image('assets/detect_fig5.png')

elif section == "推論":
    st.header("推論")
    st.markdown("##### 推論開始")
    st.write("推論対象として「動画」または「画像」のボタンを押してください。")
    st.write("推論用の動画ファイルまたは画像ファイルをアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.write("※「動画」または「画像」のどちらかの推論を行った後にもう一方の推論を行うときはページのリロードを行ってください。")
    st.write("また、学習済みモデルであるptファイルをアップローダーにアップロードしてください。")
    st.write("※ファイル名に日本語が含まれている場合はアップロードしないでください。")
    st.image('assets/detect_predict_fig1.png')
    st.write("ファイルおよびモデルのアップロードが完了しましたら、「start」ボタンを押してください。")
    st.markdown("##### 推論結果")
    st.write("画像ファイルの場合、ドロップダウンメニューから他の画像を選択することが可能です。")
    st.image('assets/detect_predict_fig2.png')
    st.write("動画ファイルの場合、推論結果を動画として出力し、プレビュー表示が可能です。")
    st.write("「動画を停止」ボタンをクリックすると、再生中の動画を停止できます。")
    st.image('assets/detect_predict_fig3.png')
    st.markdown("##### 結果のダウンロード")
    st.write("推論が完了すると「物体検出の推論結果をダウンロード」ボタンが表示されます。クリックして、推論結果をダウンロードしてください。")

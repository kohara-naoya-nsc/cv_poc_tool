import streamlit as st
import os
import uuid
import shutil
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory
import requests

##################
# 共有ディレクトリ
##################
SHARE_DIR = "shared/train/"


#####################
# パラメータ項目
#####################
TARGET_KEYS = {
    "image_width": 480,
    "image_height": 480,
    "evaluation_score_thresh": 0.95,
    "evaluation_np_thresh": 0.8,
    "learning_rate": 1e-5,
    "is_evaluate": True,
    "early_stopping_patience": 20,
    "fix_layer": 5
}


def on_download_click(share_path):
    """不要になった共有ディレクトリを削除する関数"""
    shutil.rmtree(share_path)
    st.session_state.view=False


def parameter_reset():
    """学習パラメータを初期化する関数"""
    for key, value in TARGET_KEYS.items():
        st.session_state[key] = value


def image_classification_prediction_tf2():
    """画像分類：VGG予測ページ"""

    st.title("【推論】画像分類")

    # データセットのアップロード
    train_object = st.file_uploader(
        "推論用画像をアップロード",
        type=["zip"],
        key="train_object"
        )

    # 学習済みモデルをアップロード
    model_object = st.file_uploader(
        "学習済みモデルをアップロード",
        type=["h5"],
        key="model_object"
        )


    ##################################
    # パラメータ設定
    ##################################
    with st.expander("オプション"):
        st.slider("image_width：画像の幅", min_value=32, max_value=1024, value=480, key="image_width")
        st.slider("image_height：画像の高さ", min_value=32, max_value=1024, value=480, key="image_height")
        st.number_input("evaluation_score_thresh：評価スコア閾値", min_value=0.0, max_value=1.0, value=0.95, key="evaluation_score_thresh")
        st.number_input("evaluation_np_thresh：評価NP閾値", min_value=0.0, max_value=1.0, value=0.8, key="evaluation_np_thresh")
        st.number_input("learning_rate：学習率", min_value=1e-7, max_value=1e-1, value=1e-5, key="learning_rate")
        st.checkbox("is_evaluate：評価を行うか", value=True, key="is_evaluate")

    with st.expander("高度なオプション"):
        st.slider("early_stopping_patience：早期終了の忍耐値", min_value=1, max_value=100, value=20, key="early_stopping_patience")
        st.slider("fix_layer：固定する層の数", min_value=0, max_value=20, value=5, key="fix_layer")


    ##########################################
    # "Start"ボタンと"パラメータ初期化"ボタンの作成
    ##########################################
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start", key="train_start")
    with col2:
        st.button("パラメータ初期化", on_click=parameter_reset, key="reset_button")


    if start_button:
        ################################
        # アップロードファイルの存在チェック
        ################################
        if train_object is None:
            st.error("推論用画像をアップロードしてください。")
            st.stop()
        elif model_object is None:
            st.error("学習済みモデルをアップロードしてください。")
            st.stop()


        ##################################
        # 学習開始処理の呼び出し
        ##################################
        # UUIDの取得して共有フォルダのパスを作成
        uuid_num = str(uuid.uuid4())
        share_path = Path(SHARE_DIR, uuid_num)


        #####################################
        # データセットを共有ディレクトリに保存
        #####################################
        if os.path.exists(share_path):
            st.error("再度実行してください")
            st.stop()
        # 保存先ディレクトリを作成
        input_path = Path(share_path, "input")
        os.makedirs(input_path, exist_ok=True)

        # データセットを一時ディレクトリ内で解答
        with TemporaryDirectory() as tmp_dir:
            # 一時ディレクトリ内にデータセットを保存
            dataset_name = train_object.name
            tmp_path = Path(tmp_dir, dataset_name)
            with open(tmp_path, "wb") as f:
                f.write(train_object.getbuffer())

            # 一時ディレクトリ内のZIPファイルを解凍して入力ディレクトリに展開
            with ZipFile(tmp_path, "r") as zip_ref:
                zip_ref.extractall(input_path)


        ###################################################
        # 学習済みモデルを共有ディレクトリに保存
        ###################################################
        model_path = Path(share_path, "model")
        os.makedirs(model_path, exist_ok=True)
        
        with open(Path(model_path, model_object.name), "wb") as f:
            f.write(model_object.getbuffer())
        
        ###################################################
        # 学習結果格納用ディレクトリの作成
        ###################################################
        output_dir = Path(share_path, "output")
        output_dir.mkdir(parents=True, exist_ok=True)


        ####################################################
        # パラメータ整形をしてFastAPIに post
        ####################################################
        payload = {
            key: st.session_state.get(key)
            for key in TARGET_KEYS
        }
        payload["uuid"] = uuid_num

        response = requests.post(
            "http://vgg:8506/predict",
            json=payload,
        )


        ###################################
        # ステータスコードによる処理の分岐
        ###################################
        if response.status_code == 200:
            # 学習結果の存在確認
            output_pth = Path(share_path, "output")
            if not os.path.exists(output_pth):
                st.error(f"推論結果がありません")
            else:
                # 学習結果をzipファイルに圧縮してダウンロード用に提供
                output_zip = shutil.make_archive(output_pth, format='zip', root_dir=str(output_pth))
                with open(output_zip, "rb") as f:
                    st.download_button(
                        label="推論結果のダウンロード",
                        data=f.read(),
                        file_name=f"{uuid_num}_vgg16_weights_tf_dim_ordering_tf_kernels_notop_推論結果.zip",
                        mime="application/zip",
                        on_click=lambda:on_download_click(share_path),
                    )
        elif response.status_code == 500:
            st.error("予期せぬエラーが発生しました。")
            st.error("表示されているエラーメッセージをコピーのうえ、サポート窓口までご連絡ください。")            
        
        else:
            error_msg = response.json(["message"])
            st.error(f"ステータスコード{response.status_code}：{error_msg}")
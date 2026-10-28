import os
import uuid
import streamlit as st
import shutil
import requests
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory


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
    "epoch": 300,
    "batch_size": 64,
    "learning_rate": 1e-5,
    "monimonitoring_metric": "val_loss",
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


def image_classification_tf2(show_title=True):
    """画像分類：VGG学習ページ"""

    if show_title:
        st.title("【学習】画像分類")

    # データセットのアップロード
    train_object = st.file_uploader(
        "学習用画像をアップロード",
        type=["zip"],
        key="train_object"
        )


    ##################################
    # パラメータ設定
    ##################################
    with st.expander("オプション"):
        st.slider("image_width：画像の幅", min_value=32, max_value=1024, value=480, key="image_width")
        st.slider("image_height：画像の高さ", min_value=32, max_value=1024, value=480, key="image_height")
        st.slider("epoch：学習回数", min_value=1, max_value=1000, value=300, key="epoch")
        st.slider("batch_size：バッチサイズ", min_value=1, max_value=256, value=64, key="batch_size")
        st.number_input("learning_rate：学習率", min_value=1e-7, max_value=1e-1, value=1e-5, key="learning_rate")

    with st.expander("高度なオプション"):
        st.text_input("monimonitoring_metric：監視指標", value="val_loss", key="monimonitoring_metric")
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


    ##################################
    # 学習開始処理
    ##################################
    if start_button:
        # UUIDの取得して共有フォルダのパスを作成
        uuid_num = str(uuid.uuid4())
        share_path = Path(SHARE_DIR, uuid_num)


        #####################################
        # データセットを共有ディレクトリに保存
        #####################################
        if os.path.exists(share_path):
            st.error("再度実行してください")
            st.stop()
        
        if train_object:
            # 保存先ディレクトリを作成
            input_path = Path(share_path, "input")
            os.makedirs(input_path)

            # データセットを一時ディレクトリ内で解凍
            with TemporaryDirectory() as tmp_dir:
                # 一時ディレクトリ内にデータセットを保存
                dataset_name = train_object.name
                tmp_path = Path(tmp_dir, dataset_name)
                with open(tmp_path, "wb") as f:
                    f.write(train_object.getbuffer())

                # 一時ディレクトリ内のZIPファイルを解凍して入力ディレクトリに展開
                with ZipFile(tmp_path, "r") as zip_ref:
                    zip_ref.extractall(input_path)

        else:
            st.error("学習用画像をアップロードしてください")
            st.stop()


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
            "http://vgg:8506/train",
            json=payload,
        )


        ###################################
        # ステータスコードによる処理の分岐
        ###################################
        if response.status_code == 200:
            # 学習結果の存在確認
            output_pth = Path(share_path, "output")
            if not os.path.exists(output_pth):
                st.error(f"学習結果がありません")
            else:
                # 学習結果をzipファイルに圧縮してダウンロード用に提供
                output_zip = shutil.make_archive(output_pth, format='zip', root_dir=str(output_pth))
                with open(output_zip, "rb") as f:
                    st.download_button(
                        label="学習結果をダウンロード",
                        data=f.read(),
                        file_name=f"{uuid_num}_vgg16_weights_tf_dim_ordering_tf_kernels_notop_学習結果.zip",
                        mime="application/zip",
                        on_click=lambda:on_download_click(share_path),
                    )
        elif response.status_code == 500:
            st.error("予期せぬエラーが発生しました。")
            st.error("表示されているエラーメッセージをコピーのうえ、サポート窓口までご連絡ください。")            
        
        else:
            error_msg = response.json(["message"])
            st.error(f"ステータスコード{response.status_code}：{error_msg}")            
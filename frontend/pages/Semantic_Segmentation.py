import streamlit as st
import os
import uuid
import shutil
import requests
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory


##############
# 共有フォルダ
##############
SHARE_DIR = "shared/semantic_segmentation/"


##################
# パラメータ項目
##################
TARGET_KEYS = {
    "image_width": 816,
    "image_height": 612,
    "batch_size": 1,
    "multi_scale": True,
    "scale_factor":16,
    "flip": True,
    "end_epoch": 300
}


def on_download_click(share_path):
    """不要になった共有ディレクトリを削除する関数"""
    shutil.rmtree(share_path)
    st.session_state.view=False


def parameter_reset():
    """学習パラメータを初期化する関数"""
    for key, value in TARGET_KEYS.items():
        st.session_state[key] = value


def semantic_segmentation():
    """領域検知：学習ページ"""
    
    st.title("領域検知：学習ページ")


    # データセットのアップロード
    train_object = st.file_uploader(
        "学習用データセットをアップロードしてください",
        type=["zip"],
        key="train_object"
        )


    ##################################
    # パラメータ設定
    ##################################
    with st.expander("オプション"):
        st.slider("画像の幅", min_value=100, max_value=2000, value=816, key="image_width")
        st.slider("画像の高さ", min_value=100, max_value=2000, value=612, key="image_height")
        st.slider("バッチサイズ", min_value=1, max_value=64, value=1, key="batch_size")
        st.slider("終了エポック", min_value=1, max_value=1000, value=300, key="end_epoch")

    with st.expander("高度なオプション"):
        st.checkbox("マルチスケール", value=True, key="multi_scale")
        st.slider("スケール係数", min_value=1, max_value=32, value=16, key="scale_factor")
        st.checkbox("反転", value=True, key="flip")


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
        if train_object is None:
            st.warning("学習用データセットをアップロードしてください")
            st.stop()

        # UUIDの取得して共有フォルダのパスを作成
        uuid_num = str(uuid.uuid4())
        share_path = Path(SHARE_DIR, uuid_num)

        if share_path.exists():
            st.error("共有フォルダが既に存在します")
            st.stop()


        #####################################
        # データセットを共有ディレクトリに保存
        #####################################
        dataset_path = Path(share_path, "dataset")
        os.makedirs(dataset_path)

        # データセットを一時ディレクトリ内で解凍
        with TemporaryDirectory() as tmp_dir:
            # 一時ディレクトリ内にデータセットを保存
            dataset_name = train_object.name
            tmp_path = Path(tmp_dir, dataset_name)
            with open(tmp_path, "wb") as f:
                f.write(train_object.getbuffer())

            # 一時ディレクトリ内のZIPファイルを解凍して入力ディレクトリに展開
            with ZipFile(tmp_path, "r") as zip_ref:
                zip_ref.extractall(dataset_path)


        #################################
        # 成果物格納ディレクトリを作成
        #################################
        output_path = Path(share_path, "output")
        os.makedirs(output_path)


        ####################################################
        # パラメータ整形をしてFastAPIに post
        ####################################################
        payload = {
            key: st.session_state.get(key)
            for key in TARGET_KEYS
        }
        payload["uuid"] = uuid_num

        response = requests.post(
            "http://semantic_segmentation:8506/train",
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
                        label="学習結果のダウンロード",
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
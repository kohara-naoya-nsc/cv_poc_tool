import os
import streamlit as st
import uuid
import shutil
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory

from fw_preprocessing.preprocessing import data_prep


########################
# 共有フォルダ
########################
SHARE_DIR = "shared/tool_gzo_fw/"


#####################
# パラメータ項目
#####################
TARGET_KEYS = {
    "dataset_format": "Detection: YOLO(directory)",
    "prep_mode": "split_augment",
    "train_data_ratio": 0.7,
    "valid_data_ratio": 0.2,
    "test_data_ratio": 0.1,
    "contrast_augmentation_probability": 0.0,
    "blur_augmentation_probability": 0.0,
    "histogram_augmentation_probability": 0.0,
    "gauss_noise_augmentation_probability": 0.0,
    "salt_augmentation_probability": 0.0,
    "flip_augmentation_probability": 0.0,
    "rotate_augmentation_probability": 0.0,
    "rotate_augmentation_degree": 0.0
}


def on_download_click(share_path):
    """不要になった共有ディレクトリを削除する関数"""
    shutil.rmtree(share_path)
    st.session_state.view=False


def parameter_reset():
    """学習パラメータを初期化する関数"""
    for key, value in TARGET_KEYS.items():
        st.session_state[key] = value


def tool_gzo_fw():
    """前処理：画像認識FW"""

    st.title("【前処理】データセット変換")
    share_path = st.session_state.get("share_path")
    uuid_num = st.session_state.get("share_uuid")

    # データセットの形式選択
    format_options = [
        "Detection: YOLO(directory)",
        "Detection: VOC(directory)",
        "Classification: folder-tree(directory)",
        "SemanticSegmentation: Elseg(directory)",
        "SemanticSegmentation: LabelMe(directory)"
        ]
    dataset_format = st.selectbox(
            "データセットの形式を選択してください",
            format_options,
            key="dataset_format"
        )

    # データセットのアップロード
    dataset_directory = st.file_uploader(
        "対象データをアップロードしてください",
        type=["zip"],
        key="dataset_directory",
        )


    ##################################
    # パラメータ設定
    ##################################
    with st.expander("オプション"):
        prep_optios = [
            "split_augment",
            "augment_only"
        ]
        prep_mode = st.selectbox("前処理モードを選択してください", prep_optios, index=0, key="prep_mode")
        train_data_ratio = st.slider("学習データの割合", min_value=0.0, max_value=1.0, value=0.7, step=0.01, key="train_data_ratio")
        valid_data_ratio = st.slider("検証データの割合", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="valid_data_ratio")
        test_data_ratio = st.slider("テストデータの割合", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="test_data_ratio")


    with st.expander("高度なオプション"):
        contrast_augmentation_probability = st.slider("コントラスト増強の確率", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="contrast_augmentation_probability")
        blur_augmentation_probability = st.slider("ブラー増強の確率", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="blur_augmentation_probability")
        histogram_augmentation_probability = st.slider("ヒストグラム増強の確率", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="histogram_augmentation_probability")
        gauss_noise_augmentation_probability = st.slider("ガウスノイズ増強の確率", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="gauss_noise_augmentation_probability")
        salt_augmentation_probability = st.slider("ソルト増強の確率", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="salt_augmentation_probability")
        flip_augmentation_probability = st.slider("反転増強の確率", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="flip_augmentation_probability")
        rotate_augmentation_probability = st.slider("回転増強の確率", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="rotate_augmentation_probability")
        rotate_augmentation_degree = st.slider("回転増強の角度", min_value=0.0, max_value=359.0, value=0.0, step=1.0, key="rotate_augmentation_degree")


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
        # 必要ファイルの確認
        if not dataset_directory:
            st.error("データセットをアップロードしてください")
            st.stop()
        
        # UUIDの取得して共有フォルダのパスを作成
        uuid_num = str(uuid.uuid4())
        share_path = Path(SHARE_DIR, uuid_num)

        if not share_path.exists():
            share_path.mkdir(parents=True)
            st.session_state["share_path"] = str(share_path)
            st.session_state["share_uuid"] = uuid_num
        else:
            st.error(
                "共有フォルダの作成に失敗しました。 \n"
                "再度実行してください。"
                )
            st.stop()


        ########################################
        # データセットを解凍後、共有ディレクトリに保存
        ########################################
        input_path = Path(share_path, "input")
        os.makedirs(input_path)

        with TemporaryDirectory() as tmp_dir:
            # 一時ディレクトリ内にデータセットを保存
            dataset_name = dataset_directory.name
            tmp_path = Path(tmp_dir, dataset_name)
            with open(tmp_path, "wb") as f:
                f.write(dataset_directory.getbuffer())

            # 一時ディレクトリ内のZIPファイルを解凍して入力ディレクトリに展開
            with ZipFile(tmp_path, "r") as zip_ref:
                zip_ref.extractall(input_path)


        ######################################
        # 成果物格納ディレクトリの作成
        ######################################
        output_path = Path(share_path, "output")
        os.makedirs(output_path)


        #####################
        # 前処理の実行
        #####################
        try:
            data_prep(
                input_path,
                dataset_format,
                prep_mode,
                train_data_ratio,
                valid_data_ratio,
                test_data_ratio,
                contrast_augmentation_probability,
                blur_augmentation_probability,
                histogram_augmentation_probability,
                gauss_noise_augmentation_probability,
                salt_augmentation_probability,
                flip_augmentation_probability,
                rotate_augmentation_probability,
                rotate_augmentation_degree,
                output_path,
            )
        except Exception as error:
            st.exception(error)
            st.stop()


        ###########################
        # 前処理結果のダウンロード
        ###########################
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
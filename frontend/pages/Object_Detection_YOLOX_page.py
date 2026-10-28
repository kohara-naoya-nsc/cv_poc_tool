import streamlit as st
import uuid
from pathlib import Path
import os
import requests
import shutil

from verification.original_exp import validate_exp_source

##################
# 共有ディレクトリ
##################
SHARE_DIR = "shared/train/"


#####################
# パラメータ項目
#####################
TARGET_KEYS = {
    "input_width": 640,
    "input_height": 640,
    "multiscale_range": 5,
    "batch_size": 64,
    "max_epoch": 50,
    "warmup_epoch": 1,
    "warmup_learning_rate": 0.0,
    "min_learning_rate_ratio": 0.05,
    "basic_learning_rate": 0.01,
    "no_aug_epochs": 15,
    "activation_layer": "silu",
    "enable_ema": True,
    "weight_decay": 5e-4,
    "optimizer_momentum": 0.9,
    "eval_interval": 10,
    "save_history": False,
    "enable_mixup": True,
    "mixup_aug_prob": 1.0,
    "mosaic_aug_prob": 1.0,
    "hsv_aug_prob": 1.0,
    "flip_aug_prob": 0.0,
    "max_aug_labels": 120,
    "experiment_name": None,
    "name": "yolox_s",
    "dist_backend": "nccl",
    "dist_url": None,
    "devices": None,
    # "exp_file",
    "resume": False,
    # "ckpt"
    "start_epoch": 0,
    "num_machines": 1,
    "machine_rank": 0,
    "fp16": False,
    "cache": "選択しない",
    "occupy": False,
    "logger": "tensorboard"
}


def on_download_click(share_path):
    """不要になった共有ディレクトリを削除する関数"""
    shutil.rmtree(share_path)
    st.session_state.view=False


def parameter_reset():
    """学習パラメータを初期化する関数"""
    for key, value in TARGET_KEYS.items():
        st.session_state[key] = value


def object_detection_yolox():
    """物体検出（YOLOX.ver）の学習画面を表示する関数"""

    ######################
    # 画面 UI の表示
    ######################
    st.title("【学習】物体検出")

    st.markdown("##### zipファイルをアップロードしてください")

    multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
    *対応する形式：.zip
    '''
    st.markdown(multi_img)


    # データセットのアップロード
    train_object = st.file_uploader(
        "import file(zipファイルをアップロードしてください)",
        type=["zip"], accept_multiple_files=False,
        label_visibility="hidden",
        key="train_object"
        )
    st.write("\n\n\n")

    # 学習方法の選択
    train_options = ["１から学習", "再学習"]
    train_selection = st.radio("学習方法の選択", train_options, index=0, key="train_selection")

    if train_selection == "１から学習":
        # １から学習する際に必要なパラメータ
        resume = False

        # 使用するモデルの選択
        model_name_options = ["yolox_s", "yolox_m", "yolox_l", "yolox_x"]
        model_name = st.selectbox("デフォルトのモデルを選択してください", model_name_options, index=0, key="model_name")

    elif train_selection == "再学習":
        # 再学習時に必要なパラメータ
        resume = True

        # 使用するモデルの選択
        model_name_options = ["yolox_s", "yolox_m", "yolox_l", "yolox_x"]
        model_name = st.selectbox("デフォルトのモデルを選択してください", model_name_options, index=0, key="model_name")

        # 再学習するモデルをアップロード
        st.markdown("##### モデルをアップロードしてください.pth")
        multi_dml = '''*注：日本語を含むファイルはアップロードしないでください。
        *対応する形式：.pth
        '''
        st.markdown(multi_dml)

        # モデルのアップロード
        ckpt_path = st.file_uploader(
            "import model pthファイルをアップロードしてください",
            type=["pth"],
            label_visibility="hidden",
            key="ckpt_path"
            )


    ##################################
    # 独自のExpファイルを使用するかの可否
    ##################################
    exp_tf = st.toggle("独自のExp ファイルを使用するか", value=False, key="exp_tf")
    if exp_tf:
        exp_file = st.file_uploader("既存のExpファイルをアップロードしてください。", type=["py"], key="exp_file")
    st.write("\n\n\n")


    ##################################
    # パラメータ設定
    ##################################
    with st.expander("オプション"):
        # モデルのサイズの選択
        name_options = ["yolox_s", "yolox_m", "yolox_l", "yolox_x"]
        st.selectbox("name:モデルのサイズ", name_options, index=0, key="network_size")

        # 入力画像の幅と高さの設定
        st.slider("input_width:入力画像幅", value=640, key="input_width")
        st.slider("input_height:入力画像高さ", value=640, key="input_height")

        # バッチサイズの設定
        st.slider("batch-size:バッチサイズ", value=64, min_value=1, max_value=128, key="batch_size")

        # 最大トレーニングエポック数の設定
        st.slider("max_epoch:最大トレーニングエポック数", value=300, min_value=1, max_value=1000, key="max_epoch")

        # 使用するGPU数の設定
        st.slider("devices;使用するGPU数", value=0, min_value=0, max_value=5, key="devices")

        # Mixed Precision (FP16) 学習の有効化
        st.toggle("fp16:Mixed Precision (FP16) 学習の有効化", value=False, key="fp16")

        # キャッシュの設定
        cache_options = ["選択しない", "ram", "disk"]
        cache = st.selectbox("cache:高速なトレーニングのために、画像をRAMやディスクにキャッシュする。", cache_options, index=0, key="cache")

        # 実験名の設定
        st.text_input("experiment_name:実験名", value="", key="experiment_name")


    with st.expander("高度なオプション"):
        # Multi Scale Trainingのサイズ変動幅
        st.slider("multiscale_range:Multi Scale Trainingのサイズ変動幅", value=5, key="multiscale_range")

        # ウォームアップの設定
        st.slider("warmup_epoch:ウォームアップを何エポック行うか", value=1, min_value=0, max_value=100, key="warmup_epoch")
        st.slider("warmup_learning_rate:ウォームアップ開始時の学習率", value=0.0, max_value=1.0, step=0.01, key="warmup_learning_rate")

        # Cosine Schedulerの最小学習率の設定
        st.slider("min_learning_rate_ratio:Cosine Schedulerの最小学習率", value=0.05, min_value=0.0, max_value=1.0, key="min_learning_rate_ratio")

        # 基本学習率の設定
        st.slider("basic_learning_rate:学習率", value=0.01, min_value=0.0, max_value=1.0, key="basic_learning_rate")

        # Augmentation無しのエポック数の設定
        st.slider("no_aug_epochs:Augmentation無しのエポック", value=15, min_value=0, max_value=100, key="no_aug_epochs")

        # 活性化関数の設定
        activation_layer_options = ["silu", "relu"]
        st.selectbox("activation_layer:活性化関数", activation_layer_options, index=0, key="activation_layer")

        # EMAの有効化
        st.toggle("enable_ema:EMA利用の有効化", value=True, key="enable_ema")
        
        # Mixupの有効化
        enable_mixup = st.toggle("enable_mixup:Mixupの有効化", value=True, key="enable_mixup")
        if enable_mixup:
            st.slider("mixup_aug_prob:MixUpを適用する確率", value=1.0, min_value=0.0, max_value=1.0, key="mixup_aug_prob")

        # Mosaicの適用確率の設定
        st.slider("mosaic_aug_prob:Mosaicを適用する確率", value=1.0, min_value=0.0, max_value=1.0, key="mosaic_aug_prob")

        # Weight Decayの設定
        st.slider("weight_decay:Weight Decay(パラメータの自由度制限)", value=5e-4, min_value=0.0, max_value=0.01, step=1e-5, key="weight_decay")

        # オプティマイザのmomentumの設定
        st.slider("optimizer_momentum:オプティマイザのmomentum(慣性)", value=0.9, min_value=0.0, max_value=1.0, key="optimizer_momentum")

        # 評価間隔の設定
        st.slider("eval_interval:評価間隔", value=10, min_value=0, max_value=1000, key="eval_interval")

        # 途中Checkpoint保存有無の設定
        st.toggle("save_history:途中Checkpoint保存有無", value=False, key="save_history")

        # HSV変換の適用確率の設定
        st.slider("hsv_aug_prob:HSV変換を適用する確率", value=1.0, min_value=0.0, max_value=1.0, key="hsv_aug_prob")

        # 左右反転の適用確率の設定
        st.slider("flip_aug_prob:左右反転を適用する確率", value=1.0, min_value=0.0, max_value=1.0, key="flip_aug_prob")

        # Mosaic後ラベル上限の設定
        st.slider("max_aug_labels:Mosaic後ラベル上限", value=120, min_value=0, max_value=1000, key="max_aug_labels")

        # 分散トレーニングの設定
        st.text_input("dist_url:分散トレーニングの設定に使用するURL", value="", key="dist_url")
        dist_backend_options = ["nccl", "gloo", "mpi", "ucc"]
        st.selectbox("dist-backend:分散バックエンド", dist_backend_options, index=0, key="dist_backend")

        # 再学習時の開始エポックの設定
        if train_selection == "再学習":
            st.slider("start_epoch:学習再開時の開始エポック", value=0, min_value=0, max_value=300, key="start_epoch")

        # 学習に使用するノード数の設定
        st.slider("num_machines:学習に使用するノード数", value=1, min_value=1, max_value=5, key="num_machines")

        # マルチノード学習時のノードランクの設定
        st.slider("machine_rank:マルチノード学習時のノードランク", value=0, min_value=0, max_value=4, key="machine_rank")

        # 学習開始時に GPU メモリを事前確保するかの設定
        st.toggle("occupy:学習開始時に GPU メモリを事前確保", value=False, key="occupy")

        # ロガーの種類の選択
        logger_options = ["tensorboard", "wandb"]
        st.selectbox("logger:ロガー種類", logger_options, index=0, key="logger")


    ##########################################
    # "Start"ボタンと"パラメータ初期化"ボタンの作成
    ##########################################
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start", key="train_start")
    with col2:
        st.button("パラメータ初期化", on_click=parameter_reset, key="reset_button")


    ###################################
    # FastAPIに対して、パラメータ送信
    ###################################
    if start_button:
        # UUIDの取得して共有フォルダのパスを作成
        uuid_num = str(uuid.uuid4())
        share_path = Path(SHARE_DIR, uuid_num)
        
        #######################################################
        # データセットを共有フォルダに格納
        #######################################################
        if train_object is not None:
            zip_save_dir = Path(share_path, "input")
            zip_save_dir.mkdir(parents=True, exist_ok=True)

            zip_path = zip_save_dir / train_object.name
            
            # 共有ディレクトリに書き込み
            with open(zip_path, "wb") as f:
                f.write(train_object.getbuffer())
        else:
            st.warning("zipファイルをアップロードしてください")


        ##################################################
        # 再学習時の学習済みモデルを共有フォルダに格納
        ##################################################
        if train_selection == "再学習":
            if resume == True and ckpt_path is not None:
                # 格納先フォルダ作成
                pth_save_dir = Path(share_path, "model")
                pth_save_dir.mkdir(parents=True, exist_ok=True)

                pth_path = pth_save_dir / ckpt_path.name
                
                with open(pth_path, "wb") as f:
                    f.write(ckpt_path.getbuffer())
            else:
                st.warning("モデルをアップロードしてください。.pth")
        

        ####################################################
        # 既存のExpファイルを共有フォルダに格納
        ####################################################
        if exp_tf and exp_file is not None:

            try:
                exp_source = exp_file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                st.error("ファイルは UTF-8 で保存してください")

            validation = validate_exp_source(exp_source)

            if not validation.is_valid:
                st.error("Exp ファイルの検証に失敗しました")
                
                for error in validation.errors:
                    st.error(error)
                st.stop()

            for warning in validation.warnings:
                st.info(warning)
            
            # 格納先フォルダ作成
            exp_save_dir = Path(share_path, "exp")
            exp_save_dir.mkdir(parents=True, exist_ok=True)

            new_name = f"{uuid_num}.py"
            exp_save_dir_path = exp_save_dir / new_name

            with open(exp_save_dir_path, "wb") as f:
                f.write(exp_file.getbuffer())


        ###################################################
        # 学習結果格納用ディレクトリの作成
        ###################################################
        output_dir = Path(share_path, "output")
        output_dir.mkdir(parents=True, exist_ok=True)


        ######################################################
        # パラメータ整形
        ####################################################
        payload = {
            key: st.session_state.get(key)
            for key in TARGET_KEYS
        }
        payload["resume"] = resume
        payload["cache"] = None if cache == "選択しない" else cache
        payload["uuid_num"] = str(uuid_num)

        response = requests.post(
            "http://yolox:8506/train",
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
                        file_name=f"{model_name}_学習結果.zip",
                        mime="application/zip",
                        on_click=lambda:on_download_click(share_path),
                    )
        elif response.status_code == 500:
            st.error("予期せぬエラーが発生しました。")
            st.error("表示されているエラーメッセージをコピーのうえ、サポート窓口までご連絡ください。")            
        
        else:
            error_msg = response.json(["message"])
            st.error(f"ステータスコード{response.status_code}：{error_msg}")
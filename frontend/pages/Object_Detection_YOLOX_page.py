""" Streamlit UI: YOLOX 学習パラメータをFastAPI に送信するフロントエンド

- Label Studio からエクスポートした zip をアップロード
- 1から学習 / 再学習 を選択
- train.py のCLI 引数を UI で入力
- 入力されたパラメータをFastAPI に post
"""

import streamlit as st
import uuid
from pathlib import Path
import os
import requests
import shutil

from verification.original_exp import validate_exp_source

# 共有ディレクトリ
SHARE_DIR = "shared/train/"

# パラメータ項目
TARGET_KEYS = [
    "input_width",
    "input_height",
    "batch_size",
    "max_epoch",
    "basic_learning_rate",
    "enable_mixup",
    "mixup_aug_prob",
    "mosaic_aug_prob",
    "hsv_aug_prob",
    "flip_aug_prob",
    "enable_ema",
    "eval_interval",
    "save_history",
    "experiment_name",
    "name",
    "multiscale_range",
    "warmup_epoch",
    "warmup_learning_rate",
    "min_learning_rate_ratio",
    "no_aug_epochs",
    "activation_layer",
    "weight_decay",
    "optimizer_momentum",
    "max_aug_labels",
    "dist_backend",
    "dist_url",
    "devices",
    "num_machines",
    "machine_rank",
    "start_epoch",
    "fp16",
    "occupy",
    "logger",
]

##################################
# 不要になったディレクトリを削除
##################################
def on_download_click(share_path):
    shutil.rmtree(share_path)
    st.session_state.view=False


##########################
# パラメータ初期化関数
##########################
def parameter_reset():
    st.session_state["input_width"] = 640
    st.session_state["input_height"] = 640
    st.session_state["batch_size"] = 64
    st.session_state["max_epoch"] = 50
    st.session_state["basic_learning_rate"] = 0.01
    st.session_state["enable_mixup"] = True
    st.session_state["mixup_aug_prob"] = 1.0
    st.session_state["mosaic_aug_prob"] = 1.0
    st.session_state["hsv_aug_prob"] = 1.0
    st.session_state["flip_aug_prob"] = 0.0
    st.session_state["enable_ema"] = True
    st.session_state["eval_interval"] = 10
    st.session_state["save_history"] = False
    st.session_state["experiment_name"] = None
    st.session_state["name"] = "yolox_s"
    st.session_state["multiscale_range"] = 5
    st.session_state["warmup_epoch"] = 1
    st.session_state["warmup_learning_rate"] = 0.0
    st.session_state["min_learning_rate_ratio"] = 0.05
    st.session_state["no_aug_epochs"] = 15
    st.session_state["activation_layer"] = "silu"
    st.session_state["weight_decay"] = 5e-4
    st.session_state["optimizer_momentum"] = 0.9
    st.session_state["max_aug_labels"] = 120
    st.session_state["dist_backend"] = "nccl"
    st.session_state["dist_url"] = None
    st.session_state["devices"] = 0
    st.session_state["num_machines"] = 1
    st.session_state["machine_rank"] = 0
    st.session_state["start_epoch"] = 0
    st.session_state["fp16"] = False
    st.session_state["cache"] = "None"
    st.session_state["occupy"] = False
    st.session_state["logger"] = "tensorboard"


#####################
# 画面 UI
#####################
st.title("【学習】物体検出（YOLOX.ver）")

st.markdown("##### zipファイルをアップロードしてください")

multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
*対応する形式：.zip
'''
st.markdown(multi_img)

# データセットのアップロード
annotation_zip_file = st.file_uploader(
    "import file(zipファイルをアップロードしてください)",
    type=["zip"], accept_multiple_files=False,
    label_visibility="hidden",
    key="annotation_zip_file"
    )
st.write("\n\n\n")

# パラメータの初期化
ckpt_path = None
mixup_aug_prob = 0.0
start_epoch = None

# 学習方法の選択
train_options = ["再学習", "１から学習"]
train_selection = st.radio("学習方法の選択", train_options, index=0, key="train_selection")

if train_selection == "再学習":
    # 再学習時に必要なパラメータ
    resume = True

    # 使用するモデルの選択
    model_name_options = ["yolox_s", "yolox_m", "yolox_l", "yolox_x"]
    model_name = st.selectbox("デフォルトのモデルを選択してください", model_name_options, index=0, key="model_name")
    network_size = model_name[-1]

    # 再学習するモデルをアップロード
    st.markdown("##### モデルをアップロードしていください.pth")
    multi_dml = '''*注：日本語を含むファイルはアップロードしないでください。
    *対応する形式：.pth
    '''
    st.markdown(multi_dml)

    # モデルのアップロード
    ckpt_path = st.file_uploader("import model pthファイルをアップロードしてください", type=["pth"], label_visibility="hidden", key="ckpt_path")

elif train_selection == "１から学習":
    # １から学習する際に必要なパラメータ
    resume = False

    # 使用するモデルの選択
    model_name_options = ["yolox_s", "yolox_m", "yolox_l", "yolox_x"]
    model_name = st.selectbox("デフォルトのモデルを選択してください", model_name_options, index=0, key="model_name")
    network_size = model_name[-1]


# 独自のExpファイルを使用するかの可否
exp_tf = st.toggle("独自のExp ファイルを使用するか", value=False, key="exp_tf")
if exp_tf:
    exp_file = st.file_uploader("既存のExpファイルをアップロードしてください。", type=["py"], key="exp_file")
st.write("\n\n\n")


# Exp 以外で設定するパラメータ
with st.expander("基本オプション"):
    experiment_name = st.text_input("experiment_name:実験名", value="", key="experiment_name")
    batch_size = st.slider("batch-size:バッチサイズ", value=64, min_value=1, max_value=128, key="batch_size")
    devices = st.slider("devices;使用するGPU数", value=0, min_value=0, max_value=5, key="devices")
    
    dist_backend_options = ["nccl", "gloo", "mpi", "ucc"]
    dist_backend = st.selectbox("dist-backend:分散バックエンド", dist_backend_options, index=0, key="dist_backend")

    dist_url = st.text_input("dist_url:分散トレーニングの設定に使用するURL", value="", key="dist_url")
    num_machines = st.slider("num_machines:学習に使用するノード数", value=1, min_value=1, max_value=5, key="num_machines")

    if train_selection == "再学習":
        start_epoch = st.slider("start_epoch:学習再開時の開始エポック", value=0, min_value=0, max_value=300, key="start_epoch")

    fp16 = st.toggle("fp16:Mixed Precision (FP16) 学習の有効化", value=False, key="fp16") 

    cache_options = ["None", "ram", "disk"]
    cache = st.selectbox("cache:高速なトレーニングのために、画像をRAMやディスクにキャッシュする。", cache_options, index=0, key="cache")

    occupy = st.toggle("occupy:学習開始時に GPU メモリを事前確保", value=False, key="occupy")

    logger_options = ["tensorboard", "wandb"]
    logger = st.selectbox("logger:ロガー種類", logger_options, index=0, key="logger")


if not exp_tf:
    # パラメータの設定
    with st.expander("高度なオプション"):
        input_width = st.slider("input_width:入力画像幅", value=640, key="input_width")
        input_height = st.slider("input_height:入力画像高さ", value=640, key="input_height")
        max_epoch = st.slider("max_epoch:最大トレーニングエポック数", value=300, min_value=1, max_value=1000, key="max_epoch")
        basic_learning_rate = st.slider("basic_learning_rate:学習率", value=0.01, min_value=0.0, max_value=1.0, key="basic_learning_rate")

        enable_mixup = st.toggle("enable_mixup:Mixupの有効化", value=True, key="enable_mixup")
        if enable_mixup:
            mixup_aug_prob = st.slider("mixup_aug_prob:MixUpを適用する確率", value=1.0, min_value=0.0, max_value=1.0, key="mixup_aug_prob")

        mosaic_aug_prob = st.slider("mosaic_aug_prob:Mosaicを適用する確率", value=1.0, min_value=0.0, max_value=1.0, key="mosaic_aug_prob")
        hsv_aug_prob = st.slider("hsv_aug_prob:HSV変換を適用する確率", value=1.0, min_value=0.0, max_value=1.0, key="hsv_aug_prob")
        flip_aug_prob = st.slider("flip_aug_prob:左右反転を適用する確率", value=1.0, min_value=0.0, max_value=1.0, key="flip_aug_prob")
        enable_ema = st.toggle("enable_ema:EMA利用の有効化", value=True, key="enable_ema")
        eval_interval = st.slider("eval_interval:評価間隔", value=10, min_value=0, max_value=1000, key="eval_interval")
        save_history = st.toggle("save_history:途中Checkpoint保存有無", value=False, key="save_history")

        name_options = ["yolox_s", "yolox_m", "yolox_l", "yolox_x"]
        name = st.selectbox("name:YOLOX標準Exp指定", name_options, index=0, key="name")
        
        multiscale_range = st.slider("multiscale_range:Multi Scale Trainingのサイズ変動幅", value=5, key="multiscale_range")
        warmup_epoch = st.slider("warmup_epoch:ウォームアップを何エポック行うか", value=1, min_value=0, max_value=100, key="warmup_epoch")
        warmup_learning_rate = st.slider("warmup_learning_rate:ウォームアップ開始時の学習率", value=0.0, max_value=1.0, step=0.01, key="warmup_learning_rate")

        min_learning_rate_ratio = st.slider("min_learning_rate_ratio:Cosine Schedulerの最小学習率", value=0.05, min_value=0.0, max_value=1.0, key="min_learning_rate_ratio")
        no_aug_epochs = st.slider("no_aug_epochs:Augmentation無しのエポック", value=15, min_value=0, max_value=100, key="no_aug_epochs")
        
        activation_layer_options = ["silu", "relu"]
        activation_layer = st.selectbox("activation_layer:活性化関数", activation_layer_options, index=0, key="activation_layer")

        weight_decay = st.slider("weight_decay:Weight Decay(パラメータの自由度制限)", value=5e-4, min_value=0.0, max_value=0.01, step=1e-5, key="weight_decay")
        optimizer_momentum = st.slider("optimizer_momentum:オプティマイザのmomentum(慣性)", value=0.9, min_value=0.0, max_value=1.0, key="optimizer_momentum")
        max_aug_labels = st.slider("max_aug_labels:Mosaic後ラベル上限", value=120, min_value=0, max_value=1000, key="max_aug_labels")
        machine_rank = st.slider("machine_rank:マルチノード学習時のノードランク", value=0, min_value=0, max_value=4, key="machine_rank")


# 列の作成
col1, col2 = st.columns(2)
with col2:
    st.button("パラメータ初期化", on_click=parameter_reset, key="reset_button")
with col1:
    start_button = st.button("Start", key="train_start")


###################################
# FastAPIに対して、パラメータ送信
###################################
if start_button:
    # UUIDの取得
    uuid_num = str(uuid.uuid4())
    share_path = Path(SHARE_DIR, uuid_num)
    
    #######################################################
    # アップロードされたzipファイルを共有フォルダに格納
    #######################################################
    if annotation_zip_file is not None:
        zip_save_dir = Path(share_path, "input")
        zip_save_dir.mkdir(parents=True, exist_ok=True)

        zip_path = zip_save_dir / annotation_zip_file.name
        
        # 共有ディレクトリに書き込み
        with open(zip_path, "wb") as f:
            f.write(annotation_zip_file.getbuffer())
    else:
        st.warning("zipファイルをアップロードしてください")


    ##################################################
    # 再学習時のモデルを共有フォルダに格納
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
        
        is_valid_exp = validation.is_valid
        
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
    payload["network_size"] = network_size
    payload["resume"] = resume
    payload["ckpt"] = str(pth_path) if train_selection == "再学習" else None
    payload["cache"] = None if cache == "None" else cache
    payload["opts"] = []
    payload["exp_tf"] = exp_tf
    payload["uuid_num"] = str(uuid_num)

    response = requests.post(
        "http://yolox:8506/train",
        json=payload,
    )

    ###################################
    # ステータスコードによる処理の分岐
    ###################################

    # 成功時の処理
    if response.status_code == 200:
        output_pth = Path(share_path, "output")
        if not os.path.exists(output_pth):
            st.error(f"学習結果がありません")
        else:
            output_zip = shutil.make_archive(output_pth, format='zip', root_dir=str(output_pth))
            with open(output_zip, "rb") as f:
                st.download_button(
                    label="学習結果のダウンロード",
                    data=f.read(),
                    file_name=f"{model_name}_best_ckpt.zip",
                    mime="application/zip",
                    on_click=lambda:on_download_click(share_path),
                )


    ############################################
    # 失敗時の処理
    ############################################
    elif response.status_code == 402:
        error_msg = response.json()["detail"]
        st.error(error_msg)
        
    elif response.status_code == 404:
        error_msg = response.json()["detail"]
        st.error(error_msg)
    
    else:
        st.error("予期せぬエラーが発生しました。")
        st.error("表示されているエラーメッセージをコピーのうえ、サポート窓口までご連絡ください。")

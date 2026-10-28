import streamlit as st
import uuid
from pathlib import Path
import os
import requests
import shutil
import zipfile
from tempfile import TemporaryDirectory


# 共有ディレクトリ
SHARE_DIR = "shared/prediction/"


##########################
# 設定できるパラメータと初期値
##########################
TARGET_KEYS = {
    "is_evaluate": True,
    "is_trimmedimage_output": False,
    "conf": 0.3,
    "nms": 0.3,
    "experiment_name": None,
    "name": "yolox_s",
    "evaluation_score_thresh": 0.8,
    "evaluation_iou_thresh": 0.5,
    "camid": 0,
    "device": "cpu",
    "fp16": False,
    "legacy": False,
    "fuse": False,
    "trt": False,
}


def on_download_click(share_path):
    """不要になったディレクトリを削除"""
    shutil.rmtree(share_path)
    st.session_state.view=False


def parameter_reset():
    """パラメータ初期化関数"""
    for key, value in TARGET_KEYS.items():
        st.session_state[key] = value

def object_prediction_yolox():
    """物体検出（YOLOX.ver）の推論画面を表示する関数"""

    #####################
    # 画面 UI
    #####################
    st.title("【推論】物体検出")

    # 推論する対象の選択
    demo_options = ["動画", "画像"]
    demo_type = st.radio("推論するものをアップロードしてください。", demo_options, index=0, key="demo_type")
    
    # 動画の場合
    if demo_type == "動画":
        st.markdown("###### 動画をアップロードしてください")
        multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
        *対応する形式：.avi, .mp4
        '''
        st.markdown(multi_img)

        # 動画ファイルのアップロード
        predict_object = st.file_uploader(
            'import images(動画をアップロードしてください)',
            type=['avi', 'mp4'],
            accept_multiple_files=True,
            label_visibility="hidden",
            key="predict_object"
            )
        st.write("\n\n\n")

    # 画像の場合
    elif demo_type == "画像":
        st.markdown("###### 画像をアップロードしてください")
        multi_img = '''*注：日本語を含むファイルはアップロードしないでください。
        *対応する形式：.jpg, .png, .jpeg
        '''
        st.markdown(multi_img)

        # 画像ファイルのアップロード
        predict_object = st.file_uploader(
            'import images(画像をアップロードしてください)',
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            label_visibility="hidden",
            key="predict_object"
            )
        st.write("\n\n\n\n\n")

    # 学習済みモデルとjsonを含んだzipのアップロード
    st.markdown("###### モデルとjsonを含んだzipをアップロードしてください")
    multi_mdl = '''*注：日本語を含むファイルはアップロードしないでください。
            *対応する形式：.zip
    '''
    st.markdown(multi_mdl)

    model_path = st.file_uploader(
        'import model(モデル名.zipをアップロードしてください)',
        type=['zip'],
        label_visibility="hidden"
        )

    model_option = ["yolox_s", "yolox_m", "yolox_l", "yolox_x"]
    model = st.selectbox("使用しているモデルの選択", model_option, index=0, key="model_option")
    st.write("\n\n\n")


    #####################
    # オプション設定
    #####################
    with st.expander("オプション"):

        device_options = ["cpu", "gpu"]
        st.selectbox("device:推論にGPUとCPUどちらを使用するか。", device_options, index=0, key="device")

        st.slider("conf:信頼度しきい値", value=0.3, min_value=0.0, max_value=1.0, key="conf")
        st.toggle("is_trimmedimage_output:検出したBOXでトリミングした画像を出力するか", value=True, key="is_trimmedimage_output")
        st.text_input("experiment-name:結果を保存するフォルダ名", value=None, key="experiment_name")

    with st.expander("高度なオプション"):
        st.slider("nms:NMSのloUしきい値", value=0.3, min_value=0.0, max_value=1.0, key="nms")
        st.toggle("is_evaluate:検出したBOXでトリミングした画像を出力するか", value=True, key="is_evaluate")
        st.slider("evaluation_score_thresh:精度評価スコアのしきい値", value=0.8, min_value=0.0, max_value=1.0, key="evaluation_score_thresh")
        st.slider("evaluation_iou_thresh:IoUのしきい値", value=0.5, min_value=0.0, max_value=1.0, key="evaluation_iou_thresh")
        st.number_input("camid:WebカメラのID", value=0, key="camid")
        st.toggle("fp16:FP16で推論を行うかどうか", value=False, key="fp16")
        st.toggle("legacy:旧YOLOXで学習したモデルを使用している場合は有効化", value=False, key="legacy")
        st.toggle("fuse:Conv層とBN層を1つのConv層に融合", value=False, key="fuse")
        st.toggle("trt:推論にTensorRTモデルを使用するか。", value=False, key="trt")
    

    ##########################################
    # "Start"ボタンと"パラメータ初期化"ボタンの作成
    ##########################################
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start")
    with col2:
        st.button("パラメータ初期化", on_click=parameter_reset)


    ######################################
    # FastAPIに対して、パラメータ送信
    ######################################
    if start_button:

        # 一意となるキーと共有ディレクトリパス作成
        uuid_num = str(uuid.uuid4())
        share_path = Path(SHARE_DIR, uuid_num)


        #################################################
        # 推論対象を共有フォルダに格納
        #################################################
        # 推論を行う対象を格納する共有フォルダを作成
        demo_save_dir = Path(share_path, "dmeo")    
        if not os.path.exists(demo_save_dir):
            os.makedirs(demo_save_dir)

        # 推論対象のファイルを共有フォルダに格納
        if predict_object is not None:
            for file in predict_object:
                save_path = demo_save_dir / file.name

                with open(save_path, "wb") as object_file:
                    object_file.write(file.getbuffer())
        else:
            st.error("推論対象をアップロードしてください")


        ############################################################
        # 学習モデルを共有フォルダに格納
        ############################################################        
        # 学習済みモデルを共有フォルダに格納
        pth_save_dir = Path(share_path, "model")
        if not os.path.exists(pth_save_dir):
            os.makedirs(pth_save_dir)

        # 一時ディレクトリ内で解凍 -> 共有ディレクトリに格納
        with TemporaryDirectory() as tmp:
            model_path.seek(0)

            with zipfile.ZipFile(model_path) as zip_file:
                zip_file.extractall(tmp)

            for item in Path(tmp).iterdir():
                shutil.move(str(item), pth_save_dir)


        #############################################################
        # 出力結果の共有ディレクトリ作成
        #############################################################
        output_path = Path(share_path, "output")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        #######################################
        # 送信パラメータ整形してpost
        #######################################
        payload = {
            key: st.session_state.get(key, default)
            for key, default in TARGET_KEYS.items()
        }
        payload["name"] = model
        payload["demo_type"] = demo_type
        payload["uuid_num"] = uuid_num

        # FastAPI にpost
        response = requests.post(
            "http://yolox:8506/predict",
            json=payload,
        )

        ###################################
        # ステータスコードによる処理の分岐
        ###################################
        if response.status_code == 200:
            result_file = Path(share_path, "output", "predict_output_archive.zip")
            if not os.path.exists(result_file):
                st.error("推論結果がありません")
            
            with open(result_file, "rb") as result:
                st.download_button(
                    label="推論結果をダウンロード",
                    data=result.read(),
                    file_name=result_file.name,
                    mime="application/zip",
                    on_click=lambda:on_download_click(share_path),
                )

        elif response.status_code == 500:
            st.error("予期せぬエラーが発生しました。")
            st.error("表示されているエラーメッセージをコピーのうえ、サポート窓口までご連絡ください。")
        
        else:
            error_msg = response.json(["message"])
            st.error(f"ステータスコード{response.status_code}：{error_msg}")
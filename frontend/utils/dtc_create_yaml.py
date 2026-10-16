import os
import yaml
from tempfile import NamedTemporaryFile
import streamlit as st  
from . import dtc_model_train

def dtc_create_yaml(img_dir_path, val_dir_path, yaml_dir_path,class_text,epochs,batch,save,cache, device, workers, hsv_h, hsv_s, hsv_v, degrees, flipud, fliplr, result_name, run_id,translate,scale,shear,perspective,mosaic,mixup,copy_paste):
    """
    物体検出のデータセット設定と学習パラメータをYAMLファイルに保存する。

    img_dir_pathの親フォルダにyaml_data.yamlとyaml_para.yamlを作成する。
    同名ファイルが存在する場合は上書きする。学習自体は実行しない。

    引数:
        img_dir_path (str): 学習用フォルダのパス。親フォルダをYAMLの保存先とする。
        val_dir_path (str): 検証用フォルダのパス。現在の処理では未使用。
        yaml_dir_path (str): YAMLの保存先。img_dir_pathの親フォルダで上書きされる。
        class_text (str): 改行区切りのクラス名。空文字列の行は除外する。
        epochs (int): 学習を繰り返す回数。
        batch (int): バッチサイズ。
        save (bool): モデルを保存するかどうか。
        cache (bool): データセットをキャッシュするかどうか。
        device: 学習に使用するデバイスの指定。
        workers (int): データ読み込み用のワーカー数。
        hsv_h (float): 色相の変更範囲。
        hsv_s (float): 彩度の変更範囲。
        hsv_v (float): 明度の変更範囲。
        degrees (float): 回転角度の範囲。
        flipud (float): 上下反転を適用する確率。
        fliplr (float): 左右反転を適用する確率。
        result_name (str): 学習結果の名前。現在の処理では未使用。
        run_id (str): 学習実行の識別子。データセットの基準パスに使用する。
        translate (float): 平行移動の範囲。
        scale (float): 拡大・縮小の範囲。
        shear (float): シアー変換の角度範囲。
        perspective (float): 透視変換の強さ。
        mosaic (float): モザイク処理を適用する確率。
        mixup (float): 画像を混合する処理を適用する確率。
        copy_paste (float): コピーペースト処理を適用する確率。

    戻り値:
        None。
    """

    yaml_dir_path=os.path.dirname(img_dir_path)

    lines = class_text.split('\n')
    filter_lines = [line for line in lines if line]
    line_count = len(filter_lines)
    yaml_data = {
        "path": f"/app/shared/train/{run_id}/pre_data/",
        "train": "train/images",
        "val": "val/images",
        "nc": line_count,
        "name": filter_lines,
    }
    yaml_para ={
        "epochs": epochs,
        "batch": batch,
        "save": save,
        "cache": cache,
        "device": device,
        "workers": workers,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "degrees": degrees,
        "flipud": flipud,
        "fliplr": fliplr,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "perspective": perspective,
        "mosaic": mosaic,
        "mixup": mixup,
        "copy_paste": copy_paste,
    }

    data_yaml_path=os.path.join(yaml_dir_path,"yaml_data.yaml")
    para_yaml_path=os.path.join(yaml_dir_path,"yaml_para.yaml")
    with open(data_yaml_path,"w") as data_yaml:
        yaml.dump(yaml_data,data_yaml)
    with open(para_yaml_path,"w") as para_yaml:
        yaml.dump(yaml_para,para_yaml)
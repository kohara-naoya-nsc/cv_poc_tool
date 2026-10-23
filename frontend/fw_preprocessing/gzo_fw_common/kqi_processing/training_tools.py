#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
**作成日:** 2023/08/02
**作成者:** 佃　智行 \n

学習操作の共通処理

"""
import torch

from pathlib import Path
from logging import getLogger

# ロガーを読み込む
logger = getLogger("log")


def torch2onnx(
    input_model_path: str, output_dir_path: str, dummy_data: any, opset=11
) -> str:
    """Pytorch形式のモデルをONNX形式に変換する。

    Args:
        input_model_path(str): モデル格納先のファイルパス
        output_dir_path(str): ONNX形式モデルの出力先ファイルパス
        dummy_data(any): モデル変換用ダミーデータ
        opset(int): ONNXのopsetバージョン番号
    Returns:
        onnx_model_path(str): ONNX変換済みモデル格納先のファイルパス
    Raise:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生

    """

    logger.info("開始：ONNX変換処理")

    # データ格納領域確認
    model_file = Path(input_model_path)
    if model_file.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(model_file))

    # モデルのロード
    model = torch.load(model_file)
    model = model.cuda()

    # evaluation mode
    model.eval()

    if Path(output_dir_path).exists() is False:
        raise FileNotFoundError("出力フォルダが見つかりません：" + output_dir_path)

    output_file = Path(output_dir_path, "convert_model.onnx")

    # モデル変換処理
    torch.onnx.export(
        model,
        dummy_data,
        str(output_file),
        verbose=True,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"}},
    )

    logger.info("終了：ONNX変換処理")

    return str(output_file)

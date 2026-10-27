#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
**作成日:** 2021/05/19
**作成者:** 成田　伊織 \n

入力データ操作の共通処理

"""

import pathlib
import json
import zipfile as zp
import configparser
from logging import getLogger
from gzo_fw_common.common.common_tools import load_common_config
import time

# 定数
# 標準の入力ディレクトリパス
INPUT_DIR_PATH = "/kqi/input"

# 共通設定を読み込む
common_config = load_common_config()

# ロガーを読み込む
logger = getLogger("log")


def decompress_zip(output_dir_path, input_dir_path=INPUT_DIR_PATH):
    """
    入力ディレクトリ以下の全てのzipファイルを出力ディレクトリ以下に解凍する。

    Args:
        output_dir_path (str): 出力ディレクトリパス
        input_dir_path (str): 入力ディレクトリパス (default = INPUT_DIR_PATH)

    Raises:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        NotADirectoryError: 引数に指定したディレクトリがディレクトリではない場合に発生
        BadZipFile: zipファイルが不正な場合に発生
    """

    logger.info("開始：zip解凍処理")

    # Pathオブジェクトの取得
    input_dir = pathlib.Path(input_dir_path)
    output_dir = pathlib.Path(output_dir_path)

    # 存在チェック
    if input_dir.exists() is False:
        raise FileNotFoundError("ディレクトリが見つかりません：" + str(input_dir))
    if output_dir.exists() is False:
        raise FileNotFoundError("ディレクトリが見つかりません：" + str(output_dir))

    # ディレクトリチェック
    if input_dir.is_dir() is False:
        raise NotADirectoryError("ディレクトリではありません：" + str(input_dir))
    if output_dir.is_dir() is False:
        raise NotADirectoryError("ディレクトリではありません：" + str(output_dir))

    # zipファイルの存在確認
    if input_dir.glob("**/*.zip") is not None:

        # 全てのzipファイルを出力ディレクトリ以下に解凍
        # ZIP64形式にも対応する
        for zip_file in input_dir.glob("**/*.zip"):
            try:
                with zp.ZipFile(zip_file, allowZip64=True) as thaw:
                    thaw.extractall("{}/{}".format(output_dir, zip_file.stem))
            except zp.BadZipFile:
                # zipファイルが不正な場合
                raise zp.BadZipFile("zipファイルが不正です：" + str(zip_file))

    logger.info("終了：zip解凍処理")


def make_image_list(input_dir_path=INPUT_DIR_PATH):
    """
    入力ディレクトリ以下の画像を探索してパスのリストを返す。

    Args:
        input_dir_path (str): 入力ディレクトリパス (default = INPUT_DIR_PATH)

    Returns:
        [str]: image_path_list -- 画像のパスのリスト

    Raises:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        NotADirectoryError: 引数に指定したディレクトリがディレクトリではない場合に発生
    """

    logger.info("開始：画像リスト作成")

    # Pathオブジェクトの取得
    input_dir = pathlib.Path(input_dir_path)

    # 存在チェック
    if input_dir.exists() is False:
        raise FileNotFoundError("ディレクトリが見つかりません：" + str(input_dir))

    # ディレクトリチェック
    if input_dir.is_dir() is False:
        raise NotADirectoryError("ディレクトリではありません：" + str(input_dir))

    # 画像の拡張子を取得
    image_suffix = json.loads(common_config.get("common", "image_suffix"))

    # jpg, JPG, png, PNGファイルのパスをstrで取得し、リスト化
    image_path_list = [
        str(img) for img in input_dir.glob("**/*") if img.suffix in image_suffix
    ]

    logger.info("終了：画像リスト作成")
    return image_path_list


def load_config(file_path):
    """
    外出しパラメータファイル(.ini)から設定値をロードする。

    Args:
        file_path (str): パラメータファイルのパス

    Returns:
        ConfigParser: config -- 設定パラメータ一覧

    Raises:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
    """

    logger.info("開始：パラメータファイル(.ini)ロード")

    # Pathオブジェクトの取得
    config_file = pathlib.Path(file_path)

    # 存在チェック
    if config_file.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(config_file))

    # iniファイルを読み込む
    config = configparser.ConfigParser()
    config.read(config_file, encoding="utf-8")

    logger.info("終了：パラメータファイル(.ini)ロード")
    return config


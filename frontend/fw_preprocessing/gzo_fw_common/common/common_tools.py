#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
**作成日:** 2021/05/24
**作成者:** 成田　伊織 \n

画像認識FW共通モジュールが呼び出す共通モジュール
"""

import pathlib
import configparser

# 定数
# common_config.iniの相対パス
COMMON_CONFIG_PATH = "gzo_fw_common/common/common_config.ini"


def load_common_config():
    """
    共通パラメータファイル(common_config.ini)から設定値をロードする。

    Returns:
        ConfigParser: config -- 設定パラメータ一覧
    """

    # Pathオブジェクトの取得
    config_file = pathlib.Path(COMMON_CONFIG_PATH)

    # 存在チェック
    if config_file.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(config_file))

    # iniファイルを読み込む
    config = configparser.ConfigParser()
    config.read(config_file, encoding="utf-8")
    return config

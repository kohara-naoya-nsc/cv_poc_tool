#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
**作成日:** 2021/05/19
**作成者:** 成田　伊織 \n

前処理操作の共通処理

"""

import pathlib
import random
import json
import ast
import csv
import os
import shutil
import cv2
import pandas as pd
import numpy as np
# import mltable
from PIL import Image
from math import ceil, floor
from xml.etree import ElementTree
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation
from logging import getLogger
from lxml.etree import Element, SubElement, tostring
from fw_preprocessing.gzo_fw_common.common.common_tools import load_common_config
from fw_preprocessing.gzo_fw_common.common.gzo_fw_exception import ImageReadError, RatioValueError

# 定数
# 該当要素が存在しないに代入する文字列
NA = "NA"
# 学習用画像出力フォルダ名
TRAIN_DIR_NAME = "train"
# テスト用画像出力フォルダ名
TEST_DIR_NAME = "test"
# 検証用画像出力フォルダ名
VALIDATION_DIR_NAME = "valid"
# 学習用画像一覧出力txtファイル名
TRAIN_FILE_NAME = "train.txt"
# テスト用画像一覧出力txtファイル名
TEST_FILE_NAME = "test.txt"
# 検証用画像一覧出力txtファイル名
VALIDATION_FILE_NAME = "valid.txt"
# txtファイル標準文字コード
TXT_CODE = "utf-8"
# 標準改行コード
NEWLINE = "\n"
# 振り分け比率の有効数字
RATIO_PREC = 4
# YOLO形式の計算精度
YOLO_PREC = 7
# YOLO形式の座標の桁
YOLO_ROUND = 0.000001
# AzureML形式の計算精度
AZUREML_PREC = 22
# AzureML形式の座標の桁
AZUREML_ROUND = 0.00000000000000001

# 画像増幅関連
# ガンマ補正値
GAMMA = 1.4
# 平滑化用
AVERAGE_SQUARE = (5, 5)
# 画像画素数に対する塩ノイズ作成数の割合のデフォルト値
SALT_RATIO = 0.01
# 画像画素数に対するゴマノイズ作成数の割合のデフォルト値
PEPPER_RATIO = 0.01
# ガウス分布に基ずくノイズのデフォルト平均
GAUSS_NOISE_MEAN = 0
# ガウス分布に基ずくノイズのデフォルト標準偏差
GAUSS_NOISE_SIGMA = 15
# jpeg圧縮品質のデフォルト値
JPEG_QUALITY = 95

# 共通設定を読み込む
common_config = load_common_config()

# ロガーを読み込む
logger = getLogger("log")


def _convert_bbox_yolo_to_pascalvoc2007(class_id, width, height, x, y, w, h):
    """
    yolo形式のクラス番号とbbox座標をPascalVOC形式に変換する。

    Args:
        class_id (int or float): クラス番号
        width (float): 画像幅 (px)
        height (float): 画像高さ (px)
        x (float): bbox中心相対x座標
        y (float): bbox中心相対y座標
        w (float): bbox相対幅
        h (float): bbox相対高さ

    Returns:
        int: class_id -- クラス番号
        float: xmin -- bbox最小絶対x座標
        float: xmax -- bbox最大絶対x座標
        float: ymin -- bbox最小絶対y座標
        float: ymax -- bbox最大絶対y座標
    """
    xmax = float((x * width) + (w * width) / 2.0)
    xmin = float((x * width) - (w * width) / 2.0)
    ymax = float((y * height) + (h * height) / 2.0)
    ymin = float((y * height) - (h * height) / 2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)


def convert_annotation_yolo_to_pascalvoc2007(
    annotation_file_path, image_file_path, class_names, output_file_path
):
    """
    YOLO形式のアノテーションデータをPascalVOC2007形式に変換する。

    Args:
        annotation_file_path (str): 入力アノテーションファイルパス
        image_file_path (str): 入力画像ファイルパス
        class_names ([str]): クラス名のリスト (coco.namesやclasses.txtと同じ順番に要素を格納する)
        output_file_path (str): 出力ファイルパス

    Raise:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        NotADirectoryError: 引数に指定したディレクトリがディレクトリではない場合に発生
        ImageReadError: 画像の読み込みに失敗した場合に発生
    """

    logger.info("開始：アノテーション形式変換(yolo_to_pascalvoc2007)")

    # Pathオブジェクトの取得
    annotation_file = pathlib.Path(annotation_file_path)
    image_file = pathlib.Path(image_file_path)
    output_file = pathlib.Path(output_file_path)

    # 存在チェック
    if annotation_file.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(annotation_file))
    if image_file.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(image_file))

    # 出力ファイルの親ディレクトリの存在チェック
    if output_file.parent.exists() is False:
        raise FileNotFoundError("ディレクトリが見つかりません：" + str(output_file.parent))

    # 出力ファイルの親ディレクトリのディレクトリチェック
    if output_file.parent.is_dir() is False:
        raise NotADirectoryError("ディレクトリではありません：" + str(output_file.parent))

    # アノテーションファイルをnumpy行列として読み込む
    yolo_bboxes = np.loadtxt(annotation_file).reshape(-1, 5)

    # 画像ファイルの親ディレクトリを取得
    img_dir = image_file.parent.name

    # 画像ファイルの読み込み
    img = cv2.imread(str(image_file))

    # 画像の読み込みに失敗した場合
    if img is None:
        raise ImageReadError("画像の読み込みに失敗しました：" + str(image_file))

    # 画像ファイルの高さ、幅、チャンネル数を取得
    height, width, channels = img.shape

    # Element, SubElementクラスを用いて、xmlのデータ構造を成形する。
    node_root = Element("annotation")
    node_folder = SubElement(node_root, "folder")
    node_folder.text = img_dir

    img_name = image_file.name
    node_filename = SubElement(node_root, "filename")
    node_filename.text = img_name
    node_path = SubElement(node_root, "path")
    node_path.text = str(image_file)

    node_source = SubElement(node_root, "source")
    node_database = SubElement(node_source, "database")
    node_database.text = NA

    node_size = SubElement(node_root, "size")
    node_width = SubElement(node_size, "width")
    node_width.text = str(width)

    node_height = SubElement(node_size, "height")
    node_height.text = str(height)

    node_depth = SubElement(node_size, "depth")
    node_depth.text = str(channels)

    node_segmented = SubElement(node_root, "segmented")
    node_segmented.text = "0"

    # bbox関連処理
    for bbox in yolo_bboxes:
        # bboxの変換
        pascalvoc_bbox = _convert_bbox_yolo_to_pascalvoc2007(
            bbox[0], width, height, bbox[1], bbox[2], bbox[3], bbox[4]
        )

        node_object = SubElement(node_root, "object")
        node_name = SubElement(node_object, "name")
        node_name.text = class_names[pascalvoc_bbox[0]]

        node_pose = SubElement(node_object, "pose")
        node_pose.text = "Unspecified"

        node_truncated = SubElement(node_object, "truncated")
        node_truncated.text = "0"
        node_difficult = SubElement(node_object, "difficult")
        node_difficult.text = "0"
        node_bndbox = SubElement(node_object, "bndbox")
        node_xmin = SubElement(node_bndbox, "xmin")
        node_xmin.text = str(
            Decimal("{:f}".format(pascalvoc_bbox[1])).quantize(
                1, rounding=ROUND_HALF_UP
            )
        )
        node_ymin = SubElement(node_bndbox, "ymin")
        node_ymin.text = str(
            Decimal("{:f}".format(pascalvoc_bbox[3])).quantize(
                1, rounding=ROUND_HALF_UP
            )
        )
        node_xmax = SubElement(node_bndbox, "xmax")
        node_xmax.text = str(
            Decimal("{:f}".format(pascalvoc_bbox[2])).quantize(
                1, rounding=ROUND_HALF_UP
            )
        )
        node_ymax = SubElement(node_bndbox, "ymax")
        node_ymax.text = str(
            Decimal("{:f}".format(pascalvoc_bbox[4])).quantize(
                1, rounding=ROUND_HALF_UP
            )
        )

    # xmlを出力する
    xml = tostring(node_root, pretty_print=True)
    f = open(output_file, "wb")
    f.write(xml)
    f.close()

    logger.info("終了：アノテーション形式変換(yolo_to_pascalvoc2007)")


def convert_annotation_pascalvoc2007_to_yolo(
    annotation_file_path: str, class_names: list, output_file_path: str
):
    """
    YOLO形式のアノテーションデータをPascalVOC2007形式に変換する。

    Args:
        annotation_file_path (str): 入力アノテーションファイルパス
        class_names ([str]): クラス名のリスト (coco.namesやclasses.txtと同じ順番に要素を格納する)
        output_file_path (str): 出力ファイルパス

    Raise:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        NotADirectoryError: 引数に指定したディレクトリがディレクトリではない場合に発生
    """

    logger.info("開始：アノテーション形式変換(pascalvoc2007_to_yolo)")

    # Decimal型の有効数字を設定
    getcontext().prec = YOLO_PREC

    # yolo形式の座標の小数点以下桁数を設定
    yolo_round = Decimal(str(YOLO_ROUND))

    # Pathオブジェクトの取得
    annotation_file = pathlib.Path(annotation_file_path)
    output_file = pathlib.Path(output_file_path)

    # 存在チェック
    if annotation_file.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(annotation_file))

    # 出力ファイルの親ディレクトリの存在チェック
    if output_file.parent.exists() is False:
        raise FileNotFoundError("ディレクトリが見つかりません：" + str(output_file.parent))

    # 出力ファイルの親ディレクトリのディレクトリチェック
    if output_file.parent.is_dir() is False:
        raise NotADirectoryError("ディレクトリではありません：" + str(output_file.parent))

    # xml読み込み
    xml_path = pathlib.Path(annotation_file_path)
    tree = ElementTree.parse(str(xml_path))
    root = tree.getroot()

    # xmlから画像の幅と高さを取得
    image_width = int(root.find("size").find("width").text)
    image_height = int(root.find("size").find("height").text)

    # 結果を1行ずつ格納するリストの初期化
    detections = []

    # xmlファイルから検出物体のリストを取得する
    objects = [child for child in root if child.tag == "object"]

    # 検出物体
    for object in objects:
        # 座標、クラス名を取得
        xmin = int(float(object.find("bndbox").find("xmin").text))
        ymin = int(float(object.find("bndbox").find("ymin").text))
        xmax = int(float(object.find("bndbox").find("xmax").text))
        ymax = int(float(object.find("bndbox").find("ymax").text))
        class_name = str(object.find("name").text)

        # 座標を変換してDecimal型で取得
        xcenter = Decimal(str((xmin + xmax) / 2 / image_width)).quantize(
            yolo_round, rounding=ROUND_HALF_UP
        )
        ycenter = Decimal(str((ymin + ymax) / 2 / image_height)).quantize(
            yolo_round, rounding=ROUND_HALF_UP
        )
        width = Decimal(str((xmax - xmin) / image_width)).quantize(
            yolo_round, rounding=ROUND_HALF_UP
        )
        height = Decimal(str((ymax - ymin) / image_height)).quantize(
            yolo_round, rounding=ROUND_HALF_UP
        )

        # クラス番号を取得
        class_no = class_names.index(class_name)

        # 結果のリストに検出物体の各変数を格納
        detections.append([class_no, xcenter, ycenter, width, height])

    np_detections = np.array(detections, dtype=str)
    np.savetxt(str(output_file), np_detections, fmt="%s")

    logger.info("終了：アノテーション形式変換(pascalvoc2007_to_yolo)")


def convert_annotation_yolo_to_azureml(
    annotation_data_list: list, class_names: list, output_file_path: str
):
    """
    yolo形式のアノテーションデータをAzureML形式に変換する。

    Args:
        annotation_data_list (list): アノテーション情報(dict)のリスト
        class_names ([str]): クラス名のリスト (coco.namesやclasses.txtと同じ順番に要素を格納する)
        output_file_path (str): 出力ファイルパス

    Raise:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        NotADirectoryError: 引数に指定したディレクトリがディレクトリではない場合に発生
        ImageReadError: 画像の読み込みに失敗した場合に発生
    """

    logger.info("開始：アノテーション形式変換(yolo_to_azureml)")

    # Decimal型の有効数字を設定
    getcontext().prec = AZUREML_PREC

    # AzureML形式の座標の小数点以下桁数を設定
    azuleml_round = Decimal(str(AZUREML_ROUND))

    # 結果格納リストの初期化
    detections = []

    for annotation_data_dict in annotation_data_list:
        # Pathオブジェクトの取得
        annotation_file = pathlib.Path(annotation_data_dict["annotation_filename"])
        output_file = pathlib.Path(output_file_path)

        # 存在チェック
        if annotation_file.exists() is False:
            raise FileNotFoundError("ファイルが見つかりません：" + str(annotation_file))

        # 出力ファイルの親ディレクトリの存在チェック
        if output_file.parent.exists() is False:
            raise FileNotFoundError("ディレクトリが見つかりません：" + str(output_file.parent))

        # 出力ファイルの親ディレクトリのディレクトリチェック
        if output_file.parent.is_dir() is False:
            raise NotADirectoryError("ディレクトリではありません：" + str(output_file.parent))

        # アノテーションファイルをnumpy行列として読み込む
        yolo_bboxes = np.loadtxt(annotation_file, dtype="unicode").reshape(-1, 5)

        # 検出物体数を取得
        object_num = yolo_bboxes.shape[0]

        # 検出物体の数だけ座標を計算する処理
        object_data = ""
        confidences = []
        for i in range(object_num):
            # bboxのラベル、中心座標、幅を取得
            label = yolo_bboxes[i][0]
            x_center = float(yolo_bboxes[i][1])
            y_center = float(yolo_bboxes[i][2])
            width = float(yolo_bboxes[i][3])
            height = float(yolo_bboxes[i][4])

            # 画像パスの取得
            path = annotation_data_dict["image_filename"]

            # 画像ファイルの読み込み
            img = cv2.imread(annotation_data_dict["image_filename"])

            # 画像の読み込みに失敗した場合
            if img is None:
                raise ImageReadError(
                    "画像の読み込みに失敗しました：" + annotation_data_dict["image_filename"]
                )

            # 画像ファイルの高さ、幅、チャンネル数を取得
            image_height, image_width, channels = img.shape

            # YOLO形式の中心座標からx座標の最大最小、y座標の最大最小を計算
            bbox = _convert_bbox_yolo_to_pascalvoc2007(
                label, image_width, image_height, x_center, y_center, width, height
            )

            # アノテーション部の初期化
            annotation_dict = {}

            # 結果のリストに検出物体の各変数を格納
            annotation_dict["label"] = class_names[bbox[0]]
            annotation_dict[
                "topX"
            ] = f"{Decimal(bbox[1] / image_width).quantize(azuleml_round, rounding=ROUND_HALF_UP):.{AZUREML_PREC}f}"
            annotation_dict[
                "topY"
            ] = f"{Decimal(bbox[3] / image_height).quantize(azuleml_round, rounding=ROUND_HALF_UP):.{AZUREML_PREC}f}"
            annotation_dict[
                "bottomX"
            ] = f"{Decimal(bbox[2] / image_width).quantize(azuleml_round, rounding=ROUND_HALF_UP):.{AZUREML_PREC}f}"
            annotation_dict[
                "bottomY"
            ] = f"{Decimal(bbox[4] / image_height).quantize(azuleml_round, rounding=ROUND_HALF_UP):.{AZUREML_PREC}f}"

            object_data = object_data + str(annotation_dict) + " "
            confidences.append(1)
        detections.append(
            [
                path,
                object_data,
                ".".join(map(str, confidences)),
                image_width,
                image_height,
            ]
        )

    # ヘッダ出力
    detections_header = [
        "image_url",
        "label",
        "label_confidence",
        "image_height",
        "image_width",
    ]
    np_detections_header = np.array(detections_header, dtype=str)
    np.savetxt(output_file, np_detections_header[None], fmt="%s", delimiter=",")

    # アノテーション部出力
    np_detections = np.array(detections, dtype=str)
    with open(output_file, "a") as f_handle:
        np.savetxt(f_handle, np_detections, fmt='%s,"[%s]","[%s]",%s,%s', delimiter=",")

    logger.info("終了：アノテーション形式変換(yolo_to_azureml)")


def convert_annotation_pascalvoc2007_to_azureml(
    annotation_data_list: list, output_file_path: str
):
    """
    PascalVOC形式のアノテーションデータをAzureML形式に変換する。

    Args:
        annotation_data_list (list): アノテーション情報(dict)のリスト
        output_file_path (str): 出力ファイルパス

        アノテーション情報(dict)
          annotation_filename:アノテーションファイル名
          image_filename:画像ファイル名

    Raise:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        NotADirectoryError: 引数に指定したディレクトリがディレクトリではない場合に発生
    """

    logger.info("開始：アノテーション形式変換(pascalvoc2007_to_azureml)")

    # Decimal型の有効数字を設定
    getcontext().prec = AZUREML_PREC

    # AzureML形式の座標の小数点以下桁数を設定
    azuleml_round = Decimal(str(AZUREML_ROUND))

    # 結果格納リストの初期化
    detections = []

    for annotation_data_dict in annotation_data_list:
        # Pathオブジェクトの取得
        annotation_file = pathlib.Path(annotation_data_dict["annotation_filename"])
        output_file = pathlib.Path(output_file_path)

        # 存在チェック
        if annotation_file.exists() is False:
            raise FileNotFoundError("ファイルが見つかりません：" + str(annotation_file))

        # 出力ファイルの親ディレクトリの存在チェック
        if output_file.parent.exists() is False:
            raise FileNotFoundError("ディレクトリが見つかりません：" + str(output_file.parent))

        # 出力ファイルの親ディレクトリのディレクトリチェック
        if output_file.parent.is_dir() is False:
            raise NotADirectoryError("ディレクトリではありません：" + str(output_file.parent))

        # xml読み込み
        tree = ElementTree.parse(str(annotation_file))
        root = tree.getroot()

        # xmlから画像パスを取得
        path = annotation_data_dict["image_filename"]

        # xmlから画像の幅と高さを取得
        image_width = int(root.find("size").find("width").text)
        image_height = int(root.find("size").find("height").text)

        # アノテーション部の初期化
        annotation_dict = {}

        # xmlファイルから検出物体のリストを取得する
        objects = [child for child in root if child.tag == "object"]

        # 検出物体
        for object in objects:
            # 座標、クラス名を取得
            xmin = int(object.find("bndbox").find("xmin").text)
            ymin = int(object.find("bndbox").find("ymin").text)
            xmax = int(object.find("bndbox").find("xmax").text)
            ymax = int(object.find("bndbox").find("ymax").text)
            class_name = str(object.find("name").text)

            # 結果のリストに検出物体の各変数を格納
            annotation_dict["label"] = class_name
            annotation_dict[
                "topX"
            ] = f"{Decimal(xmin / image_width).quantize(azuleml_round, rounding=ROUND_HALF_UP):.{AZUREML_PREC}f}"

            annotation_dict[
                "topY"
            ] = f"{Decimal(ymin / image_height).quantize(azuleml_round, rounding=ROUND_HALF_UP):.{AZUREML_PREC}f}"

            annotation_dict[
                "bottomX"
            ] = f"{Decimal(xmax / image_width).quantize(azuleml_round, rounding=ROUND_HALF_UP):.{AZUREML_PREC}f}"

            annotation_dict[
                "bottomY"
            ] = f"{Decimal(ymax / image_height).quantize(azuleml_round, rounding=ROUND_HALF_UP):.{AZUREML_PREC}f}"

            detections.append(
                [path, annotation_dict.copy(), "[1.]", image_width, image_height]
            )

    # ヘッダ出力
    detections_header = [
        "image_url",
        "label",
        "label_confidence",
        "image_height",
        "image_width",
    ]
    np_detections_header = np.array(detections_header, dtype=str)
    np.savetxt(output_file, np_detections_header[None], fmt="%s", delimiter=",")

    # アノテーション部出力
    np_detections = np.array(detections, dtype=str)
    with open(output_file, "a") as f_handle:
        np.savetxt(f_handle, np_detections, fmt='%s,"[%s]",%s,%s,%s', delimiter=",")

    logger.info("終了：アノテーション形式変換(pascalvoc2007_to_azureml)")


def convert_azureml_folder_for_classification(
    annotation_file_path: str, image_file_path: str, output_file_path: str
):
    """
    AzureML形式のデータセットをアノテーションファイルを基にフォルダ構成を変更する

    Args:
        annotation_file_path (str): アノテーションファイル
        image_file_path(str): 画像ファイル格納パス
        output_file_path (str): 出力ファイルパス

    Raise:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        NotADirectoryError: 引数に指定したディレクトリがディレクトリではない場合に発生
    """
    # Pathオブジェクトの取得
    annotation_file = pathlib.Path(annotation_file_path)
    output_file = pathlib.Path(output_file_path)

    # 存在チェック
    if annotation_file.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(annotation_file))

    # 出力ファイルパスの存在チェック
    if output_file.exists() is False:
        raise FileNotFoundError("ディレクトリが見つかりません：" + str(output_file.parent))

    # 出力ファイルパスのディレクトリチェック
    if output_file.is_dir() is False:
        raise NotADirectoryError("ディレクトリではありません：" + str(output_file.parent))

    # アノテーションファイル読み込み
    df = pd.read_csv(annotation_file)

    col_list = df.columns.to_list()
    test_only_flag = "split" not in col_list

    # 出力先のフォルダ作成
    if not test_only_flag:
        # train, validはsplitタグが存在する場合にのみ作成。
        output_train_dir = pathlib.Path(output_file, "images", TRAIN_DIR_NAME)
        os.makedirs(output_train_dir)  # train

        output_validation_dir = pathlib.Path(output_file, "images", VALIDATION_DIR_NAME)
        os.makedirs(output_validation_dir)

    output_test_dir = pathlib.Path(output_file, "images", TEST_DIR_NAME)
    os.makedirs(output_test_dir)

    for row in df.itertuples():
        if test_only_flag:
            split_dir = "test"
        else:
            split_dir = row.split

        class_dir = pathlib.Path(output_file, "images", split_dir, row.label)

        if class_dir.exists() is False:
            # クラスディレクトリがない場合は作成する
            os.mkdir(class_dir)

        image_file = pathlib.Path(row.image_url).name
        shutil.move(
            str(pathlib.Path(image_file_path, image_file)),
            class_dir,
        )  # Classフォルダ以下に移動


def convert_mltable_urifolder(mltable_dataset_path: str, output_dir: str) -> None:
    """mltable形式をフォルダ形式に変換

    Args:
        mltable_dataset_path (str): mltable形式のパス
        output_dir (str): 出力先のパス
    Notes:
        出力先(output_dir)に指定されたディレクトリ直下に下記ファイルを出力する
        * mltable.csv: mltable形式のpandas.DataFrameをCSV形式にて保存
            * 但し画像のパス(image_url)はローカルの絶対パスに変換
        * image_urlからダウンロードした画像
    """
    dataset = mltable.load(mltable_dataset_path)
    mltable_df = dataset.to_pandas_dataframe()

    for i, image_url in enumerate(mltable_df["image_url"]):
        with image_url.open() as f:
            img_path = pathlib.Path(output_dir) / str(image_url).split("/")[-1]
            img = Image.open(f)
            img.save(img_path)
        mltable_df.iloc[i, mltable_df.columns.get_loc("image_url")] = str(img_path)

    mltable_df.to_csv(
        str(pathlib.Path(output_dir) / "mltable.csv"), header=True, index=False
    )


def augmente_images(mode, image_file_path, bboxes=None, degree=None):
    """
    指定したモードで画像増幅を行う。

    Args:
        mode (str):
            画像増幅のモード。下記から選択 \n
            "contrast" -- コントラスト強調 \n
            "blur" -- 平滑化(画像のぼかし) \n
            "histogram" -- ヒストグラム平坦化 \n
            "gauss_noise" -- ガウス分布に基ずくノイズ \n
            "salt" -- 塩ゴマノイズ \n
            "flip" -- 左右反転 \n
            "rotate" -- 回転(引数degreeの指定が必要・指定角度で画像を左回転する)
        image_file_path (str): 入力画像ファイルパス
        bboxes ([[int]]): 画像の推論結果におけるbboxのリスト(bbox形式[xmin, ymin, xmax, ymax])
        degree(int or float): "rotate"モードでの回転角度

    Returns:
        Mat: result_image -- 増幅結果画像
        [[int]]: result_bboxes -- 増幅結果のbboxリスト (bbox形式[xmin, ymin, xmax, ymax])

    Raise:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        ImageReadError: 画像の読み込みに失敗した場合に発生
        TypeError: 引数degreeに数値型以外が入力された場合に発生
        ValueError:
            モードに"rotate"を指定し、かつ引数degreeが入力されていない場合に発生
            引数modeに一致するモードが存在しない場合に発生

    ToDo:
        回転処理の際にリサイズを行うか確認
    """

    def compress_image(img):
        """
        画像を圧縮する。
        画像オブジェクトのdtypeをfloat64からuint8に変換するために用いる。

        Args:
             img (Mat): 読み込み画像

        Returns:
            Mat : result_img -- 圧縮した画像
        """

        # 画像のチャンネル数を取得
        _, _, ch = img.shape

        # jpegにエンコード
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        _, encoded_img = cv2.imencode(".jpg", img, encode_param)

        # jpegをデコード
        result_img = cv2.imdecode(encoded_img, ch)

        return result_img

    def contrast_augmentation(img, bboxes):
        """
        コントラスト強調処理

        Args:
            img (Mat): 読み込み画像

        Returns:
            Mat : result_img -- 増幅した画像
            [[int]]: result_bboxes -- 増幅結果のbboxリスト (bbox形式[xmin, ymin, xmax, ymax])
        """

        # ガンマ値を使って ルックアップテーブルを作成
        # テーブルの初期化
        lookUpTable = np.zeros((256, 1), dtype="uint8")
        # ガンマ補正 これにより画像の輝度を所望の状態にする
        lookUpTable = np.asarray([(i / 255) ** (1 / GAMMA) for i in range(256)]) * 255

        # コントラスト強調処理
        result_img = cv2.LUT(img, lookUpTable)

        # dtypeをfloat64からuint8に変換
        result_img = compress_image(result_img)

        return result_img, bboxes

    def blur_augmentation(img, bboxes):
        """
        平滑化(画像のぼかし)処理

        Args:
            img (Mat): 読み込み画像

        Returns:
            Mat: result_img -- 増幅した画像
            [[int]]: result_bboxes -- 増幅結果のbboxリスト (bbox形式[xmin, ymin, xmax, ymax])
        """

        # 平滑化(画像のぼかし)処理
        result_img = cv2.blur(img, AVERAGE_SQUARE)

        return result_img, bboxes

    def histogram_augmentation(img, bboxes):
        """
        ヒストグラム平坦化処理

        Args:
            img (Mat): 読み込み画像

        Returns:
            Mat: result_img -- 増幅した画像
            [[int]]: result_bboxes -- 増幅結果のbboxリスト (bbox形式[xmin, ymin, xmax, ymax])
        """

        # ヒストグラム平坦化処理
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # Yチャンネルのヒストグラムを均等化
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        result_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        return result_img, bboxes

    def gauss_noise_augmentation(img, bboxes):
        """
        ガウス分布に基ずくノイズ処理

        Args:
            img (Mat): 読み込み画像

        Returns:
            Mat: result_img -- 増幅した画像
            [[int]]: result_bboxes -- 増幅結果のbboxリスト (bbox形式[xmin, ymin, xmax, ymax])
        """

        # 画像オブジェクトのサイズ取得
        row, col, ch = img.shape

        # 平均=0, 標準偏差=15, (幅，高さ，チャンネル数) で正規乱数を出力
        gauss = np.random.normal(GAUSS_NOISE_MEAN, GAUSS_NOISE_SIGMA, (row, col, ch))

        # 形状変換
        gauss = gauss.reshape(row, col, ch)

        # 画像に合成
        result_img = img + gauss

        # dtypeをfloat64からuint8に変換
        result_img = compress_image(result_img)

        return result_img, bboxes

    def salt_and_pepper_augmentation(img, bboxes):
        """
        塩ゴマノイズ処理

        Args:
            img (Mat): 読み込み画像

        Returns:
            Mat: result_img -- 増幅した画像
            [[int]]: result_bboxes -- 増幅結果のbboxリスト (bbox形式[xmin, ymin, xmax, ymax])
        """

        row, col, _ = img.shape

        # 塩ノイズ・ゴマノイズの作成数を計算
        salt_count = int(floor(row * col * SALT_RATIO))
        pepper_count = int(floor(row * col * PEPPER_RATIO))

        # 画像内に塩（白い斑点）を作成
        # 0から(col-1)までの乱数を作る
        pts_x = np.random.randint(0, col - 1, salt_count)
        pts_y = np.random.randint(0, row - 1, salt_count)
        # y,xの順番になることに注意
        img[(pts_y, pts_x)] = (255, 255, 255)

        # 画像内にゴマ(黒い斑点)を作成
        pts_x = np.random.randint(0, col - 1, pepper_count)
        pts_y = np.random.randint(0, row - 1, pepper_count)
        img[(pts_y, pts_x)] = (0, 0, 0)

        return img, bboxes

    def flip_augmentation(img, bboxes):
        """
        左右反転処理

        Args:
            img (Mat): 読み込み画像

        Returns:
            Mat: result_img -- 増幅した画像
            [[int]]: result_bboxes -- 増幅結果のbboxリスト (bbox形式[xmin, ymin, xmax, ymax])
        """

        # 画像を左右反転
        result_img = cv2.flip(img, 1)

        # 画像サイズ取得
        _, width, _ = img.shape

        # bboxにデフォルト値のNoneが指定されている場合は、result_bboxesはNoneを返す
        if bboxes is None:
            result_bboxes = None

        else:
            # bboxが入力された場合は、bboxを変換する
            # 結果bboxを初期化
            result_bboxes = []

            # bboxesを左右反転
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox
                new_xmin = int(width - xmax)
                new_ymin = ymin
                new_xmax = int(width - xmin)
                new_ymax = ymax

                # 結果のリストに追加
                result_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])

        return result_img, result_bboxes

    def rotate_augmentation(img, bboxes, degree):
        """
        指定角度で画像を左回転する

        Args:
            img (Mat): 読み込み画像
            degree (int or float):

        Returns:
            Mat: result_img -- 増幅した画像
            [[int]]: result_bboxes -- 増幅結果のbboxリスト (bbox形式[xmin, ymin, xmax, ymax])

        Raises:
            TypeError: 引数degreeに数値型以外が入力された場合
        """

        # 画像サイズ取得
        height, width, _ = img.shape

        x_center, y_center = width // 2, height // 2

        # 回転行列を取得
        try:
            rotation_matrix = cv2.getRotationMatrix2D((x_center, y_center), degree, 1.0)
        except TypeError:
            raise TypeError("引数degreeの入力値の型が不正です：" + str(degree))

        # 回転行列の変換先画像サイズを設定
        # 元の画像がはみ出ないように設定する
        cos = rotation_matrix[0, 0]
        sin = rotation_matrix[0, 1]
        new_width = int((height * np.abs(sin)) + (width * np.abs(cos)))
        new_height = int((height * np.abs(cos)) + (width * np.abs(sin)))

        # 回転行列の中心座標を補正
        rotation_matrix[0, 2] += (new_width / 2) - x_center
        rotation_matrix[1, 2] += (new_height / 2) - y_center

        # 回転処理の実行
        result_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))

        # bboxにデフォルト値のNoneが指定されている場合は、result_bboxesはNoneを返す
        if bboxes is None:
            result_bboxes = None

        else:
            # bboxが入力された場合は、bboxを変換する
            # 結果bboxを初期化
            result_bboxes = []

            # bboxesを回転
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox
                left_top = [xmin, ymin]
                left_bottom = [xmin, ymax]
                right_top = [xmax, ymin]
                right_bottom = [xmax, ymax]

                # bboxの左上・左下・右上・右下座標を画像中心を回転中心として回転する。
                # 回転結果座標を格納するリスト
                new_corner_list = []

                for point in [left_top, left_bottom, right_top, right_bottom]:
                    # 回転中心が原点になるよう点を平行移動
                    moved_point = [point[0] - x_center, point[1] - y_center]
                    # 座標を回転変換
                    rotated_point = [
                        (cos * moved_point[0]) + (sin * moved_point[1]),
                        -(sin * moved_point[0]) + (cos * moved_point[1]),
                    ]
                    # 回転中心が画像中心になるよう点を平行移動
                    moved_rotated_point = [
                        rotated_point[0] + x_center * new_width // width,
                        rotated_point[1] + y_center * new_height // height,
                    ]

                    # 回転結果座標をリストに格納
                    new_corner_list.append(moved_rotated_point)

                # 回転結果座標からx最大・最小値、y最大・最小値を取得する
                new_xmin = ceil(min([point[0] for point in new_corner_list]))
                new_ymin = ceil(min([point[1] for point in new_corner_list]))
                new_xmax = floor(max([point[0] for point in new_corner_list]))
                new_ymax = floor(max([point[1] for point in new_corner_list]))

                # bboxの変換結果をリストに追加
                result_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])

        return result_img, result_bboxes

    """
    以下、画像増幅のメイン処理
    """

    # Pathオブジェクトの取得
    image_file = pathlib.Path(image_file_path)

    # 存在チェック
    if image_file.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(image_file))

    # 画像の読み込み
    img = cv2.imread(str(image_file))

    # 画像の読み込みに失敗した場合
    if img is None:
        raise ImageReadError("画像の読み込みに失敗しました：" + str(image_file))

    # モードに対応した処理を実行し、値を返す
    # コントラスト強調
    if mode == "contrast":
        result_img, result_bboxes = contrast_augmentation(img, bboxes)
        logger.info(
            "画像増幅実行完了： mode={}, image_file_path={}".format(
                str(mode), str(image_file_path)
            )
        )
        return result_img, result_bboxes

    # 平滑化(画像のぼかし)
    elif mode == "blur":
        result_img, result_bboxes = blur_augmentation(img, bboxes)
        logger.info(
            "画像増幅実行完了： mode={}, image_file_path={}".format(
                str(mode), str(image_file_path)
            )
        )
        return result_img, result_bboxes

    # ヒストグラム平坦化
    elif mode == "histogram":
        result_img, result_bboxes = histogram_augmentation(img, bboxes)
        logger.info(
            "画像増幅実行完了： mode={}, image_file_path={}".format(
                str(mode), str(image_file_path)
            )
        )
        return result_img, result_bboxes

    # ガウス分布に基ずくノイズ
    elif mode == "gauss_noise":
        result_img, result_bboxes = gauss_noise_augmentation(img, bboxes)
        logger.info(
            "画像増幅実行完了： mode={}, image_file_path={}".format(
                str(mode), str(image_file_path)
            )
        )
        return result_img, result_bboxes

    # 塩ゴマノイズ
    elif mode == "salt":
        result_img, result_bboxes = salt_and_pepper_augmentation(img, bboxes)
        logger.info(
            "画像増幅実行完了： mode={}, image_file_path={}".format(
                str(mode), str(image_file_path)
            )
        )
        return result_img, result_bboxes

    # 左右反転
    elif mode == "flip":
        result_img, result_bboxes = flip_augmentation(img, bboxes)
        logger.info(
            "画像増幅実行完了： mode={}, image_file_path={}".format(
                str(mode), str(image_file_path)
            )
        )
        return result_img, result_bboxes

    # 回転
    elif mode == "rotate":
        if degree is None:
            raise ValueError("回転処理を実施する場合は引数degreeを設定してください。")
        try:
            result_img, result_bboxes = rotate_augmentation(img, bboxes, degree)
        except TypeError:
            raise
        logger.info(
            "画像増幅実行完了： mode={}, image_file_path={}, degree={}".format(
                str(mode), str(image_file_path), str(degree)
            )
        )
        return result_img, result_bboxes

    # 引数modeに一致するモードが存在しない場合
    else:
        raise ValueError("モードに不正な値が設定されています：" + str(mode))


def augmente_images_segmentation(image_file_path, points=None):
    """
    Segmentationの画像増幅（左右反転）を行う。

    Args:
        image_file_path (str): 入力画像ファイルパス
        points ([[int]]): 画像の推論結果におけるアノテーションの点群リスト(形式[[x, y], [x, y],...])

    Returns:
        Mat: result_image -- 増幅結果画像
        [[int]]: result_points -- 増幅結果の点群リスト (形式[[x, y], [x, y],...])

    Raise:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        ImageReadError: 画像の読み込みに失敗した場合に発生

    """

    def flip_augmentation(img, points):
        """
        左右反転処理

        Args:
            img (Mat): 読み込み画像
            points: アノテーションの点群リスト
        Returns:
            Mat: result_img -- 増幅した画像
            [[int]]: result_points -- 増幅結果の点群リスト (形式[[x, y], [x, y],...])
        """

        # 画像を左右反転
        result_img = cv2.flip(img, 1)

        # 画像サイズ取得
        _, width, _ = img.shape

        # 点群にデフォルト値のNoneが指定されている場合は、result_pointsはNoneを返す
        if points is None:
            result_points = None

        else:
            # 点群が入力された場合は、点群を変換する
            # 結果の点群を初期化

            result_points = []

            # 点群を左右反転
            for points_list in points:
                result_point_list = []
                for point in points_list:
                    x, y = point
                    new_x = int(width - x)
                    new_y = y

                    # 結果のリストに追加
                    result_point_list.append([new_x, new_y])

                result_points.append(result_point_list)

        return result_img, result_points

    """
    以下、画像増幅のメイン処理
    """

    # Pathオブジェクトの取得
    image_file = pathlib.Path(image_file_path)

    # 存在チェック
    if image_file.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(image_file))

    # 画像の読み込み
    img = cv2.imread(str(image_file))

    # 画像の読み込みに失敗した場合
    if img is None:
        raise ImageReadError("画像の読み込みに失敗しました：" + str(image_file))

    # 左右反転
    result_img, result_points = flip_augmentation(img, points)
    logger.info("画像増幅実行完了： image_file_path={}".format(str(image_file_path)))
    return result_img, result_points


def distribute_images_for_detection(
    input_dir_path, output_dir_path, train_ratio, test_ratio, validation_ratio
):
    """
    Detection用振り分け処理を行う。\n
    入力ディレクトリ以下の画像ファイルを学習用、テスト用、検証用に振り分け、txtファイルに出力する。

    Args:
        input_dir_path (str): 入力ディレクトリパス
        output_dir_path (str): 出力ディレクトリパス
        train_ratio (float or Decimal): 学習データの割合 (0 - 1)
        test_ratio (float or Decimal): テストデータの割合 (0 - 1)
        validation_ratio (float or Decimal): 検証データの割合 (0 - 1)

    Raises:
            ValueError: 引数train_ratio, test_ratio, validation_ratioに数字以外が指定された場合に発生
            RatioValueError:
                引数train_ratio, test_ratio, validation_ratioの合計値が、有効数字4桁で1.000に一致しない場合に発生
            FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
            NotADirectoryError: 引数に指定したディレクトリがディレクトリではない場合に発生
    """

    logger.info("開始：画像振り分け(detection)")

    # Decimal型の有効数字を設定
    getcontext().prec = RATIO_PREC

    # 10進数で正確な四則演算を行うために比率をDecimal型に変換
    try:
        train_ratio = Decimal(str(train_ratio))
        test_ratio = Decimal(str(test_ratio))
        validation_ratio = Decimal(str(validation_ratio))
    except InvalidOperation:
        raise ValueError(
            "次の引数のいずれかにfloat以外の値が設定されています：train_ratio = {}, test_ratio = {}, validation_ratio = {}".format(
                str(train_ratio), str(test_ratio), str(validation_ratio)
            )
        )

    # 比の合計が1以外の場合
    if not (train_ratio + test_ratio + validation_ratio) == 1:
        raise RatioValueError(
            "train_ratio, test_ratio, validation_ratioの合計値が1になるように設定してください。"
        )

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

    # 画像の拡張子を取得
    image_suffix = json.loads(common_config.get("common", "image_suffix"))

    # jpg, JPG, png, PNGファイルのパスを取得し、リスト化
    image_list = [
        file for file in input_dir.glob("**/*") if file.suffix in image_suffix
    ]

    # 画像ファイル総数を取得
    image_num = len(image_list)

    # テスト枚数・検証枚数を計算する。端数は四捨五入する。
    test_num = int(
        Decimal(str(image_num * test_ratio)).quantize(1, rounding=ROUND_HALF_UP)
    )
    validation_num = int(
        Decimal(str(image_num * validation_ratio)).quantize(1, rounding=ROUND_HALF_UP)
    )

    # テスト枚数と検証枚数の合計値が画像枚数を上回っている場合は検証枚数を再計算
    # 例：画像15枚でtrain_ratio, test_ratio, validation_ratio = 0,  0.5, 0.5
    if test_num + validation_num > image_num:
        validation_num = image_num - test_num

    # 画像リストを学習用画像リスト・テスト用画像リスト・検証用画像リストに分割する
    train_images = image_list
    test_images = random.sample(train_images, test_num)
    train_images = list(set(train_images) - set(test_images))
    validation_images = random.sample(train_images, validation_num)
    train_images = list(set(train_images) - set(validation_images))

    # txtファイル名定義
    train_file_path = output_dir / pathlib.Path(TRAIN_FILE_NAME)
    test_file_path = output_dir / pathlib.Path(TEST_FILE_NAME)
    validation_file_path = output_dir / pathlib.Path(VALIDATION_FILE_NAME)

    # txtファイルに記入するためにファイル名のリストを作成
    train_image_names = [str(path) + NEWLINE for path in train_images]
    test_image_names = [str(path) + NEWLINE for path in test_images]
    validation_image_names = [str(path) + NEWLINE for path in validation_images]

    # 書き込む内容が存在する場合は、txtファイル出力
    if not len(train_image_names) == 0:
        with open(train_file_path, "w", encoding=TXT_CODE, newline=NEWLINE) as f:
            f.writelines(train_image_names)

    if not len(test_image_names) == 0:
        with open(test_file_path, "w", encoding=TXT_CODE, newline=NEWLINE) as f:
            f.writelines(test_image_names)

    if not len(validation_image_names) == 0:
        with open(validation_file_path, "w", encoding=TXT_CODE, newline=NEWLINE) as f:
            f.writelines(validation_image_names)

    logger.info("終了：画像振り分け(detection)")


def resize_image(input_image_path, size):
    """
    画像を指定のサイズに合わせてリサイズする

    Args:
        input_image_path (str): 入力画像ファイルパス
        size (tuple(int, int)): 出力ファイルのサイズ (width (px), height (px))

    Returns:
        Mat: reseized_image -- リサイズされた画像

    Raise:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        ImageReadError: 画像の読み込みに失敗した場合に発生
    """

    logger.info("開始：画像リサイズ")

    # Pathオブジェクトの取得
    input_image = pathlib.Path(input_image_path)

    # 存在チェック
    if input_image.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(input_image))

    # 画像の読込
    img = cv2.imread(str(input_image))

    # 画像の読み込みに失敗した場合
    if img is None:
        raise ImageReadError("画像の読み込みに失敗しました：" + str(img))

    # 画像のリサイズ
    resized_image = cv2.resize(img, dsize=size)

    logger.info("終了：画像リサイズ")

    return resized_image


def get_class_list_from_voc_detection_annotation(annotation_data_list: list) -> list:
    """
    PascalVOC形式のアノテーションデータをAzureML形式に変換する。

    Args:
        annotation_data_list (list): アノテーションファイルパス(str)のリスト

    Returns:
        list: クラス名(Str)のリスト

    Raise:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        NotADirectoryError: 引数に指定したディレクトリがディレクトリではない場合に発生
    """

    # 結果格納リストの初期化
    class_list = []

    for annotation_data in annotation_data_list:
        # Pathオブジェクトの取得
        annotation_file = pathlib.Path(annotation_data)

        # 存在チェック
        if annotation_file.exists() is False:
            raise FileNotFoundError("ファイルが見つかりません：" + str(annotation_file))

        # xml読み込み
        tree = ElementTree.parse(str(annotation_file))
        root = tree.getroot()

        # xmlファイルから検出物体のリストを取得する
        objects = [child for child in root if child.tag == "object"]

        # 検出物体
        for object in objects:
            # 座標、クラス名を取得
            class_name = str(object.find("name").text)
            class_list.append(class_name)

    return set(class_list)


def convert_annotation_azureml_to_yolo(output_dir, input_csv_absolute_path) -> list:
    """
    AzureML形式のアノテーションファイルをYOLO形式へ変換
    出力フォルダにアノテーションファイルとclasses.txtを作成

    Args:
        output_dir（str）: 出力フォルダのディレクトリ絶対パス
        input_csv_absolute_path (str): AzureMLのアノテーションファイル格納ディレクトリ絶対パス

    Returns:
        result[bool]: 正常終了：true/異常終了：false

    Raise:
        FileNotFoundError: 引数に指定したディレクトリが存在しない場合
    """
    # Pathオブジェクトの取得
    target = pathlib.Path(input_csv_absolute_path)

    # AzureMLのアノテーションファイルの存在チェック
    if target.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(input_csv_absolute_path))

    # 出力用のディレクトリを作成
    output_annotations = pathlib.Path(output_dir)
    if not output_annotations.exists():
        os.makedirs(str(output_annotations))

    class_dict = dict()
    with open(input_csv_absolute_path, encoding=TXT_CODE, newline="") as f:
        csvreader = csv.reader(f)
        # ヘッダー情報の取得
        header = next(csvreader)
        idx = 0
        # 一行づつ読込
        for row in csv.DictReader(f, header):
            # ラベル情報を取得
            if "label" not in row.keys():
                raise Exception("CSVのヘッダーに'label'が存在しません。")
            csv_detail = ast.literal_eval(row["label"].replace("}\n", "},"))

            image_url = row["image_url"]
            width = row["image_width"]
            height = row["image_height"]

            # 画像ファイルの親ディレクトリを取得
            img_path = pathlib.Path(target.parent, image_url)

            # 画像ファイルのコピー
            shutil.move(str(img_path), pathlib.Path(output_annotations, img_path.name))

            txt_path = pathlib.Path(
                output_annotations, img_path.with_suffix(".txt").name
            )

            f = open(str(txt_path), "w")
            for i in csv_detail:
                if idx > 0:
                    f.write("\n")

                if i["label"] not in class_dict.values():
                    class_dict[idx] = i["label"]
                    idx = idx + 1

                key = [k for k, v in class_dict.items() if v == i["label"]]
                x_center = (float(i["bottomX"]) - float(i["topX"])) / 2
                y_center = (float(i["bottomY"]) - float(i["topY"])) / 2

                f.write(
                    str(key[0])
                    + " "
                    + str(x_center)
                    + " "
                    + str(y_center)
                    + " "
                    + str(width)
                    + " "
                    + str(height)
                )
            f.close()

        # classes.txtの出力
        if len(class_dict) > 0:
            classes_path = pathlib.Path(output_dir, "classes.txt")
            f = open(str(classes_path), "w")
            for i in range(len(class_dict)):
                if i > 0:
                    f.write(NEWLINE)
                f.write(class_dict.get(i))
            f.close()


def convert_annotation_azureml_to_pascalvoc2007(
    input_csv: str, output_voc_root_dir: str, default_img_set: str = "train"
) -> None:
    """mltable形式からPascalVOC2007形式への変換

    Args:
        input_csv (str): mltable形式のCSVファイル
        output_voc_root_dir (str): 出力先のルートディレクトリパス
        default_img_set (str, optional): splitカラムが無い場合のデフォルト振り分け先(train※default/valid/test)

    Raises:
        FileNotFoundError: mltable形式のファイルが存在しない場合は例外をエスカレーション
    """
    input_csv_path = pathlib.Path(input_csv)
    output_dir_path = pathlib.Path(output_voc_root_dir)

    # 存在チェック
    if not input_csv_path.exists():
        raise FileNotFoundError("ファイルが見つかりません: {}".format(str(input_csv_path)))

    # 出力ディレクトリ作成
    if not output_dir_path.exists():
        os.makedirs(str(output_dir_path), exist_ok=True)
    # VOCフォルダ構造作成
    voc_dir_path = output_dir_path / "VOC2007"
    img_dir_path = voc_dir_path / "JPEGImages"
    annotation_dir_path = voc_dir_path / "Annotations"
    img_set_dir_path = voc_dir_path / "ImageSets" / "Main"
    for dir_path in [voc_dir_path, img_dir_path, annotation_dir_path, img_set_dir_path]:
        os.makedirs(str(dir_path), exist_ok=True)

    # データ変換
    mltable_df = pd.read_csv(str(input_csv_path))
    for row in mltable_df.itertuples():
        image_url = row.image_url
        label = row.label
        try:
            split = row.split
        except AttributeError:
            # splitカラムが無い場合はデフォルト振り分け先とする
            split = default_img_set
        # 画像のコピー
        image_path = img_dir_path / pathlib.Path(image_url).name
        shutil.copy(input_csv_path.parent / image_url, str(image_path))

        # Element, SubElementクラスを用いて、xmlのデータ構造を成形する。
        node_root = Element("annotation")
        node_folder = SubElement(node_root, "folder")
        node_folder.text = str(img_dir_path)

        img_name = pathlib.Path(image_url).name
        node_filename = SubElement(node_root, "filename")
        node_filename.text = img_name
        node_path = SubElement(node_root, "path")
        node_path.text = str(image_path)

        node_source = SubElement(node_root, "source")
        node_database = SubElement(node_source, "database")
        node_database.text = "NA"

        img = cv2.imread(str(image_path))
        height, width, channels = img.shape

        node_size = SubElement(node_root, "size")
        node_width = SubElement(node_size, "width")
        node_width.text = str(width)

        node_height = SubElement(node_size, "height")
        node_height.text = str(height)

        node_depth = SubElement(node_size, "depth")
        node_depth.text = str(channels)

        node_segmented = SubElement(node_root, "segmented")
        node_segmented.text = "0"

        # bboxの変換
        obj_list = ast.literal_eval(label.replace("}\n", "},"))
        for obj in obj_list:
            node_object = SubElement(node_root, "object")
            node_name = SubElement(node_object, "name")
            node_name.text = obj["label"]

            node_pose = SubElement(node_object, "pose")
            node_pose.text = "Unspecified"

            node_truncated = SubElement(node_object, "truncated")
            node_truncated.text = "0"
            node_difficult = SubElement(node_object, "difficult")
            node_difficult.text = "0"
            node_bndbox = SubElement(node_object, "bndbox")

            node_xmin = SubElement(node_bndbox, "xmin")
            xmin = Decimal("{:f}".format(obj["topX"])) * Decimal(width)
            node_xmin.text = str(xmin.quantize(1, rounding=ROUND_HALF_UP))

            ymin = Decimal("{:f}".format(obj["topY"])) * Decimal(height)
            node_ymin = SubElement(node_bndbox, "ymin")
            node_ymin.text = str(ymin.quantize(1, rounding=ROUND_HALF_UP))

            xmax = Decimal("{:f}".format(obj["bottomX"])) * Decimal(width)
            node_xmax = SubElement(node_bndbox, "xmax")
            node_xmax.text = str(xmax.quantize(1, rounding=ROUND_HALF_UP))

            ymax = Decimal("{:f}".format(obj["bottomY"])) * Decimal(height)
            node_ymax = SubElement(node_bndbox, "ymax")
            node_ymax.text = str(ymax.quantize(1, rounding=ROUND_HALF_UP))

        # アノテーションファイル作成
        xml_str = tostring(node_root, pretty_print=True)
        with open(
            str(pathlib.Path(annotation_dir_path, image_path.stem + ".xml")),
            "wb",
        ) as f:
            f.write(xml_str)

        # 振り分けファイル書き込み
        txt_file_path = pathlib.Path(img_set_dir_path, split + ".txt")
        with open(str(txt_file_path), "a", encoding="utf-8") as f:
            f.write(pathlib.Path(image_url).stem + "\n")


def generate_classestxt_from_json(
    dataset_format: str,
    dataset_json_path_list: list,
    classestxt_output_dir_path: str,
) -> str:
    """SemanticSegmentationクラスリスト生成
    アノテーションツール(SemanticSegmentation: EIseg(directory)/
    SemanticSegmentation: LabelMe(directory)/
    SemanticSegmentation: AzureML(mltable))
    から出力されたjsonファイルをもとにclasses.txtを生成する。

    Args:
        dataset_format (str): データセットフォーマット(SemanticSegmentation: EIseg(directory)/SemanticSegmentation: LabelMe(directory)/SemanticSegmentation: AzureML(mltable))
        dataset_json_path_list (list[str]): データセットに含まれるjsonのパスリスト
        classestxt_output_dir_path (str): classes.txtの出力ディレクトリパス

    Returns:
        str: 生成したclasses.txtのパス

    Raise:
        str: 空文字

    """
    classlist = []
    try:
        # 指定されたパスの存在チェック
        if classestxt_output_dir_path.exists() is False:
            raise Exception(
                "データ格納先ディレクトリ({})が見つかりません。".format(str(classestxt_output_dir_path))
            )
        # 指定されたパスのディレクトリ確認
        if os.path.isdir(classestxt_output_dir_path) is False:
            raise Exception("指定されたパスはディレクトリではありません：" + str(classestxt_output_dir_path))

        for json_path in dataset_json_path_list:
            # EISegは1つのjsonファイルにデータセット全体の情報を持つが、
            # 他の形式と合わせてパスはリスト型で受け取る

            # Pathオブジェクトの取得
            target = pathlib.Path(json_path)

            # jsonファイルの存在チェック
            if target.exists() is False:
                raise FileNotFoundError(
                    "ファイルが見つかりません：({})。classes.txt作成を中断します。".format(str(json_path))
                )

            # 入力されたデータ形式によって、クラス読み込み処理を変える
            if dataset_format == "SemanticSegmentation: EIseg(directory)":
                # アノテーションファイルの読み込み
                with open(json_path, encoding="utf-8") as f:
                    annotation_dict = json.load(f)

                # クラスを取得し、クラスリストを生成
                for full_category in annotation_dict["categories"]:
                    if full_category["name"] not in classlist:
                        classlist.append(full_category["name"])

            elif dataset_format == "SemanticSegmentation: LabelMe(directory)":
                # アノテーションファイルの読み込み
                with open(json_path, encoding="utf-8") as f:
                    annotation_dict = json.load(f)

                # クラスを取得し、クラスリストを生成
                for anno in annotation_dict["shapes"]:
                    if anno["label"] not in classlist:
                        classlist.append(anno["label"])

            elif dataset_format == "SemanticSegmentation: AzureML(mltable)":
                # アノテーションファイルの読み込み
                with open(json_path, encoding="utf-8") as f:
                    annotation_dict = json.load(f)

                # クラスを取得し、クラスリストを生成
                for anno in annotation_dict["label"]:
                    if anno["label"] not in classlist:
                        classlist.append(anno["label"])

            else:
                raise Exception("設定パラメータdataset_formatの形式に誤りがあります。")

        # クラスリストをソート
        result_classlist = []
        if len(classlist) > 0:
            for classes in classlist:
                result_classlist.append(str(classes))
            result_classlist.sort()

        # 指定されたディレクトリにclasses.txtを出力
        classes_path = pathlib.Path(classestxt_output_dir_path, "classes.txt")
        if len(result_classlist) > 0:
            with open(classes_path, "w") as new_file:
                for classes in result_classlist:
                    new_file.write(classes + "\n")
        else:
            raise Exception(
                "不正な形式のデータセットが入力されています。データセットのフォルダ構造やファイルの内容等に誤りが無いかをご確認ください。"
            )

    except Exception:
        # 処理失敗 (空の文字列を出力)
        logger.exception("例外によりクラスリスト生成に失敗")
        classes_path = ""
        return classes_path

    logger.info("終了：classes.txt生成(SemanticSegmentation)")
    return classes_path


def generate_graycolor_image_from_json(
    dataset_format: str,
    dataset_json_path_list: list,
    dataset_classlist_path: str,
    graycolor_image_output_dir_path: str,
) -> list:
    """SemanticSegmentationグレースケール画像生成
    アノテーションツール(SemanticSegmentation: EIseg(directory)/SemanticSegmentation: LabelMe(directory)/SemanticSegmentation: AzureML(mltable))
    から出力されたjsonファイルをもとにグレースケール画像を生成する。

    Args:
        dataset_format (str): データセットフォーマット
        (SemanticSegmentation: EIseg(directory)/SemanticSegmentation: LabelMe(directory)/SemanticSegmentation: AzureML(mltable))
        dataset_json_path_list (list[str]): データセットに含まれるjsonのパスリスト
        dataset_classlist_path (str): データセットに含まれるクラスリストのパス
        graycolor_image_output_dir_path (str): グレースケール画像の出力ディレクトリパス

    Returns:
        str: 生成したグレースケール画像のパスリスト
    """
    try:
        # 最終出力の初期化
        result_path_list = []
        # クラスリストの取得
        with open(dataset_classlist_path) as f:
            class_list = f.read().split("\n")
        # グレースケールの0は背景のため先頭にNoneを入れてずらす
        class_list.insert(0, None)

        if dataset_format == "SemanticSegmentation: EIseg(directory)":
            for json_path in dataset_json_path_list:
                # EISegは1つのjsonファイルにデータセット全体の情報を持つが、
                # 他の形式と合わせてパスはリスト型で受け取る

                # アノテーションファイルの読み込み
                with open(json_path, encoding="utf-8") as f:
                    annotation_dict = json.load(f)

                # カテゴリリストを検索しやすい形式へ変換
                full_category_list = annotation_dict["categories"]
                # EISegのIDは1始まりなので先頭の0番目はNoneで初期化
                category_list = [None]
                for full_category in full_category_list:
                    category_list.insert(
                        int(full_category["id"]), full_category["name"]
                    )

                # 黒背景で初期化したイメージリストを作成
                full_image_list = annotation_dict["images"]
                # EISegのIDは1始まりなので先頭の0番目はNoneで初期化
                image_list = [None]
                image_file_name_list = [None]
                for full_image in full_image_list:
                    image_list.insert(
                        int(full_image["id"]),
                        np.zeros(
                            (full_image["height"], full_image["width"], 1), np.uint8
                        ),
                    )
                    image_file_name_list.insert(
                        int(full_image["id"]), full_image["file_name"]
                    )

                # アノテーションデータに従って色を塗っていく
                for anno in annotation_dict["annotations"]:
                    # クラスリストよりインデックスを取得し、グレースケールを決定
                    graycolor = class_list.index(category_list[anno["category_id"]])

                    for pcloud in anno["segmentation"]:
                        # 点群をndarrayのリストに変換
                        points = []
                        for i in range(int(len(pcloud) / 2)):
                            points.append([pcloud[2 * i], pcloud[2 * i + 1]])
                        # グレースケールで塗りつぶす
                        cv2.fillPoly(
                            image_list[anno["image_id"]],
                            np.array([points]).astype(np.int32),
                            [graycolor],
                        )

                # グレースケール画像を保存する
                for idx, img in enumerate(image_list):
                    if image_list[idx] is None:
                        continue
                    img_save_path = pathlib.Path(
                        graycolor_image_output_dir_path,
                        pathlib.Path(image_file_name_list[idx]).stem + "_gray.png",
                    )
                    cv2.imwrite(
                        str(img_save_path),
                        img,
                    )
                    if img_save_path.exists():
                        result_path_list.append(img_save_path)

        elif dataset_format == "SemanticSegmentation: LabelMe(directory)":
            for json_path in dataset_json_path_list:
                # アノテーションファイルの読み込み
                with open(json_path, encoding="utf-8") as f:
                    annotation_dict = json.load(f)
                # 黒背景で初期化したイメージ作成
                img = np.zeros(
                    (annotation_dict["imageHeight"], annotation_dict["imageWidth"], 1),
                    np.uint8,
                )
                # アノテーションデータに従って色を塗っていく
                for anno in annotation_dict["shapes"]:
                    # クラスリストよりインデックスを取得し、グレースケールを決定
                    graycolor = class_list.index(anno["label"])
                    cv2.fillPoly(
                        img, np.array([anno["points"]]).astype(np.int32), [graycolor]
                    )
                # グレースケール画像を保存
                img_save_path = pathlib.Path(
                    graycolor_image_output_dir_path,
                    pathlib.Path(annotation_dict["imagePath"]).stem + "_gray.png",
                )
                cv2.imwrite(str(img_save_path), img)
                if img_save_path.exists():
                    result_path_list.append(img_save_path)

        elif dataset_format == "SemanticSegmentation: AzureML(mltable)":
            for json_path in dataset_json_path_list:
                # アノテーションファイルの読み込み
                with open(json_path, encoding="utf-8") as f:
                    annotation_dict = json.load(f)
                # 黒背景で初期化したイメージ作成
                img = np.zeros(
                    (
                        annotation_dict["image_height"],
                        annotation_dict["image_width"],
                        1,
                    ),
                    np.uint8,
                )
                # アノテーションデータに従って色を塗っていく
                for anno in annotation_dict["label"]:
                    # クラスリストよりインデックスを取得し、グレースケールを決定
                    graycolor = class_list.index(anno["label"])
                    # 正規化された点群の座標をピクセル値に変換
                    pcloud = anno["polygon"][0]
                    points = []
                    for i in range(int(len(pcloud) / 2)):
                        # 恐らくmltableのPolygon座標を正規化するロジックが間違っているので下記で正しい座標が得られる
                        points.append(
                            [
                                pcloud[2 * i] * annotation_dict["image_width"],
                                pcloud[2 * i + 1] * annotation_dict["image_height"],
                            ]
                        )
                    cv2.fillPoly(img, np.array([points]).astype(np.int64), [graycolor])
                # グレースケール画像を保存
                img_save_path = pathlib.Path(
                    graycolor_image_output_dir_path,
                    pathlib.Path(
                        annotation_dict["image_url"].split("/")[-1],
                    ).stem
                    + "_gray.png",
                )
                cv2.imwrite(str(img_save_path), img)
                if img_save_path.exists():
                    result_path_list.append(img_save_path)
        else:
            raise Exception(
                "不正なデータセットフォーマットです。SemanticSegmentation: EIseg(directory)/SemanticSegmentation: LabelMe(directory)/SemanticSegmentation: AzureML(mltable)から選択が必要です。"
            )
    except Exception:
        # 処理失敗
        logger.exception("例外によりグレースケール画像生成に失敗")
        return result_path_list

    # 処理成功
    return result_path_list

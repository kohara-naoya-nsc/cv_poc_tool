#!/usr/bin/python
# -*- coding: utf-8 -*-
"""AzureML前処理コンポーネント
**画像認識FW
**AzureMLのコンポーネントとして汎用的な前処理(画像増幅・振り分け)機能を提供する
"""

import argparse
import pathlib
import logging
import json
import os
import shutil
import cv2
import subprocess
import sys
import time
import random
import traceback
import pandas as pd
# import mlflow
import ast
import itertools
import csv

# MLTable処理を無効化中
# import mltable
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple
from xml.etree import ElementTree
from lxml.etree import Element, SubElement, tostring
from decimal import Decimal, ROUND_HALF_UP
from configparser import ConfigParser
from configparser import SectionProxy
# from azureml.core import Workspace, Dataset
# import mltable
import cv2
from PIL import Image
import numpy as np

from fw_preprocessing.gzo_fw_common import logging_tools
from fw_preprocessing.gzo_fw_common.kqi_processing import input_handler, preprocessing_tools
from fw_preprocessing.gzo_fw_common.common.gzo_fw_exception import ImageReadError
from fw_preprocessing.gzo_fw_common.logging_tools import FWTimer

# 定数定義
KQI_INPUT_DIR_PATH = "/kqi/input"
KQI_OUTPUT_DIR_PATH = "/kqi/output"
IMAGE_SUFFIX = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
AUGMENTED_IMAGES_DIRECTORY_NAME = "augmented_images"


def _yolo_dataset_augmentation(
    augmentation: dict,
    augmentation_ratio: dict,
    image_path_list: list,
    augmentation_output_dir: str,
    classes_list: list,
) -> tuple:
    """
    水増し画像の生成

    Args:
        augmentation (dict): augmentationの設定
            {“contrast”: True, “blur”: True, “histogram”: True, “gauss_noise”: True, “salt”: True, “flip”: True, “rotate”:3}
        augmentation_ratio (dict): augmentationの割合
            {“contrast”:1, “blur”:1, “histogram”:1, “gauss_noise”:1, “salt”:1, “flip”:1, “rotate”:1}
        image_path_list (list): 画像ファイルパス(pathlib.Path)リスト
        augmentation_output_dir (str): 水増し画像の結果格納領域
        class_list (list): クラス名(str)リスト
    Return:
        tuple:
            list: 増幅画像ファイルパス(pathlib.Path)リスト
            bool: True->VOC, False->YOLO
    """
    # augmentationの出力場所を作成
    augmentation_output_dir_path = Path(augmentation_output_dir)
    if not augmentation_output_dir_path.exists():
        os.makedirs(augmentation_output_dir, exist_ok=True)
    augmented_image_path_list = []

    for yolo_image_file in image_path_list:
        # ファイルが存在しなければ処理しない
        is_voc = False
        if yolo_image_file.exists():
            if yolo_image_file.with_suffix(".txt").exists():
                pass
            elif yolo_image_file.with_suffix(".xml").exists():
                is_voc = True
                preprocessing_tools.convert_annotation_pascalvoc2007_to_yolo(
                    yolo_image_file.with_suffix(".xml"),
                    classes_list,
                    yolo_image_file.with_suffix(".txt"),
                )
            else:
                continue
        else:
            continue

        augmented_image = cv2.imread(str(yolo_image_file))
        image_height, image_width, _ = augmented_image.shape[:3]

        # アノテーションファイル(YOLO形式)をオープン
        yolo_annotation_file = yolo_image_file.with_suffix(".txt")
        with open(yolo_annotation_file, encoding="utf-8") as f2:

            yolo_lines = f2.readlines()
            bboxes = []
            label_nums = []
            for yolo_ln in yolo_lines:
                yolo_line = yolo_ln.rstrip("\n")
                yolo_datas = yolo_line.split()

                label_nums.append(int(yolo_datas[0]))

                # bboxに変換
                x_center = float(yolo_datas[1]) * image_width
                y_center = float(yolo_datas[2]) * image_height
                width = float(yolo_datas[3]) * image_width
                height = float(yolo_datas[4]) * image_height
                bboxes.append(
                    [
                        int(x_center - width / 2),
                        int(y_center - height / 2),
                        int(x_center + width / 2),
                        int(y_center + height / 2),
                    ]
                )

            # augmentationの種類毎に処理をする
            for aug in augmentation:
                if not augmentation[aug]:
                    continue

                # augmentationがかかる確率
                if augmentation_ratio is not None:
                    if random.uniform(0, 1) > float(augmentation_ratio[aug]):
                        continue

                degree = None

                if aug == "rotate":
                    # rotateはBBOXは大きくなるので注意
                    degree = augmentation[aug]
                else:
                    if augmentation[aug] is False:
                        continue

                # augmentationの実行
                result_image, result_bboxes = preprocessing_tools.augmente_images(
                    aug, str(yolo_image_file), bboxes, degree
                )
                image_height, image_width, _ = result_image.shape[:3]

                # augmentation用のファイルに出力する
                yolo_annotation_file_path = augmentation_output_dir_path / (
                    yolo_annotation_file.stem + "_" + aug + ".txt"
                )
                with open(
                    str(yolo_annotation_file_path),
                    "w",
                    encoding="utf-8",
                ) as f3:
                    ind = 0
                    for lbl in label_nums:

                        width = (
                            result_bboxes[ind][2] - result_bboxes[ind][0]
                        ) / image_width
                        height = (
                            result_bboxes[ind][3] - result_bboxes[ind][1]
                        ) / image_height
                        x_center = result_bboxes[ind][0] / image_width + width / 2
                        y_center = result_bboxes[ind][1] / image_height + height / 2

                        f3.write(
                            "{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f}\n".format(
                                label_nums[ind], x_center, y_center, width, height
                            )
                        )
                        ind = ind + 1

                augmented_file_name = augmentation_output_dir_path / Path(
                    str(yolo_image_file.stem) + "_" + aug + yolo_image_file.suffix
                )

                # 画像出力
                cv2.imwrite(str(augmented_file_name), result_image)

                # imageリストに追加
                augmented_image_path_list.append(augmented_file_name)

                # VOCの場合
                if is_voc:
                    preprocessing_tools.convert_annotation_yolo_to_pascalvoc2007(
                        str(yolo_annotation_file_path),
                        str(augmented_file_name),
                        classes_list,
                        str(yolo_annotation_file_path.with_suffix(".xml")),
                    )
                    os.remove(str(yolo_annotation_file_path))

    return (augmented_image_path_list, is_voc)


def _SemanticSegmentation_EIseg_dataset_augmentation(
    flip_augmentation_probability: float,
    image_path_list: list,
    json_path: str,
    augmentation_output_dir: str,
) -> list:
    """
    水増し画像の生成（SemanticSegmentation_EIseg）

    Args:
        flip_augmentation_probability (float):画像反転の割合
        image_path_list (list): 画像ファイルのパスリスト
        json_path(str):jsonファイルのパス
        augmentation_output_dir (str): 水増し画像の結果格納領域

    return:
        augmented_image_path_list(list):増幅画像のパスリスト

    """
    augmented_image_path_list = []
    # 画像反転の割合が0より大きい場合に増幅処理を行う
    if flip_augmentation_probability > 0:

        # augmentationの出力場所を作成
        augmentation_output_dir_path = Path(augmentation_output_dir)
        if not augmentation_output_dir_path.exists():
            os.makedirs(augmentation_output_dir, exist_ok=True)

        for image_path in image_path_list:
            # 画像ファイルが存在しなければ処理しない
            if Path(image_path).exists():
                if Path(image_path).suffix in IMAGE_SUFFIX:
                    pass
                else:
                    continue
            else:
                continue

            # augmentationがかかる確率
            if flip_augmentation_probability is not None:
                if random.uniform(0, 1) > float(flip_augmentation_probability):
                    continue

            # 増幅対象の画像ファイル名取得
            imagefile_name = str(image_path).split("/")[-1]

            # アノテーションファイルの読み込み
            with open(json_path, encoding="utf-8") as f:
                annotation_dict = json.load(f)
            max_id = max(dct["id"] for dct in annotation_dict["annotations"])
            max_image_id = max(dct["id"] for dct in annotation_dict["images"])
            # EIseg_Jsonファイルのimagesから増幅対象の画像を検索
            for image_dict in annotation_dict["images"]:
                if str(image_dict["file_name"]) == str(imagefile_name):
                    max_image_id = max_image_id + 1
                    annotation_in_image = []

                    # annotationsから物体情報を取得
                    for annotations_dict in annotation_dict["annotations"]:
                        if annotations_dict["image_id"] == image_dict["id"]:
                            max_id = max_id + 1
                            annotation_in_image.append(
                                {
                                    "id": max_id,
                                    "iscrowd": annotations_dict["iscrowd"],
                                    "image_id": max_image_id,
                                    "category_id": annotations_dict["category_id"],
                                    "segmentation": annotations_dict["segmentation"],
                                    "area": annotations_dict["area"],
                                    "bbox": [],
                                }
                            )
                    # 増幅に必要な点群リスト生成
                    annotation_points = []
                    for annotation in annotation_in_image:
                        points = (
                            np.array(annotation["segmentation"]).reshape(-1, 2).tolist()
                        )
                        annotation_points.append(points)
                    # 画像増幅実行
                    (
                        result_image,
                        result_points_list,
                    ) = preprocessing_tools.augmente_images_segmentation(
                        image_path, annotation_points
                    )
                    # 出力画像名作成
                    augmented_file_name = augmentation_output_dir_path / Path(
                        str(Path(image_path).stem) + "_flip" + Path(image_path).suffix
                    )
                    # 画像出力
                    cv2.imwrite(str(augmented_file_name), result_image)

                    # 返り値の増幅画像のパスリストを生成
                    augmented_image_path_list.append(augmented_file_name)

                    # 増幅実行後の座標取得
                    for annotation, result_points in zip(
                        annotation_in_image,
                        result_points_list,
                    ):
                        annotation["segmentation"] = [
                            list(itertools.chain.from_iterable(result_points))
                        ]

                    # 増幅実行後のannotations情報を追加
                    annotation_dict["annotations"].extend(annotation_in_image)
                    # 増幅実行後のimages情報を追加
                    annotation_image = {
                        "id": max_image_id,
                        "width": image_dict["width"],
                        "height": image_dict["height"],
                        "file_name": str(augmented_file_name).split("/")[-1],
                        "license": image_dict["license"],
                        "flickr_url": image_dict["flickr_url"],
                        "coco_url": image_dict["coco_url"],
                        "date_captured": image_dict["date_captured"],
                    }
                    annotation_dict["images"].append(annotation_image)
                    with open(json_path, "w") as f:
                        json.dump(annotation_dict, f, indent=4)

    return augmented_image_path_list


def _SemanticSegmentation_LabelMe_dataset_augmentation(
    flip_augmentation_probability: float,
    image_path_list: list,
    json_path_list: list,
    augmentation_output_dir: str,
) -> list:
    """
    水増し画像の生成（SemanticSegmentation_LabelMe）

    Args:
        flip_augmentation_probability (float):画像反転の割合
        image_path_list (list): 画像ファイルのパスリスト
        json_path_list(list):jsonファイルのパスリスト
        augmentation_output_dir (str): 水増し画像の結果格納領域

    return:
        augmented_image_path_list(list):増幅画像のパスリスト
        augmented_json_path_list(list):増幅したアノテーションデータ（json）のパスリスト

    """
    augmented_image_path_list = []
    augmented_json_path_list = []
    # 画像反転の割合が0より大きい場合に増幅処理を行う
    if flip_augmentation_probability > 0:

        # augmentationの出力場所を作成
        augmentation_output_dir_path = Path(augmentation_output_dir)
        if not augmentation_output_dir_path.exists():
            os.makedirs(augmentation_output_dir, exist_ok=True)

        for image_path in image_path_list:
            # 画像ファイルが存在しなければ処理しない
            if Path(image_path).exists():
                if Path(image_path).suffix in IMAGE_SUFFIX:
                    pass
                else:
                    continue
            else:
                continue

            # augmentationがかかる確率
            if flip_augmentation_probability is not None:
                if random.uniform(0, 1) > float(flip_augmentation_probability):
                    continue

            # 増幅対象の画像ファイル名取得
            imagefile_name = Path(str(image_path).split("/")[-1]).stem

            # アノテーションファイルの読み込み
            for json_path in json_path_list:
                jsonfile_name = Path(str(json_path).split("/")[-1]).stem
                if str(jsonfile_name) == str(imagefile_name):

                    # jsonファイルの読み込み
                    with open(json_path, encoding="utf-8") as f:
                        annotation_dict = json.load(f)

                    # annotationsから物体情報を取得
                    annotation_points = []
                    for annotation_shape in annotation_dict["shapes"]:
                        annotation_points.append(annotation_shape["points"])

                    # 画像増幅実行
                    (
                        result_image,
                        result_points_list,
                    ) = preprocessing_tools.augmente_images_segmentation(
                        image_path, annotation_points
                    )
                    # 出力画像名作成
                    augmented_file_name = augmentation_output_dir_path / Path(
                        str(Path(image_path).stem) + "_flip" + Path(image_path).suffix
                    )
                    # 画像出力
                    cv2.imwrite(str(augmented_file_name), result_image)

                    # 返り値の増幅画像のパスリストを生成
                    augmented_image_path_list.append(augmented_file_name)

                    # 反転座標を入力
                    for annotation_shape, result_point in zip(
                        annotation_dict["shapes"], result_points_list
                    ):
                        annotation_shape["points"] = result_point

                    # 反転画像名を入力
                    annotation_dict["imagePath"] = str(
                        Path(
                            str(Path(image_path).stem)
                            + "_flip"
                            + Path(image_path).suffix
                        )
                    )

                    with open(
                        Path(augmentation_output_dir, jsonfile_name + "_flip.json"), "w"
                    ) as f:
                        json.dump(annotation_dict, f, indent=4)
                    augmented_json_path_list.append(
                        str(Path(augmentation_output_dir, jsonfile_name + "_flip.json"))
                    )

    return augmented_image_path_list, augmented_json_path_list


def _get_detections_from_pascalvoc2007(annotation_filepath: Path):

    # 存在チェック
    if annotation_filepath.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(annotation_filepath))

    # xml読み込み
    tree = ElementTree.parse(str(annotation_filepath))
    root = tree.getroot()

    # 結果を1行ずつ格納するリストの初期化
    class_names = []
    bboxes = []

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

        # 結果のリストに検出物体の各変数を格納
        class_names.append(class_name)
        bboxes.append((xmin, ymin, xmax, ymax))

    return class_names, bboxes


def _create_pascalvoc2007(image_file_path, output_file_path, class_names, bboxes):

    # 画像ファイルの親ディレクトリを取得
    img_dir = image_file_path.parent.name

    # 画像ファイルの読み込み
    img = cv2.imread(str(image_file_path))

    # 画像の読み込みに失敗した場合
    if img is None:
        raise ImageReadError("画像の読み込みに失敗しました：" + str(image_file_path))

    # 画像ファイルの高さ、幅、チャンネル数を取得
    height, width, channels = img.shape

    # Element, SubElementクラスを用いて、xmlのデータ構造を成形する。
    node_root = Element("annotation")
    node_folder = SubElement(node_root, "folder")
    node_folder.text = img_dir

    img_name = image_file_path.name
    node_filename = SubElement(node_root, "filename")
    node_filename.text = img_name
    node_path = SubElement(node_root, "path")
    node_path.text = str(image_file_path)

    node_source = SubElement(node_root, "source")
    node_database = SubElement(node_source, "database")
    node_database.text = "NA"

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
    for idx, bbox in enumerate(bboxes):

        node_object = SubElement(node_root, "object")
        node_name = SubElement(node_object, "name")
        node_name.text = class_names[idx]

        node_pose = SubElement(node_object, "pose")
        node_pose.text = "Unspecified"

        node_truncated = SubElement(node_object, "truncated")
        node_truncated.text = "0"
        node_difficult = SubElement(node_object, "difficult")
        node_difficult.text = "0"
        node_bndbox = SubElement(node_object, "bndbox")
        node_xmin = SubElement(node_bndbox, "xmin")
        node_xmin.text = str(bbox[0])
        node_ymin = SubElement(node_bndbox, "ymin")
        node_ymin.text = str(bbox[1])
        node_xmax = SubElement(node_bndbox, "xmax")
        node_xmax.text = str(bbox[2])
        node_ymax = SubElement(node_bndbox, "ymax")
        node_ymax.text = str(bbox[3])

    # xmlを出力する
    xml = tostring(node_root, pretty_print=True)
    f = open(output_file_path, "wb")
    f.write(xml)
    f.close()


def _voc_dataset_augmentation(
    augmentation: dict,
    augmentation_ratio: dict,
    image_path_list: list,
    augmentation_output_dir: str,
    split: str = "",
) -> None:
    """
    水増し画像の生成

    Args:
        augmentation (dict): augmentationの設定
            {“contrast”: True, “blur”: True, “histogram”: True, “gauss_noise”: True, “salt”: True, “flip”: True, “rotate”:3}
        augmentation_ratio (dict): augmentationの割合
            {“contrast”:1, “blur”:1, “histogram”:1, “gauss_noise”:1, “salt”:1, “flip”:1, “rotate”:1}
        image_path_list (list): 画像ファイルパス(pathlib.Path)リスト(VOC2007/JPEGImagesディレクトリ以下)
        augmentation_output_dir (str): 水増し画像の結果格納領域(VOC2007ディレクトリ)
        split (str): 振り分け種別("/train" or "/valid" or "/test") ※無しの場合は空文字("")
    Return:
        list: 増幅画像ファイルパス(pathlib.Path)リスト
    """
    # augmentationの出力場所を作成
    if not Path(augmentation_output_dir).exists():
        os.makedirs(augmentation_output_dir, exist_ok=True)

    # 拡張したファイル名一覧保存用
    augmented_file_names = []

    # 拡張画像一覧を取得
    for image_file_path in image_path_list:

        # 拡張子無しのファイル名を取得
        image_file_no_suffix = image_file_path.stem

        # 拡張子を付与したファイル名を取得
        image_file = image_file_path.name

        # 読み込みアノテーションファイル名を取得
        parent_dir_path = str(image_file_path.parent)
        annotation_file_path = Path(
            parent_dir_path.replace("JPEGImages", "Annotations"),
            image_file_no_suffix + ".xml",
        )
        if not annotation_file_path.exists():
            raise Exception("アノテーションファイルが存在しません: {}".format(str(annotation_file_path)))

        # augmentationの種類毎に処理をする
        for aug in augmentation:
            if not augmentation[aug]:
                continue

            # 角度
            degree = None

            # augmentationがかかる確率
            if random.uniform(0, 1) > float(augmentation_ratio[aug]):
                continue

            if aug == "rotate":
                # rotateはBBOXは大きくなるので注意
                degree = augmentation[aug]

            # Detectionsを作成
            class_names, bboxes = _get_detections_from_pascalvoc2007(
                annotation_file_path
            )

            # augmentationの実行
            result_image, result_bboxes = preprocessing_tools.augmente_images(
                aug, str(image_file_path), bboxes, degree
            )

            # augmentation用のファイルに出力する
            if aug == "rotate":
                _create_pascalvoc2007(
                    image_file_path,
                    str(
                        Path(
                            augmentation_output_dir,
                            "Annotations" + split,
                            image_file_no_suffix + "_" + aug + ".xml",
                        )
                    ),
                    class_names,
                    result_bboxes,
                )
            else:
                # rotate以外はコピー
                shutil.copy(
                    annotation_file_path,
                    str(
                        Path(
                            augmentation_output_dir,
                            "Annotations" + split,
                            image_file_no_suffix + "_" + aug + ".xml",
                        )
                    ),
                )

            # 画像出力
            augmented_file_name = Path(
                augmentation_output_dir,
                "JPEGImages" + split,
                image_file_no_suffix + "_" + aug + ".jpg",
            )
            cv2.imwrite(str(augmented_file_name), result_image)

            # 拡張した画像ファイルのPathを一覧に追加
            augmented_file_names.append(augmented_file_name)

    return augmented_file_names


def _mltable_dataset_augmentation(
    augmentation: dict,
    augmentation_ratio: dict,
    input_image_parent_dir: str,
    mltable_dataframe: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    水増し画像の生成

    Args:
        augmentation (dict): augmentationの設定
            {“contrast”: True, “blur”: True, “histogram”: True, “gauss_noise”: True, “salt”: True, “flip”: True, “rotate”:3}
        augmentation_ratio (dict): augmentationの割合
            {“contrast”:1, “blur”:1, “histogram”:1, “gauss_noise”:1, “salt”:1, “flip”:1, “rotate”:1}
        input_image_parent_dir (Str): 画像の親ディレクトリ
        mltable_dataframe (pandas.DataFrame) : mltable
    Return:
        pandas.DataFrame: 増幅結果mltable
    """
    # 拡張結果の作成
    mltable_columns = list(mltable_dataframe)
    result_dataframe = pd.DataFrame(columns=mltable_columns)
    for row in mltable_dataframe.itertuples():
        yolo_image_file = row.image_url
        label_array = row.label
        label_confidence = row.label_confidence
        split = row.split
        # ファイルが存在しなければ処理しない
        yolo_image_file = Path(input_image_parent_dir, str(yolo_image_file))
        if not yolo_image_file.exists():
            continue
        # testデータは処理しない
        if split == "test":
            continue

        label_list = []
        if type(label_array) == np.ndarray:
            # Detection
            label_list = label_array.tolist()
        elif type(label_array) == str:
            # Classification
            label_list.append(label_array)
        else:
            raise Exception("unexpected label format")

        # ラベルとBboxに分解
        bboxes = []
        labels = []
        for yolo_ln in label_list:
            if type(yolo_ln) == dict:
                labels.append(yolo_ln["label"])
                bboxes.append(
                    [
                        yolo_ln["topX"],
                        yolo_ln["topY"],
                        yolo_ln["bottomX"],
                        yolo_ln["bottomY"],
                    ]
                )
            else:
                # Classificationの場合
                labels.append(yolo_ln)
                bboxes = None

        # augmentationの種類毎に処理をする
        for aug in augmentation:
            if not augmentation[aug]:
                continue

            # augmentationがかかる確率
            if augmentation_ratio is not None:
                if random.uniform(0, 1) > float(augmentation_ratio[aug]):
                    continue

            degree = None

            if aug == "rotate":
                # rotateはBBOXは大きくなるので注意
                degree = augmentation[aug]
            else:
                if augmentation[aug] is False:
                    continue

            # augmentationの実行
            result_image, result_bboxes = preprocessing_tools.augmente_images(
                aug, str(yolo_image_file), bboxes, degree
            )
            augmented_image_height, augmented_image_width, _ = result_image.shape[:3]

            augmented_file_name = yolo_image_file.parent / Path(
                str(yolo_image_file.stem) + "_" + aug + yolo_image_file.suffix
            )
            # 画像出力
            cv2.imwrite(str(augmented_file_name), result_image)

            # 増幅後のラベル作成
            augmented_label_list = []
            if bboxes is not None:
                for idx, lbl in enumerate(labels):
                    augmented_label = {
                        "label": lbl,
                        "topX": result_bboxes[idx][0],
                        "topY": result_bboxes[idx][1],
                        "bottomX": result_bboxes[idx][2],
                        "bottomY": result_bboxes[idx][3],
                    }
                    augmented_label_list.append(augmented_label)
            else:
                # Classificationの場合
                augmented_label_list = labels[0]
            record = np.array(
                [
                    [
                        augmented_file_name.name,
                        augmented_label_list,
                        label_confidence,
                        augmented_image_height,
                        augmented_image_width,
                        split,
                    ]
                ]
            )

            result_dataframe = pd.concat(
                [result_dataframe, pd.DataFrame(record, columns=mltable_columns)],
                axis=0,
            )

    return result_dataframe


def _get_list_from_txt(txt_path: str) -> list:
    """ "テキストからリストを取得する

    Args:
        txt_path (str): テキストパス_

    Returns:
        list: 文字列リスト
    """
    result_list = []

    with open(txt_path, "r", encoding="utf-8") as f:
        txt_lines = f.readlines()
    for l in txt_lines:
        result_list.append(l.replace("\r", "").replace("\n", ""))

    return result_list


def _classification_dataset_augmentation(
    augmentation: dict,
    augmentation_ratio: dict,
    image_path_list: list,
    augmentation_output_dir: str,
) -> None:
    """
    水増し画像の生成

    Args:
        augmentation (dict): augmentationの設定
            {“contrast”: True, “blur”: True, “histogram”: True, “gauss_noise”: True, “salt”: True, “flip”: True, “rotate”:3}
        augmentation_ratio (dict): augmentationの割合
            {“contrast”:1, “blur”:1, “histogram”:1, “gauss_noise”:1, “salt”:1, “flip”:1, “rotate”:1}
        image_path_list (list): 画像ファイルパス(pathlib.Path)リスト(VOC2007/JPEGImagesディレクトリ以下)
        augmentation_output_dir (str): 水増し画像の結果格納領域(クラス毎ディレクトリの親ディレクトリ)
    Return:
        list: 増幅画像ファイルパス(pathlib.Path)リスト
    """
    # augmentationの出力場所を作成
    if not Path(augmentation_output_dir).exists():
        os.makedirs(augmentation_output_dir, exist_ok=True)

    augmented_image_path_list = []
    # 拡張画像一覧を取得
    for input_image in image_path_list:

        # 拡張子無しのファイル名を取得
        image_file_no_suffix = input_image.stem

        # augmentationの種類毎に処理をする
        for aug in augmentation:
            if not augmentation[aug]:
                continue

            # 角度
            degree = None

            # augmentationがかかる確率
            if random.uniform(0, 1) > float(augmentation_ratio[aug]):
                continue

            if aug == "rotate":
                # rotateはBBOXは大きくなるので注意
                degree = augmentation[aug]

            # augmentationの実行
            result_image, result_bboxes = preprocessing_tools.augmente_images(
                aug, str(input_image), None, degree
            )

            # 画像出力
            image_output_dir_path = Path(
                augmentation_output_dir, input_image.parent.name
            )
            try:
                if not image_output_dir_path.exists():
                    os.makedirs(str(image_output_dir_path))
            except Exception:
                continue
            augmented_file_name = Path(
                str(image_output_dir_path), image_file_no_suffix + "_" + aug + ".jpg"
            )
            cv2.imwrite(str(augmented_file_name), result_image)

            augmented_image_path_list.append(augmented_file_name)
    return augmented_image_path_list


def _SemanticSegmentation_mltable_dataset_augmentation(
    augmentation_input_dir: str,
    mltable_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    SemanticSegmentationのmltable形式データを増加した内容をmltableに反映する

    Args:
        augmentation_input_dir (str): 水増しjsonファイルの結果格納領域
        mltable_dataframe (pandas.DataFrame) : mltable
    Return:
        pandas.DataFrame: 増幅結果mltable
    """
    # 拡張結果の作成
    mltable_columns = list(mltable_dataframe)
    result_dataframe = pd.DataFrame(columns=mltable_columns)

    for file_path in Path(augmentation_input_dir).glob("**/*"):
        if file_path.suffix == ".json":
            with open(file_path, encoding="utf-8") as f:
                annotation_dict = json.load(f)

            image_uri = Path(augmentation_input_dir, annotation_dict["imagePath"])

            augmented_image_height = annotation_dict["imageHeight"]
            augmented_image_width = annotation_dict["imageWidth"]

            # annotationsから物体情報を取得
            label = []
            label_confidence = []
            for annotation_shape in annotation_dict["shapes"]:

                # 矩形取得
                x, y, w, h = cv2.boundingRect(
                    np.array(annotation_shape["points"]).astype(int)
                )
                min_x = x / augmented_image_width
                min_y = y / augmented_image_height
                max_x = (x + w) / augmented_image_width
                max_y = (y + h) / augmented_image_height
                bbox_list = [min_x, min_y, max_x, max_y]

                # 座標の変更
                polygon = []
                for point in annotation_shape["points"]:
                    point_x = point[0] / augmented_image_width
                    point_y = point[1] / augmented_image_height
                    polygon.append(point_x)
                    polygon.append(point_y)

                augmented_label = {
                    "label": annotation_shape["label"],
                    "isCrowd": False,
                    "polygon": np.array([polygon]),
                    "bbox": np.array(bbox_list),
                }
                label.append(augmented_label)
                label_confidence.append(1.0)

            record = np.array(
                [
                    [
                        image_uri.name,
                        label,
                        label_confidence,
                        augmented_image_height,
                        augmented_image_width,
                    ]
                ]
            )

            result_dataframe = pd.concat(
                [result_dataframe, pd.DataFrame(record, columns=mltable_columns)],
                axis=0,
            )

    return result_dataframe


def data_prep(
    dataset_directory: str,
    # mltable_dataset: str,
    dataset_format: str,
    prep_mode: str,
    train_data_ratio: float,
    valid_data_ratio: float,
    test_data_ratio: float,
    contrast_augmentation_probability: float,
    blur_augmentation_probability: float,
    histogram_augmentation_probability: float,
    gauss_noise_augmentation_probability: float,
    salt_augmentation_probability: float,
    flip_augmentation_probability: float,
    rotate_augmentation_probability: float,
    rotate_augmentation_degree: float,
    preped_dataset_directory: str,
) -> None:
    # with mlflow.start_run() as mlflow_run:
    try:
        # KQIディレクトリ構造の追加
        os.makedirs(KQI_INPUT_DIR_PATH, exist_ok=True)
        kqi_input_dir_path = Path(KQI_INPUT_DIR_PATH)
        os.makedirs(KQI_OUTPUT_DIR_PATH, exist_ok=True)
        kqi_output_dir_path = Path(KQI_OUTPUT_DIR_PATH)

        # ロガー取得
        logger = logging_tools.set_logger(KQI_OUTPUT_DIR_PATH)
        # logger.info("start job: \n{}".format(str(mlflow_run.info)))

        # コンポーネント出力ディレクトリの取得
        preped_dataset_dir_path = Path(preped_dataset_directory)
        # 増幅画像出力ディレクトリの作成
        augmented_images_output_dir_path = (
            kqi_output_dir_path / AUGMENTED_IMAGES_DIRECTORY_NAME
        )
        os.makedirs(str(augmented_images_output_dir_path / "train"), exist_ok=True)
        os.makedirs(str(augmented_images_output_dir_path / "valid"), exist_ok=True)
        os.makedirs(str(augmented_images_output_dir_path / "test"), exist_ok=True)

        # 入力取得
        if dataset_directory is not None:
            # uri_folderからデータダウンロード
            logger.info("download input data: {}".format(str(dataset_directory)))
            shutil.copytree(
                dataset_directory, str(kqi_input_dir_path), dirs_exist_ok=True
            )
        elif mltable_dataset is not None:
            raise NotImplementedError("MLTable形式のデータ処理は無効化されています")

        else:
            raise Exception("not found input data")

        # zip解凍
        logger.info("decompose zip")
        input_handler.decompress_zip(
            str(kqi_input_dir_path), str(kqi_input_dir_path)
        )

        # 振り分けを実行
        is_split = prep_mode == "split_augment"
        directory = [
            "Detection: YOLO(directory)",
            "Detection: VOC(directory)",
            "Classification: folder-tree(directory)",
            "SemanticSegmentation: EIseg(directory)",
            "SemanticSegmentation: LabelMe(directory)",
        ]
        train_image_path_list = []
        valid_image_path_list = []
        test_image_path_list = []
        num_test_image = 0
        split_txt_file_tmp_dir = Path(os.getcwd())

        # 振り分けモード（directory）
        if is_split:
            logger.info("split_augment mode")

            # directoryの場合
            if dataset_format in directory:
                preprocessing_tools.distribute_images_for_detection(
                    str(kqi_input_dir_path),
                    str(split_txt_file_tmp_dir),
                    train_data_ratio,
                    test_data_ratio,
                    valid_data_ratio,
                )
                train_txt_path = (
                    split_txt_file_tmp_dir / preprocessing_tools.TRAIN_FILE_NAME
                )
                valid_txt_path = (
                    split_txt_file_tmp_dir
                    / preprocessing_tools.VALIDATION_FILE_NAME
                )
                test_txt_path = (
                    split_txt_file_tmp_dir / preprocessing_tools.TEST_FILE_NAME
                )
                if train_txt_path.exists():
                    train_image_path_list = [
                        Path(p) for p in _get_list_from_txt(str(train_txt_path))
                    ]
                else:
                    train_txt_path.touch()
                if valid_txt_path.exists():
                    valid_image_path_list = [
                        Path(p) for p in _get_list_from_txt(str(valid_txt_path))
                    ]
                else:
                    valid_txt_path.touch()
                if test_txt_path.exists():
                    test_image_path_list = [
                        Path(p) for p in _get_list_from_txt(str(test_txt_path))
                    ]
                else:
                    test_txt_path.touch()
                # メトリック表示用の数値を取得
                num_test_image = len(test_image_path_list)

                if (
                    len(train_image_path_list) <= 0
                    and len(valid_image_path_list) <= 0
                    and len(test_image_path_list) <= 0
                ):
                    raise Exception("対象となる画像がありません")

        elif not is_split:
            logger.info("augment only mode")
            # 振り分けを実行しない場合は全てtrain
            for file_path in kqi_input_dir_path.glob("**/*"):
                if file_path.suffix in IMAGE_SUFFIX:
                    train_image_path_list.append(Path(file_path))

            if len(train_image_path_list) <= 0:
                raise Exception("対象となる画像がありません")

            with open(
                str(split_txt_file_tmp_dir / preprocessing_tools.TRAIN_FILE_NAME),
                "w",
                encoding="utf-8",
            ) as f:
                f.writelines([p.name + "\n" for p in train_image_path_list])

        # augmentation設定の作成
        is_augmentation = {
            "contrast": True if contrast_augmentation_probability > 0 else False,
            "blur": True if blur_augmentation_probability > 0 else False,
            "histogram": True if histogram_augmentation_probability > 0 else False,
            "gauss_noise": True
            if gauss_noise_augmentation_probability > 0
            else False,
            "salt": True if salt_augmentation_probability > 0 else False,
            "flip": True if flip_augmentation_probability > 0 else False,
            "rotate": rotate_augmentation_degree
            if rotate_augmentation_probability > 0
            else 0,
        }
        augmentation_setting = {
            "contrast": contrast_augmentation_probability,
            "blur": blur_augmentation_probability,
            "histogram": histogram_augmentation_probability,
            "gauss_noise": gauss_noise_augmentation_probability,
            "salt": salt_augmentation_probability,
            "flip": flip_augmentation_probability,
            "rotate": rotate_augmentation_probability,
        }

        if dataset_format == "Detection: YOLO(directory)":
            logger.info("dataset format: YOLO")
            # ディレクトリ構造の作成
            (preped_dataset_dir_path / "data").mkdir(parents=True, exist_ok=True)
            (preped_dataset_dir_path / "text").mkdir(parents=True, exist_ok=True)

            # クラスリスト抽出
            yolo_classes_txt_path = None
            for file_path in kqi_input_dir_path.glob("**/*"):
                if file_path.is_file() and file_path.name == "classes.txt":
                    yolo_classes_txt_path = file_path
                    break

            if yolo_classes_txt_path is None:
                raise Exception("not found classes.txt")
            yolo_class_list = _get_list_from_txt(yolo_classes_txt_path)
            shutil.move(
                str(yolo_classes_txt_path),
                str(preped_dataset_dir_path / "text" / "classes.txt"),
            )
            logger.info("get class list: {}".format(str(yolo_class_list)))

            # 画像増幅
            is_voc = False
            if len(train_image_path_list) > 0:
                logger.info(
                    "train image augmentation. original {} images".format(
                        str(len(train_image_path_list))
                    )
                )
                (
                    augmented_train_image_path_list,
                    is_voc,
                ) = _yolo_dataset_augmentation(
                    is_augmentation,
                    augmentation_setting,
                    train_image_path_list,
                    str(preped_dataset_dir_path / "data"),
                    yolo_class_list,
                )
                # mlflow.log_metric(
                #     "学習(train)画像枚数",
                #     len(train_image_path_list)
                #     + len(augmented_train_image_path_list),
                # )
                for augmented_train_image_path in augmented_train_image_path_list:
                    shutil.copy(
                        str(augmented_train_image_path),
                        str(
                            augmented_images_output_dir_path
                            / "train"
                            / augmented_train_image_path.name
                        ),
                    )

            if len(valid_image_path_list) > 0:
                logger.info(
                    "valid image augmentation. original {} images".format(
                        str(len(valid_image_path_list))
                    )
                )
                (
                    augmented_valid_image_path_list,
                    is_voc,
                ) = _yolo_dataset_augmentation(
                    is_augmentation,
                    augmentation_setting,
                    valid_image_path_list,
                    str(preped_dataset_dir_path / "data"),
                    yolo_class_list,
                )
                # mlflow.log_metric(
                #     "検証(valid)画像枚数",
                #     len(valid_image_path_list)
                #     + len(augmented_valid_image_path_list),
                # )
                for augmented_valid_image_path in augmented_valid_image_path_list:
                    shutil.copy(
                        str(augmented_valid_image_path),
                        str(
                            augmented_images_output_dir_path
                            / "valid"
                            / augmented_valid_image_path.name
                        ),
                    )

            # 元画像・アノテーションファイルの移動
            annotation_suffix = ".xml" if is_voc else ".txt"
            for train_image_path in train_image_path_list:
                shutil.move(
                    str(train_image_path),
                    str(preped_dataset_dir_path / "data" / train_image_path.name),
                )
                shutil.move(
                    str(
                        train_image_path.parent
                        / (train_image_path.stem + annotation_suffix)
                    ),
                    str(
                        preped_dataset_dir_path
                        / "data"
                        / (train_image_path.stem + annotation_suffix)
                    ),
                )
            for valid_image_path in valid_image_path_list:
                shutil.move(
                    str(valid_image_path),
                    str(preped_dataset_dir_path / "data" / valid_image_path.name),
                )
                shutil.move(
                    str(
                        valid_image_path.parent
                        / (valid_image_path.stem + annotation_suffix)
                    ),
                    str(
                        preped_dataset_dir_path
                        / "data"
                        / (valid_image_path.stem + annotation_suffix)
                    ),
                )
            for test_image_path in test_image_path_list:
                shutil.move(
                    str(test_image_path),
                    str(preped_dataset_dir_path / "data" / test_image_path.name),
                )
                shutil.move(
                    str(
                        test_image_path.parent
                        / (test_image_path.stem + annotation_suffix)
                    ),
                    str(
                        preped_dataset_dir_path
                        / "data"
                        / (test_image_path.stem + annotation_suffix)
                    ),
                )

            # 画像リスト作成
            if len(train_image_path_list) > 0:
                train_image_path_list.extend(augmented_train_image_path_list)
                train_image_list_txt_path = (
                    preped_dataset_dir_path / "text" / "train.txt"
                )
                with open(
                    str(train_image_list_txt_path), "w", encoding="utf-8"
                ) as f:
                    f.writelines([p.name + "\n" for p in train_image_path_list])

            if len(valid_image_path_list) > 0:
                valid_image_path_list.extend(augmented_valid_image_path_list)
                valid_image_list_txt_path = (
                    preped_dataset_dir_path / "text" / "valid.txt"
                )
                with open(
                    str(valid_image_list_txt_path), "w", encoding="utf-8"
                ) as f:
                    f.writelines([p.name + "\n" for p in valid_image_path_list])

            if len(test_image_path_list) > 0:
                test_image_list_txt_path = (
                    preped_dataset_dir_path / "text" / "test.txt"
                )
                with open(
                    str(test_image_list_txt_path), "w", encoding="utf-8"
                ) as f:
                    f.writelines([p.name + "\n" for p in test_image_path_list])

        elif dataset_format == "Detection: VOC(directory)":
            logger.info("dataset format: VOC")

            for d in [
                "VOC2007",
                "VOC2007/Annotations",
                "VOC2007/ImageSets/Main",
                "VOC2007/JPEGImages",
            ]:
                os.makedirs(str(preped_dataset_dir_path / d), exist_ok=True)

            if len(train_image_path_list) > 0:
                logger.info(
                    "train image augmentation. original {} images".format(
                        str(len(train_image_path_list))
                    )
                )
                augmented_train_image_path_list = _voc_dataset_augmentation(
                    is_augmentation,
                    augmentation_setting,
                    train_image_path_list,
                    str(preped_dataset_dir_path / "VOC2007"),
                    "",
                )
                # mlflow.log_metric(
                #     "学習(train)画像枚数",
                #     len(train_image_path_list)
                #     + len(augmented_train_image_path_list),
                # )
                for augmented_train_image_path in augmented_train_image_path_list:
                    shutil.copy(
                        str(augmented_train_image_path),
                        str(
                            augmented_images_output_dir_path
                            / "train"
                            / augmented_train_image_path.name
                        ),
                    )
            if len(valid_image_path_list) > 0:
                logger.info(
                    "valid image augmentation. original {} images".format(
                        str(len(valid_image_path_list))
                    )
                )
                augmented_valid_image_path_list = _voc_dataset_augmentation(
                    is_augmentation,
                    augmentation_setting,
                    valid_image_path_list,
                    str(preped_dataset_dir_path / "VOC2007"),
                    "",
                )
                # mlflow.log_metric(
                #     "検証(valid)画像枚数",
                #     len(valid_image_path_list)
                #     + len(augmented_valid_image_path_list),
                # )
                for augmented_valid_image_path in augmented_valid_image_path_list:
                    shutil.copy(
                        str(augmented_valid_image_path),
                        str(
                            augmented_images_output_dir_path
                            / "valid"
                            / augmented_valid_image_path.name
                        ),
                    )

            # 元画像・アノテーションファイルの移動
            all_image_path_list = (
                train_image_path_list + valid_image_path_list + test_image_path_list
            )
            for image_path in all_image_path_list:
                shutil.move(
                    str(image_path),
                    str(
                        preped_dataset_dir_path
                        / "VOC2007"
                        / "JPEGImages"
                        / image_path.name
                    ),
                )
                try:
                    shutil.move(
                        str(
                            Path(
                                str(image_path.parent).replace(
                                    "JPEGImages", "Annotations"
                                ),
                                image_path.stem + ".xml",
                            )
                        ),
                        str(
                            preped_dataset_dir_path
                            / "VOC2007"
                            / "Annotations"
                            / (image_path.stem + ".xml")
                        ),
                    )
                except Exception:
                    logger.exception("not found original annotation file")
                    continue
            # 画像リスト作成
            imageset_main_dir_path = (
                preped_dataset_dir_path / "VOC2007" / "ImageSets" / "Main"
            )
            if len(train_image_path_list) > 0:
                train_image_path_list.extend(augmented_train_image_path_list)
                train_image_list_txt_path = imageset_main_dir_path / "train.txt"
                with open(
                    str(train_image_list_txt_path), "w", encoding="utf-8"
                ) as f:
                    f.writelines([p.stem + "\n" for p in train_image_path_list])

            if len(valid_image_path_list) > 0:
                valid_image_path_list.extend(augmented_valid_image_path_list)
                valid_image_list_txt_path = imageset_main_dir_path / "valid.txt"
                with open(
                    str(valid_image_list_txt_path), "w", encoding="utf-8"
                ) as f:
                    f.writelines([p.stem + "\n" for p in valid_image_path_list])

            if len(test_image_path_list) > 0:
                test_image_list_txt_path = imageset_main_dir_path / "test.txt"
                with open(
                    str(test_image_list_txt_path), "w", encoding="utf-8"
                ) as f:
                    f.writelines([p.stem + "\n" for p in test_image_path_list])


        ###############################################
        # Classification: folder-tree(directory) の場合
        ###############################################
        elif dataset_format == "Classification: folder-tree(directory)":
            logger.info("dataset format: folder-tree")

            ########################
            # 分割先ディレクトリ作成
            ########################
            os.makedirs(
                str(preped_dataset_dir_path / "images" / "train"), exist_ok=True
            )
            os.makedirs(
                str(preped_dataset_dir_path / "images" / "valid"), exist_ok=True
            )
            os.makedirs(
                str(preped_dataset_dir_path / "images" / "test"), exist_ok=True
            )


            #######################################################
            # 学習用データの画像増強と、生成画像を別の出力フォルダもコピー
            #######################################################
            if len(train_image_path_list) > 0:
                logger.info(
                    "train image augmentation. original {} images".format(
                        str(len(train_image_path_list))
                    )
                )
                augmented_train_image_path_list = (
                    _classification_dataset_augmentation(
                        is_augmentation,
                        augmentation_setting,
                        train_image_path_list,
                        str(preped_dataset_dir_path / "images" / "train"),
                    )
                )
                for augmented_train_image_path in augmented_train_image_path_list:
                    shutil.copy(
                        str(augmented_train_image_path),
                        str(
                            augmented_images_output_dir_path
                            / "train"
                            / augmented_train_image_path.name
                        ),
                    )

            #######################################################
            # 検証用データの画像増強と、生成画像を別の出力フォルダもコピー
            #######################################################
            if len(valid_image_path_list) > 0:
                logger.info(
                    "valid image augmentation. original {} images".format(
                        str(len(valid_image_path_list))
                    )
                )
                augmented_valid_image_path_list = (
                    _classification_dataset_augmentation(
                        is_augmentation,
                        augmentation_setting,
                        valid_image_path_list,
                        str(preped_dataset_dir_path / "images" / "valid"),
                    )
                )
                for augmented_valid_image_path in augmented_valid_image_path_list:
                    shutil.copy(
                        str(augmented_valid_image_path),
                        str(
                            augmented_images_output_dir_path
                            / "valid"
                            / augmented_valid_image_path.name
                        ),
                    )


            ###################
            # 元画像の移動
            ###################
            for train_image_path in train_image_path_list:
                train_image_output_dir_path = (
                    preped_dataset_dir_path
                    / "images"
                    / "train"
                    / train_image_path.parent.name
                )
                if not train_image_output_dir_path.exists():
                    os.makedirs(str(train_image_output_dir_path))
                shutil.move(
                    str(train_image_path),
                    str(train_image_output_dir_path / train_image_path.name),
                )
            for valid_image_path in valid_image_path_list:
                valid_image_output_dir_path = (
                    preped_dataset_dir_path
                    / "images"
                    / "valid"
                    / valid_image_path.parent.name
                )
                if not valid_image_output_dir_path.exists():
                    os.makedirs(str(valid_image_output_dir_path))
                shutil.move(
                    str(valid_image_path),
                    str(valid_image_output_dir_path / valid_image_path.name),
                )
            for test_image_path in test_image_path_list:
                test_image_output_dir_path = (
                    preped_dataset_dir_path
                    / "images"
                    / "test"
                    / test_image_path.parent.name
                )
                if not test_image_output_dir_path.exists():
                    os.makedirs(str(test_image_output_dir_path))
                shutil.move(
                    str(test_image_path),
                    str(test_image_output_dir_path / test_image_path.name),
                )

        elif dataset_format in [
            "SemanticSegmentation: EIseg(directory)",
            "SemanticSegmentation: LabelMe(directory)",
        ]:

            image_path_list = []
            json_path_list = []

            # グレースケール画像の格納ディレクトリ作成
            graycolor_image_dir = str(
                split_txt_file_tmp_dir / "graycolor_image_dir"
            )
            os.makedirs(graycolor_image_dir, exist_ok=True)

            # 存在チェック
            if kqi_input_dir_path.exists() is False:
                raise FileNotFoundError("ディレクトリが見つかりません：" + str(kqi_input_dir_path))
            # ディレクトリチェック
            if kqi_input_dir_path.is_dir() is False:
                raise NotADirectoryError("ディレクトリではありません：" + str(kqi_input_dir_path))

            # データセットから画像とjsonのパスリストをそれぞれ取得
            for file_path in kqi_input_dir_path.glob("**/*"):
                if file_path.suffix in IMAGE_SUFFIX:
                    image_path_list.append(str(Path(file_path)))
                elif file_path.suffix == ".json":
                    json_path_list.append(str(Path(file_path)))

            # 増幅データの保存先を作成（train用）
            os.makedirs(
                str(augmented_images_output_dir_path / "train"), exist_ok=True
            )
            # 増幅データの保存先を作成（valid用）
            os.makedirs(
                str(augmented_images_output_dir_path / "valid"), exist_ok=True
            )

            if dataset_format == "SemanticSegmentation: EIseg(directory)":

                # クラスリスト生成
                class_txt_path = preprocessing_tools.generate_classestxt_from_json(
                    "SemanticSegmentation: EIseg(directory)",
                    json_path_list,
                    split_txt_file_tmp_dir,
                )

                # 上書きできるように作業ディレクトリにjsonファイルを保存
                json_path = json_path_list[0]
                shutil.copy(str(json_path), str(split_txt_file_tmp_dir))

                # train用データの増幅処理
                train_augmented_image_path_list = (
                    _SemanticSegmentation_EIseg_dataset_augmentation(
                        flip_augmentation_probability,
                        train_image_path_list,
                        str(split_txt_file_tmp_dir / "annotations.json"),
                        str(augmented_images_output_dir_path / "train"),
                    )
                )

                # valid用データの増幅処理
                valid_augmented_image_path_list = (
                    _SemanticSegmentation_EIseg_dataset_augmentation(
                        flip_augmentation_probability,
                        valid_image_path_list,
                        str(split_txt_file_tmp_dir / "annotations.json"),
                        str(augmented_images_output_dir_path / "valid"),
                    )
                )

                # グレースケール画像生成
                graycolor_image_path_list = (
                    preprocessing_tools.generate_graycolor_image_from_json(
                        "SemanticSegmentation: EIseg(directory)",
                        [str(split_txt_file_tmp_dir / "annotations.json")],
                        class_txt_path,
                        graycolor_image_dir,
                    )
                )

                # annotations.json出力
                shutil.copy(
                    str(split_txt_file_tmp_dir / "annotations.json"),
                    str(augmented_images_output_dir_path),
                )

            elif dataset_format == "SemanticSegmentation: LabelMe(directory)":

                # クラスリスト生成
                class_txt_path = preprocessing_tools.generate_classestxt_from_json(
                    "SemanticSegmentation: LabelMe(directory)",
                    json_path_list,
                    split_txt_file_tmp_dir,
                )

                # train用データの増幅処理
                (
                    train_augmented_image_path_list,
                    train_augmented_json_path_list,
                ) = _SemanticSegmentation_LabelMe_dataset_augmentation(
                    flip_augmentation_probability,
                    train_image_path_list,
                    json_path_list,
                    str(augmented_images_output_dir_path / "train"),
                )

                # valid用データの増幅処理
                (
                    valid_augmented_image_path_list,
                    valid_augmented_json_path_list,
                ) = _SemanticSegmentation_LabelMe_dataset_augmentation(
                    flip_augmentation_probability,
                    valid_image_path_list,
                    json_path_list,
                    str(augmented_images_output_dir_path / "valid"),
                )

                if len(train_augmented_json_path_list) > 0:
                    json_path_list = json_path_list + train_augmented_json_path_list
                if len(valid_augmented_json_path_list) > 0:
                    json_path_list = json_path_list + valid_augmented_json_path_list

                # グレースケール画像生成
                graycolor_image_path_list = (
                    preprocessing_tools.generate_graycolor_image_from_json(
                        "SemanticSegmentation: LabelMe(directory)",
                        json_path_list,
                        class_txt_path,
                        graycolor_image_dir,
                    )
                )

            else:
                raise Exception("設定パラメータdataset_formatの形式に誤りがあります。")

            # 各画像ファイルパスリストに追加
            if len(train_augmented_image_path_list) > 0:
                image_path_list = image_path_list + train_augmented_image_path_list
                train_image_path_list = (
                    train_image_path_list + train_augmented_image_path_list
                )
            if len(valid_augmented_image_path_list) > 0:
                image_path_list = image_path_list + valid_augmented_image_path_list
                valid_image_path_list = (
                    valid_image_path_list + valid_augmented_image_path_list
                )
            # mlflow.log_metric(
            #     "学習(train)画像枚数",
            #     len(train_image_path_list),
            # )
            # mlflow.log_metric(
            #     "検証(valid)画像枚数",
            #     len(valid_image_path_list),
            # )

            # 学習コンポーネントに渡すフォルダを作成
            for d in [
                "data",
                "data/list",
                "data/list/cityscapes",
                "data/VOCDevkit",
                "data/VOCDevkit/JpegImages",
                "data/VOCDevkit/JpegImages/train",
                "data/VOCDevkit/JpegImages/valid",
                "data/VOCDevkit/JpegImages/test",
                "data/VOCDevkit/PngImages",
                "data/VOCDevkit/PngImages/train",
                "data/VOCDevkit/PngImages/valid",
                "data/VOCDevkit/PngImages/test",
            ]:
                os.makedirs(str(preped_dataset_dir_path / d), exist_ok=True)

            # classes.txtを格納
            shutil.copy(
                str(class_txt_path),
                str(preped_dataset_dir_path),
            )

            # train画像とグレースケール画像を格納
            train_lst_list = []
            for train_image_path in train_image_path_list:
                shutil.copy(
                    str(train_image_path),
                    str(
                        preped_dataset_dir_path / "data/VOCDevkit/JpegImages/train"
                    ),
                )
                gray_train_image_path = Path(
                    graycolor_image_dir,
                    Path(str(train_image_path).split("/")[-1]).stem + "_gray.png",
                )
                if gray_train_image_path.exists():
                    shutil.copy(
                        str(gray_train_image_path),
                        str(
                            preped_dataset_dir_path
                            / "data/VOCDevkit/PngImages/train"
                        ),
                    )
                    image_path = str(
                        "data/VOCDevkit/JpegImages/train"
                        / Path(str(train_image_path).split("/")[-1])
                    )
                    grayimage_path = str(
                        Path(
                            "data/VOCDevkit/PngImages/train",
                            Path(str(train_image_path).split("/")[-1]).stem
                            + "_gray.png",
                        )
                    )
                    train_lst_list.append(
                        str(image_path) + "\t" + str(grayimage_path) + "\n"
                    )
                else:
                    raise Exception("グレースケール画像が作成されていません")

            # valid画像とグレースケール画像を格納
            valid_lst_list = []
            for valid_image_path in valid_image_path_list:
                shutil.copy(
                    str(valid_image_path),
                    str(
                        preped_dataset_dir_path / "data/VOCDevkit/JpegImages/valid"
                    ),
                )
                gray_valid_image_path = Path(
                    graycolor_image_dir,
                    Path(str(valid_image_path).split("/")[-1]).stem + "_gray.png",
                )
                if gray_valid_image_path.exists():
                    shutil.copy(
                        str(gray_valid_image_path),
                        str(
                            preped_dataset_dir_path
                            / "data/VOCDevkit/PngImages/valid"
                        ),
                    )
                    image_path = str(
                        "data/VOCDevkit/JpegImages/valid"
                        / Path(str(valid_image_path).split("/")[-1])
                    )
                    grayimage_path = str(
                        Path(
                            "data/VOCDevkit/PngImages/valid",
                            Path(str(valid_image_path).split("/")[-1]).stem
                            + "_gray.png",
                        )
                    )
                    valid_lst_list.append(
                        str(image_path) + "\t" + str(grayimage_path) + "\n"
                    )
                else:
                    raise Exception("グレースケール画像が作成されていません")

            # test画像とグレースケール画像を格納
            test_lst_list = []
            for test_image_path in test_image_path_list:
                shutil.copy(
                    str(test_image_path),
                    str(preped_dataset_dir_path / "data/VOCDevkit/JpegImages/test"),
                )
                gray_test_image_path = Path(
                    graycolor_image_dir,
                    Path(str(test_image_path).split("/")[-1]).stem + "_gray.png",
                )
                if gray_test_image_path.exists():
                    shutil.copy(
                        str(gray_test_image_path),
                        str(
                            preped_dataset_dir_path
                            / "data/VOCDevkit/PngImages/test"
                        ),
                    )
                    image_path = str(
                        "data/VOCDevkit/JpegImages/test"
                        / Path(str(test_image_path).split("/")[-1])
                    )
                    grayimage_path = str(
                        Path(
                            "data/VOCDevkit/PngImages/test",
                            Path(str(test_image_path).split("/")[-1]).stem
                            + "_gray.png",
                        )
                    )
                    test_lst_list.append(
                        str(image_path) + "\t" + str(grayimage_path) + "\n"
                    )
                else:
                    raise Exception("グレースケール画像が作成されていません")

            # lstファイル名定義
            trainlst_file_path = pathlib.Path(
                preped_dataset_dir_path, "data/list/cityscapes", "train.lst"
            )
            testlst_file_path = pathlib.Path(
                preped_dataset_dir_path, "data/list/cityscapes", "test.lst"
            )
            validationlst_file_path = pathlib.Path(
                preped_dataset_dir_path, "data/list/cityscapes", "valid.lst"
            )

            # 書き込む内容が存在する場合は、lstファイル出力
            if not len(train_lst_list) == 0:
                with open(
                    trainlst_file_path, "w", encoding="utf-8", newline="\n"
                ) as f:
                    f.writelines(train_lst_list)
            if not len(test_lst_list) == 0:
                with open(
                    testlst_file_path, "w", encoding="utf-8", newline="\n"
                ) as f:
                    f.writelines(test_lst_list)
            if not len(valid_lst_list) == 0:
                with open(
                    validationlst_file_path, "w", encoding="utf-8", newline="\n"
                ) as f:
                    f.writelines(valid_lst_list)

        else:
            raise Exception("dataset_format is unexpected format")

        # 成果物の出力
        # log_artifactsの第一引数で指定したフォルダ直下がGUI上のジョブ結果に表示される
        # mlflow.log_metric("評価(test)画像枚数", num_test_image)
        # mlflow.log_artifacts(str(kqi_output_dir_path))

    except Exception as e:
        logger.exception("data prep error: {}".format(e))
        # mlflow.log_artifacts(str(kqi_output_dir_path))
        # mlflow.log_artifacts(str(preped_dataset_dir_path))
        raise
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
**作成日:** 2021/05/19
**作成者:** 成田　伊織 \n

推論操作の共通処理

"""

import glob
import pathlib
import json
import cv2
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from lxml.etree import Element, SubElement, tostring
from xml.etree import ElementTree
from PIL.ImageColor import getrgb
from PIL import Image
from logging import getLogger
from sklearn.metrics import multilabel_confusion_matrix

from fw_preprocessing.gzo_fw_common.kqi_processing import preprocessing_tools
from fw_preprocessing.gzo_fw_common.common.common_tools import load_common_config
from fw_preprocessing.gzo_fw_common.common.gzo_fw_exception import ImageReadError
import string
import shutil
import numpy as np
import csv
import os
import time
import base64

# 定数
# 該当要素が存在しないに代入する文字列
NA = "NA"

# 共通設定を読み込む
common_config = load_common_config()

# ロガーを読み込む
logger = getLogger("log")

# ファイル操作時エンコーディング
ENCODING_UTF8 = "utf-8"
IMAGE_SUFFIX = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def draw_detection_results_on_image(
    image_file_path, bboxes, classes, scores, class_names
):
    """
    Detection推論結果(bbox/class/score)を画像に描画する

    Args:
        image_file_path (str): 入力画像ファイルパス
        bboxes ([[int]]): 画像の推論結果におけるbboxのリスト(bbox形式[xmin, ymin, xmax, ymax])
        classes ([int]): 画像の推論結果におけるクラスのリスト
        scores ([float]): 画像の推論結果におけるスコアのリスト
        class_names ([str]): クラス名のリスト (coco.namesやclasses.txtと同じ順番に要素を格納する)

    Returns:
        Mat: output_image -- 出力画像

    Raise:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        ImageReadError: 画像の読み込みに失敗した場合に発生
    """

    logger.info("開始：Detection結果画像生成")

    # Pathオブジェクトの取得
    image_file = pathlib.Path(image_file_path)

    # 存在チェック
    if image_file.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(image_file))

    # 共通設定から描画用の設定を取得
    score_round = Decimal(common_config.get("draw_detection", "score_round"))
    colors = json.loads(common_config.get("draw_detection", "colors"))
    font_type = int(common_config.get("draw_detection", "font_type"))
    font_size = int(common_config.get("draw_detection", "font_size"))
    font_size_bold = font_size * 1.2
    line_thickness = int(common_config.get("draw_detection", "line_thickness"))
    line_type = int(common_config.get("draw_detection", "line_type"))

    # クラス名と色のリスト化（クラス名から色を参照するため）
    color_dict = {}
    for color_idx, class_name in enumerate(class_names):
        color = colors[color_idx % len(colors)]
        color_dict[class_name] = getrgb(color)

    # 画像の読み込み
    output_image = cv2.imread(str(image_file))

    # 画像の読み込みに失敗した場合
    if output_image is None:
        raise ImageReadError("画像の読み込みに失敗しました：" + str(image_file))

    # 画像に描画する
    for bbox, cls, score in zip(bboxes, classes, scores):
        # ラベルを取得
        label = class_names[cls]

        # 描画する色を取得
        color = color_dict[label]

        # スコアをDecimal型に変更
        score = Decimal(str(score)).quantize(score_round, rounding=ROUND_HALF_UP)

        # bboxの座標から各頂点の座標を取得
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]

        # 画像に枠線を描画
        output_image = cv2.rectangle(
            output_image, (left, top), (right, bottom), color, 3
        )

        # 表示テキストを成形
        put_text = "{} : {}".format(label, str(score))

        # テキストを表示する場所（左下の座標）
        text_org = (left, top)

        # テキストボックスのサイズ
        text_size, _ = cv2.getTextSize(put_text, font_type, font_size_bold, 0)
        text_width = text_size[0]
        text_height = int(text_size[1] * 1.2)

        # テキストボックスの描画
        cv2.rectangle(
            output_image,
            (left - 2, top - text_height),
            (left + text_width, top),
            color,
            -1,
        )

        # テキストの書き込み
        cv2.putText(
            output_image,
            put_text,
            text_org,
            font_type,
            font_size,
            (0, 0, 0),
            line_thickness + 1,
            line_type,
        )
        cv2.putText(
            output_image,
            put_text,
            text_org,
            font_type,
            font_size,
            (255, 255, 255),
            line_thickness,
            line_type,
        )

    logger.info("終了：Detection結果画像生成")
    return output_image


def export_detection_results(
    image_file_path_list: list,
    output_file_path_list: list,
    bboxes_list: list,
    classes_list: list,
    scores_list: list,
    class_names_list: list,
    save_format: str,
) -> None:
    """
    Detection推論結果をアノテーションデータ(PascalVOC2007形式またはAzureML形式)として生成する

    Args:
        image_file_path_list (list): 入力画像ファイルパスのリスト
        output_file_path_list (list): 出力ファイルパスのリスト
        bboxes_list (list): bbox([[int]])のリスト bbox形式[xmin, ymin, xmax, ymax]
        classes (list): クラス([int])のリスト
        scores (list): スコア([float])のリスト
        class_names (list): クラス名([str])のリスト (classes.txtと同じ順番に要素を格納する)
        save_format ([str]): 出力ファイルの保存形式

    Raise:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        FileExistsError: 出力ファイルが既に存在している場合に発生
        ImageReadError: 画像の読み込みに失敗した場合に発生
    """

    logger.info("開始：Detectionアノテーション出力")

    int = 0

    for image_file_path, bboxes, classes, scores in zip(
        image_file_path_list, bboxes_list, classes_list, scores_list
    ):
        # 保存形式がCSV(AzureML)の場合
        if save_format == "csv":
            # CSVファイルパスを取得
            output_file_path_csv = output_file_path_list[0]
            output_file_csv = pathlib.Path(output_file_path_csv)

            # 出力ファイルの親ディレクトリの存在チェック
            if output_file_csv.parent.exists() is False:
                raise FileNotFoundError("ディレクトリが見つかりません：" + str(output_file_csv.parent))
            # 出力ファイルが既に存在している場合は削除
            if output_file_csv.exists() is True:
                output_file_csv.unlink()

            # 出力ファイルパスを元にxmlファイル（一時ファイル）のパスを生成
            filename = os.path.splitext(os.path.basename(image_file_path))[0]
            output_file_path_xml = (
                os.path.dirname(output_file_path_csv) + "/" + filename + ".xml"
            )
        else:
            output_file_path_xml = output_file_path_list[int]
        int += 1

        # Pathオブジェクトの取得
        image_file = pathlib.Path(image_file_path)
        output_file = pathlib.Path(output_file_path_xml)

        # バリデーション
        if image_file.exists() is False:
            raise FileNotFoundError("ファイルが見つかりません：" + str(image_file))
        # 出力ファイルの親ディレクトリの存在チェック
        if output_file.parent.exists() is False:
            raise FileNotFoundError("ディレクトリが見つかりません：" + str(output_file.parent))
        # 出力ファイルが既に存在している場合は削除
        if output_file.exists() is True:
            output_file.unlink()

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
        for bbox, cls, score in zip(bboxes, classes, scores):
            # ラベルを取得
            label = class_names_list[cls]

            node_object = SubElement(node_root, "object")
            node_name = SubElement(node_object, "name")
            node_name.text = label

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
        f = open(output_file, "wb")
        f.write(xml)
        f.close()

    if save_format == "csv":
        # 出力形式がCSVの場合

        # アノテーションリスト作成
        annotation_list = []
        data = {}
        data["annotation_filename"] = output_file
        data["image_filename"] = image_file_path
        annotation_list.append(data)

        # PascalVOL→AzureML形式変換
        preprocessing_tools.convert_annotation_pascalvoc2007_to_azureml(
            annotation_list, str(output_file_csv)
        )

        # xmlを削除
        delete_files = str(output_file_csv.parent) + "/*.xml"
        file_names_list = glob.glob(delete_files)
        for file_name in file_names_list:
            os.remove(file_name)

    logger.info("終了：Detectionアノテーション出力")


def _get_detections_from_pascalvoc2007(annotation_filepath: pathlib.Path):
    # 存在チェック
    if annotation_filepath.exists() is False:
        raise FileNotFoundError("ファイルが見つかりません：" + str(annotation_filepath))

    # xml読み込み
    tree = ElementTree.parse(str(annotation_filepath))
    root = tree.getroot()

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

        # 結果のリストに検出物体の各変数を格納
        detections.append((class_name, xmin, ymin, xmax, ymax))

    return detections


def trim_detection_image(
    result_image_output_dir: pathlib.Path,
    input_image_dir: pathlib.Path,
    detection_result_dir: pathlib.Path,
    is_create_classification_dir: bool,
    input_image_filename: pathlib.Path,
    annotation_filepath: pathlib.Path,
):
    """[summary]

    Args:
        image_dir (pathlib.Path): [description]
        output_dir (pathlib.Path): [description]
        is_create_classification_dir (bool): [description]
        input_image_filepath (pathlib.Path): [description]
        annotation_filepath (pathlib.Path): [description]
    """

    # 入力画像読み込み
    img = cv2.imread(str(input_image_dir / input_image_filename))
    img_height, img_width, _ = img.shape

    # 認識結果取得(ソートはしないXMLに書かれている順番)
    detections = _get_detections_from_pascalvoc2007(annotation_filepath)

    trimmed_image_pathes = []
    for idx, detection in enumerate(detections):
        class_name = detection[0]
        xmin = detection[1]
        ymin = detection[2]
        xmax = detection[3]
        ymax = detection[4]

        # アノテーションがはみ出していた場合の処理
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax >= img_width:
            xmax = img_width - 1
        if ymax >= img_height:
            ymax = img_height - 1

        # イメージをトリム
        trimmed_image = img[ymin:ymax, xmin:xmax]

        # トリム画像の連番は５桁(99999まで)
        if is_create_classification_dir is True:
            save_dir = detection_result_dir / pathlib.Path(class_name)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_filename = (
                detection_result_dir
                / pathlib.Path(class_name)
                / pathlib.Path(
                    pathlib.Path(input_image_filename).stem
                    + "_"
                    + str(idx).zfill(5)
                    + pathlib.Path(input_image_filename).suffix
                )
            )
        else:
            save_filename = detection_result_dir / pathlib.Path(
                pathlib.Path(input_image_filename.stem)
                + "_"
                + str(idx).zfill(5)
                + pathlib.Path(input_image_filename).suffix
            )

        # トリムイメージ一覧のファイル名(ディレクトリは含まない)を戻り値にセット
        trimmed_image_pathes.append(save_filename)

        # 保存
        cv2.imwrite(str(save_filename), trimmed_image)

    return trimmed_image_pathes


def trim_detection_image_from_det(
    result_image_output_dir: pathlib.Path,
    input_image_dir: pathlib.Path,
    detection_result_dir: pathlib.Path,
    is_create_classification_dir: bool,
    input_image_filename: pathlib.Path,
    det_classes,
    det_scores,
    det_bboxes,
    name_list,
):
    """[summary]

    Args:
        image_dir (pathlib.Path): [description]
        output_dir (pathlib.Path): [description]
        is_create_classification_dir (bool): [description]
        input_image_filepath (pathlib.Path): [description]
        annotation_filepath (pathlib.Path): [description]
    """

    # 入力画像読み込み
    img = cv2.imread(str(input_image_dir / input_image_filename))
    img_height, img_width, _ = img.shape

    trimmed_image_pathes = []
    for idx, (class_idx, score, bbox) in enumerate(
        zip(det_classes, det_scores, det_bboxes)
    ):
        class_name = name_list[class_idx]
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]

        # アノテーションがはみ出していた場合の処理
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax >= img_width:
            xmax = img_width - 1
        if ymax >= img_height:
            ymax = img_height - 1

        # イメージをトリム
        trimmed_image = img[ymin:ymax, xmin:xmax]

        # トリム画像の連番は５桁(99999まで)
        if is_create_classification_dir is True:
            save_dir = detection_result_dir / pathlib.Path(class_name)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_filename = (
                detection_result_dir
                / pathlib.Path(class_name)
                / pathlib.Path(
                    pathlib.Path(input_image_filename).stem
                    + "_"
                    + str(idx).zfill(5)
                    + pathlib.Path(input_image_filename).suffix
                )
            )
        else:
            save_filename = detection_result_dir / pathlib.Path(
                pathlib.Path(input_image_filename.stem)
                + "_"
                + str(idx).zfill(5)
                + pathlib.Path(input_image_filename).suffix
            )

        # トリムイメージ一覧のファイル名(ディレクトリは含まない)を戻り値にセット
        trimmed_image_pathes.append(save_filename)

        # 保存
        cv2.imwrite(str(save_filename), trimmed_image)

    return trimmed_image_pathes


def voc_ap_2012(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


# 11-point interpolated average precision
def voc_ap_2008(rec, prec):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    # mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    # mpre.append(0)
    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min() :])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / 11
    # Generating values for the plot
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    return ap


def create_result_list(
    csv_dir,
    all_det_object_json,
    result_list_data_json,
    image_filename,
    true_annotation_path,
    det_classes,
    det_scores,
    det_bboxes,
    trimmed_image_pathes,
    names_list,
):
    # Annotationから結果を取得(ソートはしないXMLに書かれている順番)
    true_detections = _get_detections_from_pascalvoc2007(
        pathlib.Path(true_annotation_path)
    )

    # GroundTruthのBoundingBoxを作る
    os.makedirs(str(pathlib.Path(csv_dir, "groundtruths")), exist_ok=True)
    with open(
        str(
            pathlib.Path(
                csv_dir,
                "groundtruths",
                pathlib.Path(true_annotation_path).with_suffix(".txt").name,
            )
        ),
        "w",
        newline="",
    ) as f2:
        for idx, true_detection in enumerate(true_detections):
            det_class = true_detection[0]
            x1 = true_detection[1]
            y1 = true_detection[2]
            x2 = true_detection[3]
            y2 = true_detection[4]
            f2.write("{} {} {} {} {}\n".format(det_class, x1, y1, x2, y2))

    # アノテーションファイルから正解データ記録用のCSVファイルを生成
    with open(
        str(pathlib.Path(csv_dir) / pathlib.Path("all_annotations.csv")),
        "a",
        encoding="utf_8_sig",
        newline="",
    ) as f:
        for true_detection in true_detections:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)

            writer.writerow(
                [
                    pathlib.Path(true_annotation_path).name,
                    true_detection[0],
                    true_detection[1],
                    true_detection[2],
                    true_detection[3],
                    true_detection[4],
                ]
            )

    # IOUの比較で一致させる
    relation = {}
    relation_reverse = {}

    det_true_class = {}
    true_iou = {}
    # ------------------------------------------------------------------------------
    # アノテーションと推論結果の関連付け
    # ------------------------------------------------------------------------------
    # アノテーション単位
    for i in range(len(true_detections)):
        for true_idx, true_detection in enumerate(true_detections):
            # 既に紐付きの場合は何もしない
            if true_idx in relation:
                continue

            true_bbox = true_detection[1:]
            true_class = true_detection[0]

            # 最低のIOUはゼロ
            max_iou = 0
            for det_idx, det_bbox in enumerate(det_bboxes):
                # 既に相手が決まっている場合は何もしない
                if det_idx in relation_reverse:
                    continue

                # IOUを求める
                iou_value = _iou(true_bbox, det_bbox)

                # 最も高いIOUを記録
                if iou_value > max_iou:
                    relation[true_idx] = det_idx
                    true_iou[true_idx] = iou_value
                    max_iou = iou_value

        # 1つの正解BBOXに複数の推論BBOXが紐ついているかもしれないのでチェック
        deleteFlag = False
        for true_idx, true_detection in enumerate(true_detections):
            # 関連する推論BBOXがなければ何もしない
            if true_idx not in relation:
                continue

            det_idx = relation[true_idx]
            for true_idx2, true_detection2 in enumerate(true_detections):
                # 関連する推論BBOXがなければ何もしない
                if true_idx2 not in relation:
                    continue

                # 同じインデックスは何もしない
                if true_idx == true_idx2:
                    continue

                det_idx2 = relation[true_idx2]

                # 同じ推論ボックスと重なる場合
                # IOUが小さいほうを削除
                if det_idx == det_idx2:
                    # 削除したというフラグを設定
                    deleteFlag = True

                    if true_idx not in true_iou:
                        continue

                    if true_idx2 not in true_iou:
                        continue

                    # 複数のBBOXが見つかったらIOUの低い方を削除
                    if true_iou[true_idx] >= true_iou[true_idx2]:
                        del relation[true_idx2]
                        del true_iou[true_idx2]
                    else:
                        del relation[true_idx]
                        del true_iou[true_idx]
                        break

        # これで１つの正解BBOXの相手が一意に決まることから
        # 逆方向の紐付け(推論BBOXから正解BBOXを引く)を実施
        for true_idx, true_detection in enumerate(true_detections):
            if true_idx in relation:
                relation_reverse[relation[true_idx]] = true_idx

        # 紐付けが決定したためループを抜ける
        if deleteFlag is False:
            break
    # ------------------------------------------------------------------------------
    det_iou = {}
    # JSONの作成(アノテーション毎)
    for true_idx, true_detection in enumerate(true_detections):
        det_idx = -1
        if true_idx in relation:
            det_idx = relation[true_idx]

        true_bbox = true_detection[1:]
        true_class = true_detection[0]

        # 推論BBOXのIOUを決める
        if true_idx in relation:
            det_iou[det_idx] = true_iou[true_idx]
            det_true_class[det_idx] = true_class

        result_obj = {}
        result_obj["drawn_image"] = image_filename
        result_obj["drawn_image_path"] = "../" + image_filename
        result_obj["true_class"] = true_class

        result_obj["true_object_image_path"] = (
            "../trimmed_images/"
            + true_class
            + "/"
            + trimmed_image_pathes[true_idx].name
        )

        if true_idx in relation:
            result_obj["detection_class"] = names_list[det_classes[det_idx]]
            result_obj["detection_score"] = "{:.2f}".format(det_scores[det_idx])
        else:
            result_obj["detection_class"] = ""
            result_obj["detection_score"] = "0.00"

        # 画像毎のデータを格納
        result_list_data_json.append(result_obj)

    # 画像側の一覧
    for det_idx, det_bbox in enumerate(det_bboxes):
        det_object_json = {}

        det_object_json["image"] = image_filename
        det_object_json["class_name"] = names_list[det_classes[det_idx]]
        det_object_json["score"] = det_scores[det_idx]
        det_object_json["true_bbox_count"] = len(true_detections)
        det_object_json["xmin"] = det_bbox[0]
        det_object_json["ymin"] = det_bbox[1]
        det_object_json["xmax"] = det_bbox[2]
        det_object_json["ymax"] = det_bbox[3]

        # アノテーションに対応していればIOUとtrue_classを設定
        # TrueBoxとDetBoxの対応は、1対1のみとするため、紐付けられていないDetectionは再計算
        if det_idx in det_iou:
            det_object_json["iou"] = det_iou[det_idx]
            det_object_json["true_class"] = det_true_class[det_idx]
        else:
            det_object_json["iou"] = 0
            det_object_json["true_class"] = ""
            # 紐付けられていないIOUを計算
            max_iou = 0
            for true_detection_det in true_detections:
                true_bbox_det = true_detection_det[1:]
                true_class_det = true_detection_det[0]

                # IOUを求める
                iou_value = _iou(true_bbox_det, det_bbox)
                # 最も高いIOUを設定する
                if iou_value > max_iou:
                    det_object_json["iou"] = iou_value
                    det_object_json["true_class"] = true_class_det
                    max_iou = iou_value

        if det_object_json["class_name"] == det_object_json["true_class"]:
            det_object_json["correct"] = True
        else:
            det_object_json["correct"] = False

        # 推論オブジェクトを格納
        all_det_object_json.append(det_object_json)


def create_evaluation_page_for_classification(
    evaluation_flag,
    classes_list,
    true_class_list,
    score_list,
    input_image_name_list,
    csv_dir,
    docs_dir,
    classes,
    evaluation_score_thresh,
    classification_true_score_thresh,
):
    # jsonの準備
    # index_data,result_list_data,result_summary_data が必要
    result_data_json = {
        "index_data": {},
        "result_list_data": [],
        "result_summary_data": {},
    }

    # index_dataを作成
    index_data_json = {}

    # 環境変数からprediction_idを取得
    index_data_json["prediction_id"] = ""
    if "AZ_BATCH_JOB_ID" in os.environ:
        index_data_json["prediction_id"] = os.environ["AZ_BATCH_JOB_ID"]

    # 環境変数からdataset_idを取得
    index_data_json["dataset_id"] = ""
    if "AZUREML_EXPERIMENT_ID" in os.environ:
        index_data_json["dataset_id"] = os.environ["AZUREML_EXPERIMENT_ID"]

    #
    index_data_json["evaluation_score_thresh"] = str(evaluation_score_thresh)

    if evaluation_flag is True:
        index_data_json["classification_true_score_thresh"] = str(
            classification_true_score_thresh
        )

    # index_dataをセット
    result_data_json["index_data"] = index_data_json

    result_list_data_json = []

    # 合計値の初期値はゼロ
    class_record = {}
    for class_name in classes:
        class_record[class_name] = {
            "sum_score": 0,
            "count": 0,
            "true_count": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    # マルチラベルのConfusionMatrixを計算
    matrix_array_list = multilabel_confusion_matrix(
        true_class_list, classes_list, labels=classes
    )
    for idx, cls in enumerate(classes):
        # クラス毎の混同行列[TN, FP, FN, TP]
        matrix_record = matrix_array_list[idx].ravel()
        logger.info("matrix<{}>: {}".format(str(cls), str(matrix_record)))
        class_record[cls]["tp"] = matrix_record[3].item()
        class_record[cls]["tn"] = matrix_record[0].item()
        class_record[cls]["fp"] = matrix_record[1].item()
        class_record[cls]["fn"] = matrix_record[2].item()

    # 記録用CSVファイルのヘッダ生成
    with open(
        str(pathlib.Path(csv_dir) / pathlib.Path("all_predictions.csv")),
        "w",
        encoding="utf_8_sig",
        newline="",
    ) as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["image_name", "pred_class", "true_class"])

    for class_name, true_class_name, score, input_image_name in zip(
        classes_list, true_class_list, score_list, input_image_name_list
    ):
        result_data_obj = {}
        result_data_obj["classificated_image"] = pathlib.Path(input_image_name).name
        result_data_obj["classificated_image_path"] = (
            "../" + class_name + "/" + pathlib.Path(input_image_name).name
        )
        if evaluation_flag is True:
            result_data_obj["true_class"] = true_class_name
        result_data_obj["classification_class"] = class_name
        result_data_obj["classification_score"] = "{:.2f}".format(score)
        result_list_data_json.append(result_data_obj)

        # スコアの合計値
        class_record[class_name]["sum_score"] = (
            class_record[class_name]["sum_score"] + score
        )

        # 常にカウントアップ（クラスの数）
        class_record[class_name]["count"] = class_record[class_name]["count"] + 1

        if evaluation_flag is True:
            # 正解画像数をカウントアップ
            class_record[true_class_name]["true_count"] = (
                class_record[true_class_name]["true_count"] + 1
            )

            # 記録用のCSVファイルを生成
            with open(
                str(pathlib.Path(csv_dir) / pathlib.Path("all_predictions.csv")),
                "a",
                encoding="utf_8_sig",
                newline="",
            ) as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(
                    [pathlib.Path(input_image_name).name, class_name, true_class_name]
                )
        else:
            # 予測画像数をカウントアップ
            class_record[class_name]["true_count"] = (
                class_record[class_name]["true_count"] + 1
            )
            # 記録用のCSVファイルを生成
            with open(
                str(pathlib.Path(csv_dir) / pathlib.Path("all_predictions.csv")),
                "a",
                encoding="utf_8_sig",
                newline="",
            ) as f:
                # 正解クラスは空文字""とし、CSVファイルとして成立させる
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow([pathlib.Path(input_image_name).name, class_name, ""])

    class_summary_list_json = []
    for class_name in classes:
        record = class_record[class_name]

        class_summary_obj = {"class_name": class_name, "avg_score_pred": 0}

        if record["count"] != 0:
            class_summary_obj["avg_score_pred"] = "{:.2f}".format(
                record["sum_score"] / record["count"]
            )
        else:
            class_summary_obj["avg_score_pred"] = "{:.2f}".format(0)

        if evaluation_flag is True:
            class_summary_obj["image_count_true"] = record[
                "true_count"
            ]  # 正解クラスの画像数に変更する
            class_summary_obj["tp_cls"] = record["tp"]
            class_summary_obj["tn_cls"] = record["tn"]
            class_summary_obj["fp_cls"] = record["fp"]
            class_summary_obj["fn_cls"] = record["fn"]

            class_summary_obj["precision_cls"] = 0
            class_summary_obj["recall_cls"] = 0
            class_summary_obj["accuracy_cls"] = 0
            class_summary_obj["specificity_cls"] = 0
            class_summary_obj["f_measure_cls"] = 0

            if record["tp"] + record["fp"] != 0:
                class_summary_obj["precision_cls"] = (record["tp"]) / (
                    record["tp"] + record["fp"]
                )

            if record["tp"] + record["fn"] != 0:
                class_summary_obj["recall_cls"] = (record["tp"]) / (
                    record["tp"] + record["fn"]
                )

            if record["tp"] + record["tn"] + record["fp"] + record["fn"] != 0:
                class_summary_obj["accuracy_cls"] = (record["tp"] + record["tn"]) / (
                    record["tp"] + record["tn"] + record["fp"] + record["fn"]
                )

            if record["fp"] + record["tn"] != 0:
                class_summary_obj["specificity_cls"] = (record["tn"]) / (
                    record["fp"] + record["tn"]
                )

            if (
                class_summary_obj["precision_cls"] + class_summary_obj["recall_cls"]
                != 0
            ):
                class_summary_obj["f_measure_cls"] = (
                    2
                    * class_summary_obj["precision_cls"]
                    * class_summary_obj["recall_cls"]
                ) / (
                    class_summary_obj["precision_cls"] + class_summary_obj["recall_cls"]
                )

            class_summary_obj["precision_cls"] = "{:.2f}".format(
                class_summary_obj["precision_cls"]
            )
            class_summary_obj["recall_cls"] = "{:.2f}".format(
                class_summary_obj["recall_cls"]
            )
            class_summary_obj["accuracy_cls"] = "{:.2f}".format(
                class_summary_obj["accuracy_cls"]
            )
            class_summary_obj["specificity_cls"] = "{:.2f}".format(
                class_summary_obj["specificity_cls"]
            )
            class_summary_obj["f_measure_cls"] = "{:.2f}".format(
                class_summary_obj["f_measure_cls"]
            )
        else:
            class_summary_obj["image_count_pred"] = record["count"]

        class_summary_list_json.append(class_summary_obj)

    # result_summary_dataの作成
    result_summary_data_json = {"class_summary_list": [], "average": {}}
    result_summary_data_json["class_summary_list"] = class_summary_list_json
    average_json = {}
    result_summary_data_json["average"] = average_json

    result_data_json["result_list_data"] = result_list_data_json
    result_data_json["result_summary_data"] = result_summary_data_json

    # テンプレート準備
    if evaluation_flag is True:
        gzo_fw_common_template_path = pathlib.Path(
            __file__
        ).parent.parent.parent / pathlib.Path(
            "gzo_fw_common/templates/classification_evaluation"
        )
    else:
        gzo_fw_common_template_path = pathlib.Path(
            __file__
        ).parent.parent.parent / pathlib.Path(
            "gzo_fw_common/templates/classification_prediction_only"
        )

    gzo_fw_common_template_docs = gzo_fw_common_template_path / pathlib.Path("docs")

    # テンプレートのコピー
    shutil.rmtree(docs_dir, ignore_errors=True)
    shutil.copytree(gzo_fw_common_template_docs, docs_dir)

    gzo_fw_common_template_path = pathlib.Path(
        __file__
    ).parent.parent.parent / pathlib.Path("gzo_fw_common/templates")
    # -------------------------------------------------------------------------------
    # テンプレートへの書き込み
    tpl = ""
    with open(
        gzo_fw_common_template_path / pathlib.Path("result_data.js.template"),
        "r",
        encoding=ENCODING_UTF8,
    ) as f_result_list_js_template:
        tpl = f_result_list_js_template.read()

    result_list_template = string.Template(tpl)
    result_data = json.dumps(result_data_json, indent=4)
    # f_result_list_js_template.close()

    # js出力
    with open(
        str(docs_dir / pathlib.Path("results/result_data.js")),
        "w",
        encoding=ENCODING_UTF8,
    ) as f_result_list:
        f_result_list.write(result_list_template.substitute(result_data=result_data))


def create_evaluation_page_for_detection(
    evaluation_flag,
    bboxes_list,
    classes_list,
    scores_list,
    input_image_name_list,
    true_annotation_path_list,
    output_dir,
    docs_dir,
    trrimed_image_output_dir,
    names_list,
    evaluation_score_thresh,
    detection_true_iou_thresh,
    export_annotation_data_format: str = "xml",
):
    """Detection精度評価画面出力

    Args:
        evaluation_flag (bool): 精度評価の有無
        bboxes_list ([[[int, int, int, int]]]): 推論bbox
        classes_list ([[str]]): 推論class
        scores_list ([[float]]): 推論score
        input_image_name_list ([type]): 画像名一覧
        true_annotation_path_list ([[str]]):アノテーションのパス一覧
        drawn_image_name_list ([type]): 出力画像一覧
        docs_dir_path ([type]): [description]
        drawn_image_output_path ([type]): [description]
        result_annotation_output_path ([type]): [description]
        input_image_output_path ([type]): [description]
        input_annotation_output_path ([type]): [description]
        docs_template_dir_path ([type]): [description]
        export_annotation_data_format (str): アノテーション出力フォーマット(csv: azureML, xml: pascalVOC)
    """
    result_image_output_dir = output_dir
    result_annotation_output_dir = output_dir

    # jsonの準備
    # index_data,result_list_data,result_summary_data が必要
    result_data_json = {
        "index_data": {},
        "result_list_data": [],
        "result_summary_data": {},
    }

    # index_dataを作成
    index_data_json = {}

    # 環境変数からprediction_idを取得
    index_data_json["prediction_id"] = ""
    if "AZ_BATCH_JOB_ID" in os.environ:
        index_data_json["prediction_id"] = os.environ["AZ_BATCH_JOB_ID"]

    # 環境変数からdataset_idを取得
    index_data_json["dataset_id"] = ""
    if "AZUREML_EXPERIMENT_ID" in os.environ:
        index_data_json["dataset_id"] = os.environ["AZUREML_EXPERIMENT_ID"]

    index_data_json["evaluation_score_thresh"] = str(evaluation_score_thresh)

    if evaluation_flag is True:
        index_data_json["detection_true_iou_thresh"] = str(detection_true_iou_thresh)

    # index_dataをセット
    result_data_json["index_data"] = index_data_json

    result_list_data_json = []
    class_summary_list_json = []

    # 全ての推論結果
    all_det_object_json = []

    # CSVディレクトリを作成
    csv_dir = pathlib.Path(result_image_output_dir) / pathlib.Path("csv")
    if csv_dir.exists() is not True:
        os.makedirs(csv_dir)

    # 評価フラグによって作成するCSVファイルを変える
    if evaluation_flag is True:
        # アノテーションファイルから正解データ記録用のCSVファイルを生成(ヘッダのみ)
        with open(
            str(pathlib.Path(csv_dir) / pathlib.Path("all_predictions.csv")),
            "w",
            encoding="utf_8_sig",
            newline="",
        ) as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(
                [
                    "ファイル名",
                    "正解クラス名",
                    "推論クラス名",
                    "分類成功",
                    "スコア",
                    "IOU",
                    "xmin",
                    "ymin",
                    "xmax",
                    "ymax",
                ]
            )

        # アノテーションファイルから正解データ記録用のCSVファイルを生成(ヘッダのみ)
        with open(
            str(pathlib.Path(csv_dir) / pathlib.Path("all_annotations.csv")),
            "w",
            encoding="utf_8_sig",
            newline="",
        ) as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["ファイル名", "クラス名", "xmin", "ymin", "xmax", "ymax"])

    else:
        # アノテーションファイルから正解データ記録用のCSVファイルを生成(ヘッダのみ)
        with open(
            str(pathlib.Path(csv_dir) / pathlib.Path("all_predictions.csv")),
            "w",
            encoding="utf_8_sig",
            newline="",
        ) as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["ファイル名", "推論クラス名", "スコア", "xmin", "ymin", "xmax", "ymax"])

    shutil.rmtree(str(pathlib.Path(csv_dir, "detections")), ignore_errors=True)
    result_annotation_filename_list = []
    for idx, image in enumerate(input_image_name_list):
        # ファイル名をベース名と拡張子に分ける
        # example.txt -> example<str> .txt<str>
        basename = pathlib.Path(image).stem
        suf = pathlib.Path(image).suffix

        det_bboxes = bboxes_list[idx]
        det_classes = classes_list[idx]
        det_scores = scores_list[idx]

        # アノテーション結果のファイル名を作成
        result_annotation_filename_list.append(
            str(pathlib.Path(result_annotation_output_dir, basename + ".xml"))
        )

        # 認識結果画像を生成 --------------------------------------------------------------------------------------
        drawn_image = draw_detection_results_on_image(
            image, det_bboxes, det_classes, det_scores, names_list
        )

        # 拡張子の前に「result」を付与して認識結果画像のファイル名とする
        result_image_filename = "{}_result{}".format(basename, suf)
        result_image_filename = "{}/{}".format(
            result_image_output_dir, result_image_filename
        )

        # 認識結果画像を保存する
        cv2.imwrite(result_image_filename, drawn_image)
        # --------------------------------------------------------------------------------------------------------

        # DetectionのBoundingBoxを作る
        os.makedirs(str(pathlib.Path(csv_dir, "detections")), exist_ok=True)
        with open(
            str(pathlib.Path(csv_dir, "detections", basename + ".txt")), "w", newline=""
        ) as f1:
            for det_class, det_bbox, det_score in zip(
                det_classes, det_bboxes, det_scores
            ):
                f1.write(
                    "{} {} {} {} {} {}\n".format(
                        names_list[int(det_class)],
                        det_score,
                        det_bbox[0],
                        det_bbox[1],
                        det_bbox[2],
                        det_bbox[3],
                    )
                )

        if evaluation_flag is True:
            # アノテーションファイルを取得
            true_annotation_path = true_annotation_path_list[idx]

            # トリム画像を保存
            imagePath = pathlib.Path(image)
            trimmed_image_pathes = trim_detection_image(
                result_image_output_dir,
                imagePath.parent,
                trrimed_image_output_dir,
                True,
                imagePath.name,
                pathlib.Path(true_annotation_path),
            )

            # 渡す対象は1ファイル単位(画像単位に処理する)
            create_result_list(
                csv_dir,
                all_det_object_json,
                result_list_data_json,
                pathlib.Path(result_image_filename).name,
                true_annotation_path,
                det_classes,
                det_scores,
                det_bboxes,
                trimmed_image_pathes,
                names_list,
            )
        else:
            # トリム画像を保存
            imagePath = pathlib.Path(image)
            trimmed_image_pathes = trim_detection_image_from_det(
                result_image_output_dir,
                imagePath.parent,
                trrimed_image_output_dir,
                True,
                imagePath.name,
                det_classes,
                det_scores,
                det_bboxes,
                names_list,
            )

            # Detection推論結果をアノテーションデータ(PascalVOC2007形式)として生成する
            result_annotation_filename = "{}.xml".format(basename)
            result_annotation_filename = "{}/{}".format(
                result_annotation_output_dir, result_annotation_filename
            )

            for trim_idx, trimmed_image_pathe in enumerate(trimmed_image_pathes):
                result_obj = {}
                result_obj["drawn_image"] = pathlib.Path(result_image_filename).name
                result_obj["drawn_image_path"] = (
                    "../" + pathlib.Path(result_image_filename).name
                )
                result_obj["detection_class"] = names_list[det_classes[trim_idx]]
                result_obj["detection_score"] = "{:.2f}".format(det_scores[trim_idx])
                result_obj["predicted_object_image_path"] = (
                    "../trimmed_images/"
                    + names_list[det_classes[trim_idx]]
                    + "/"
                    + trimmed_image_pathe.name
                )
                result_list_data_json.append(result_obj)

            for det_idx, det_bbox in enumerate(det_bboxes):
                det_object_json = {}
                det_object_json["image"] = pathlib.Path(image).name
                det_object_json["class_name"] = names_list[det_classes[det_idx]]
                det_object_json["score"] = det_scores[det_idx]
                det_object_json["xmin"] = det_bbox[0]
                det_object_json["ymin"] = det_bbox[1]
                det_object_json["xmax"] = det_bbox[2]
                det_object_json["ymax"] = det_bbox[3]

                # 推論オブジェクトを格納
                all_det_object_json.append(det_object_json)

    # Detection推論結果をアノテーションデータ(PascalVOC2007形式/AzureML形式)として生成する
    export_detection_results(
        input_image_name_list,
        result_annotation_filename_list,
        bboxes_list,
        classes_list,
        scores_list,
        names_list,
        export_annotation_data_format,
    )

    if evaluation_flag is True:
        # アノテーションファイルから正解データ記録用のCSVファイルを生成
        with open(
            str(pathlib.Path(csv_dir) / pathlib.Path("all_predictions.csv")),
            "a",
            encoding="utf_8_sig",
            newline="",
        ) as f:
            for all_det_obj in all_det_object_json:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)

                writer.writerow(
                    [
                        all_det_obj["image"],
                        all_det_obj["true_class"],
                        all_det_obj["class_name"],
                        all_det_obj["correct"],
                        all_det_obj["score"],
                        all_det_obj["iou"],
                        all_det_obj["xmin"],
                        all_det_obj["ymin"],
                        all_det_obj["xmax"],
                        all_det_obj["ymax"],
                    ]
                )
    else:
        with open(
            str(pathlib.Path(csv_dir) / pathlib.Path("all_predictions.csv")),
            "a",
            encoding="utf_8_sig",
            newline="",
        ) as f:
            for all_det_obj in all_det_object_json:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)

                writer.writerow(
                    [
                        all_det_obj["image"],
                        all_det_obj["class_name"],
                        all_det_obj["score"],
                        all_det_obj["xmin"],
                        all_det_obj["ymin"],
                        all_det_obj["xmax"],
                        all_det_obj["ymax"],
                    ]
                )

    class_summary_list_json = []
    average_json = {}

    ##############################################
    # クラス毎のサマリーを作成
    ##############################################
    for class_name in names_list:
        if evaluation_flag is True:
            # あるクラスの正解bbox全体に対して、IoU>0.5である推論bboxの個数（正解は問わない）
            all_count_good_iou = len(
                [
                    x["true_class"]
                    for x in all_det_object_json
                    if x["true_class"] == class_name
                    and x["iou"] >= detection_true_iou_thresh
                ]
            )

            # あるクラスの正解bbox全体に対して、IoU>0.5である推論bboxの個数（クラスも正解した場合)
            all_success_count_good_iou = len(
                [
                    x["class_name"]
                    for x in all_det_object_json
                    if x["class_name"] == class_name
                    and x["true_class"] == class_name
                    and x["iou"] >= detection_true_iou_thresh
                ]
            )

            # あるクラスの正解bboxの総数(アノテーションから取得しているので推論は関係ない)
            all_true_count = len(
                [
                    x["true_class"]
                    for x in result_list_data_json
                    if x["true_class"] == class_name
                ]
            )

            # あるクラスの正解bbox全体に対して、IoU>0.5である推論bboxのスコア
            all_sucess_scores = [
                x["score"]
                for x in all_det_object_json
                if x["class_name"] == class_name
                and x["iou"] >= detection_true_iou_thresh
            ]

            # スコア(正解は問わない）
            all_scores = [
                x["score"] for x in all_det_object_json if x["class_name"] == class_name
            ]

            # IOUとスコアを求める
            all_ious = [
                x["iou"] for x in all_det_object_json if x["class_name"] == class_name
            ]

            # TP,TN,FP,FNを求める
            all_tp = all_success_count_good_iou
            all_fp = all_count_good_iou - all_success_count_good_iou
            all_fn = all_true_count - all_success_count_good_iou

            # APを求めるためのリスト(スコアで降順)
            all_list_for_ap = [
                {
                    "image": x["image"],
                    "iou": x["iou"],
                    "score": x["score"],
                    "true_class": x["true_class"],
                    "true_bbox_count": x["true_bbox_count"],
                }
                for x in all_det_object_json
                if x["class_name"] == class_name
            ]

            all_list_for_ap = sorted(
                all_list_for_ap, key=lambda x: x["score"], reverse=True
            )

            # 初期化
            class_summary_obj = {
                "class_name": class_name,
                "avg_score_pred": 0.00,
                "object_count_true": 0,
                "detection_rate": 0.00,
                "classification_rate": 0.00,
                "classification_rate": 0.00,
                "avg_iou": 0.00,
                "ap": 0.00,
            }

            # 推論クラスごとの平均スコア
            if len(all_sucess_scores) > 0:
                class_summary_obj["avg_score_pred"] = np.average(
                    np.array(all_sucess_scores)
                )

            # オブジェクト数
            class_summary_obj["object_count_true"] = all_true_count

            # 検出成功率
            # (あるクラスの正解bbox全体に対して、IoU>0.5である推論bboxの個数) / (あるクラスの正解bboxの総数)
            # クラスが一致していない推論bboxも分子に含める
            if all_true_count != 0:
                class_summary_obj["detection_rate"] = (
                    all_count_good_iou / all_true_count
                )

            # 分類成功率
            # あるクラスの正解bbox全体に対して、クラスが一致かつIoU>0.5である推論bboxの個数) / (あるクラスの正解bbox全体に対して、IoU>0.5である推論bboxの個数
            # クラスが一致していない推論bboxも分母に含める
            if all_count_good_iou != 0:
                class_summary_obj["classification_rate"] = (
                    all_success_count_good_iou / all_count_good_iou
                )

            class_summary_obj["tp_det"] = all_tp
            class_summary_obj["tn_det"] = all_tp
            class_summary_obj["fp_det"] = all_fp
            class_summary_obj["fn_det"] = all_fn

            # 検出の正答率
            class_summary_obj["accuracy_det"] = 0
            if all_tp + all_fp + all_fn != 0:
                class_summary_obj["accuracy_det"] = all_tp / (all_tp + all_fp + all_fn)

            # 検出の適合率
            precision_det = 0
            if all_tp + all_fp != 0:
                precision_det = all_tp / (all_tp + all_fp)

            # 検出の再現率
            recall_det = 0
            if all_tp + all_fn != 0:
                recall_det = all_tp / (all_tp + all_fn)

            # 検出のF値
            f_measure_det = 0
            if precision_det + recall_det != 0:
                f_measure_det = (2 * precision_det * recall_det) / (
                    precision_det + recall_det
                )

            class_summary_obj["precision_det"] = precision_det
            class_summary_obj["recall_det"] = recall_det
            class_summary_obj["f_measure_det"] = f_measure_det

            if len(all_ious) > 0:
                class_summary_obj["avg_iou"] = np.average(np.array(all_ious))

            # APを求める
            true_count = 0
            ap_precs = []
            ap_recs = []
            recstr = []
            precstr = []
            if all_true_count > 0:
                for idx, obj_for_ap in enumerate(all_list_for_ap):
                    # 正解数を1増やす
                    if (
                        obj_for_ap["iou"] >= detection_true_iou_thresh
                        and class_name == obj_for_ap["true_class"]
                    ):
                        true_count = true_count + 1

                    # PrecisionとRecallを算出
                    ap_precs.append(true_count / (idx + 1))
                    ap_recs.append(true_count / all_true_count)

                    recstr.append("rec=" + str(true_count) + "/" + str(all_true_count))
                    precstr.append("prec=" + str(true_count) + "/" + str((idx + 1)))

                class_summary_obj["ap"] = voc_ap_2008(ap_recs, ap_precs)
            else:
                class_summary_obj["ap"] = 0

            class_summary_list_json.append(class_summary_obj)

        else:
            # スコア
            all_scores = [
                x["score"] for x in all_det_object_json if x["class_name"] == class_name
            ]
            all_score_sum = np.sum(np.array(all_scores))
            obj_count = len(all_scores)

            class_summary_obj = {}
            class_summary_obj["class_name"] = class_name
            class_summary_obj["avg_score_pred"] = 0
            if obj_count != 0:
                class_summary_obj["avg_score_pred"] = all_score_sum / obj_count
            class_summary_obj["avg_score_pred"] = "{:.2f}".format(
                class_summary_obj["avg_score_pred"]
            )
            class_summary_obj["object_count_pred"] = obj_count
            class_summary_list_json.append(class_summary_obj)

    # 平均値を求める
    if evaluation_flag is True:
        average_json = {}
        object_count_true_sum = 0
        avg_score_pred_sum = 0
        classification_rate_sum = 0

        ap_sum = np.sum(np.array([x["ap"] for x in class_summary_list_json]))

        # ----------------------------------------------------------------------------
        # アノテーションから平均を算出
        # 正解クラスにあるクラスのみカウント
        exist_class_count = 0
        detection_rate_sum = 0
        for class_summary_json in class_summary_list_json:
            if class_summary_json["object_count_true"] > 0:
                exist_class_count = exist_class_count + 1
                object_count_true_sum = (
                    object_count_true_sum + class_summary_json["object_count_true"]
                )
                avg_score_pred_sum = (
                    avg_score_pred_sum
                    + class_summary_json["avg_score_pred"]
                    * class_summary_json["object_count_true"]
                )
                detection_rate_sum = (
                    detection_rate_sum
                    + class_summary_json["detection_rate"]
                    * class_summary_json["object_count_true"]
                )
                classification_rate_sum = (
                    classification_rate_sum
                    + class_summary_json["classification_rate"]
                    * class_summary_json["object_count_true"]
                )

        all_true_count = len([x["true_class"] for x in all_det_object_json])
        all_success_count_good_iou = len(
            [
                x["true_class"]
                for x in all_det_object_json
                if x["iou"] >= detection_true_iou_thresh
            ]
        )

        # クラス種類での平均
        if exist_class_count != 0:
            average_json["map"] = "{:.2f}".format(ap_sum / exist_class_count)

        # 合計値のみから算出

        # ------------------------------------------------------------------------------
        # フォーマット調整
        # ------------------------------------------------------------------------------
        for class_summary_obj in class_summary_list_json:
            class_summary_obj["avg_score_pred"] = "{:.2f}".format(
                class_summary_obj["avg_score_pred"]
            )
            class_summary_obj["detection_rate"] = "{:.2f}".format(
                class_summary_obj["detection_rate"]
            )
            class_summary_obj["classification_rate"] = "{:.2f}".format(
                class_summary_obj["classification_rate"]
            )
            class_summary_obj["accuracy_det"] = "{:.2f}".format(
                class_summary_obj["accuracy_det"]
            )
            class_summary_obj["precision_det"] = "{:.2f}".format(
                class_summary_obj["precision_det"]
            )
            class_summary_obj["recall_det"] = "{:.2f}".format(
                class_summary_obj["recall_det"]
            )
            class_summary_obj["f_measure_det"] = "{:.2f}".format(
                class_summary_obj["f_measure_det"]
            )
            class_summary_obj["avg_iou"] = "{:.2f}".format(class_summary_obj["avg_iou"])
            class_summary_obj["ap"] = "{:.2f}".format(class_summary_obj["ap"])
    else:
        average_json = {"map": 0.00}
        all_scores = [x["score"] for x in all_det_object_json]

    # result_summary_dataの作成
    result_summary_data_json = {"class_summary_list": [], "average": {}}
    result_summary_data_json["class_summary_list"] = class_summary_list_json
    result_summary_data_json["average"] = average_json

    result_data_json["result_list_data"] = result_list_data_json
    result_data_json["result_summary_data"] = result_summary_data_json

    # テンプレート準備
    if evaluation_flag is True:
        gzo_fw_common_template_path = pathlib.Path(
            __file__
        ).parent.parent.parent / pathlib.Path(
            "gzo_fw_common/templates/detection_evaluation"
        )
    else:
        gzo_fw_common_template_path = pathlib.Path(
            __file__
        ).parent.parent.parent / pathlib.Path(
            "gzo_fw_common/templates/detection_prediction_only"
        )

    gzo_fw_common_template_docs = gzo_fw_common_template_path / pathlib.Path("docs")

    # テンプレートのコピー
    for i in range(30):
        shutil.rmtree(docs_dir, ignore_errors=True)
        if pathlib.Path(docs_dir).exists() is False:
            break
        time.sleep(1)
    shutil.copytree(gzo_fw_common_template_docs, docs_dir)

    gzo_fw_common_template_path = pathlib.Path(
        __file__
    ).parent.parent.parent / pathlib.Path("gzo_fw_common/templates")
    # -------------------------------------------------------------------------------
    # テンプレートへの書き込み
    tpl = ""
    with open(
        gzo_fw_common_template_path / pathlib.Path("result_data.js.template"),
        "r",
        encoding=ENCODING_UTF8,
    ) as f_result_list_js_template:
        tpl = f_result_list_js_template.read()

    result_list_template = string.Template(tpl)
    result_data = json.dumps(result_data_json, indent=4)

    # js出力
    with open(
        str(docs_dir / pathlib.Path("results/result_data.js")),
        "w",
        encoding=ENCODING_UTF8,
    ) as f_result_list:
        f_result_list.write(result_list_template.substitute(result_data=result_data))


def create_evaluation_page_for_segmentation(
    evaluation_flag: bool,
    classes: list,
    predict_classes_list: list,
    true_class_list: list,
    input_image_name_list: list,
    csv_dir: str,
    docs_dir: str,
    segmentation_iou_thresh: float,
    predict_sec_list: list,
):
    """Segmentation精度評価画面出力

    Args:
        evaluation_flag (bool): 精度評価の有無
        classes: 推論クラス名一覧
        predict_classes_list ([str]): 推論結果画像一覧
        true_class_list ([str]): 正解（アノテーション）画像一覧
        input_image_name_list ([str]): 入力画像一覧
        csv_dir (str): CSV出力先パス
        docs_dir (str): 精度評価画面ファイル群出力先パス
        segmentation_iou_thresh (float): IoU閾値
        predict_sec_list: 推論時間(list[float])
    """

    # 記録用CSVファイルのヘッダ生成
    with open(
        str(pathlib.Path(csv_dir) / pathlib.Path("all_predictions.csv")),
        "w",
        encoding="utf_8_sig",
        newline="",
    ) as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(
            [
                "filename",
                "label",
                "IoU",
                "TP",
                "FP",
                "FN",
                "precision",
                "recall",
                "area",
                "predict_sec",
            ]
        )

    # jsonの準備
    # index_data,result_list_data,result_summary_data が必要
    result_data_json = {
        "index_data": {},
        "result_list_data": [],
        "result_summary_data": {},
    }

    # index_dataを作成
    index_data_json = {}

    # 環境変数からprediction_idを取得
    index_data_json["prediction_id"] = ""
    if "AZ_BATCH_JOB_ID" in os.environ:
        index_data_json["prediction_id"] = os.environ["AZ_BATCH_JOB_ID"]

    # 環境変数からdataset_idを取得
    index_data_json["dataset_id"] = ""
    if "AZUREML_EXPERIMENT_ID" in os.environ:
        index_data_json["dataset_id"] = os.environ["AZUREML_EXPERIMENT_ID"]

    # IoUの閾値
    if evaluation_flag is True:
        index_data_json["segmentation_iou_thresh"] = str(segmentation_iou_thresh)

    # クラス数
    index_data_json["class_count"] = len(classes) - 1

    # index_dataをセット
    result_data_json["index_data"] = index_data_json

    # 合計値の初期値はゼロ
    class_record = {}

    # アノテーション画像存在チェックフラグ(空文字リストの場合はFalse)
    ann_flg = any(true_class_list)

    # CSV出力先が存在しない場合
    if not pathlib.Path(csv_dir).exists():
        # 存在チェックエラー
        raise FileNotFoundError("csv出力先が見つかりません：" + str(csv_dir))

    # 精度評価画面ファイル群出力先パスが存在しない場合
    if not pathlib.Path(docs_dir).exists():
        # 存在チェックエラー
        raise FileNotFoundError("精度評価画面の出力先が見つかりません：" + str(docs_dir))

    result_list_data_json = []

    # 正解画像枚数
    img_count = 0

    # 合計計算用
    sum_record = {}
    for class_name in classes:
        sum_record[class_name] = {
            "sum_iou": 0,
            "count": 0,
            "true_count": 0,
            "sum_true_area": 0,
            "sum_tp": 0,
            "sum_fp": 0,
            "sum_fn": 0,
            "sum_presision": 0,
            "sum_recall": 0,
        }

    for input_file_name, predict_file_name, true_file_name, predict_sec in zip(
        input_image_name_list, predict_classes_list, true_class_list, predict_sec_list
    ):
        # 入力画像が存在しない場合
        if len(input_file_name) == 0 or not pathlib.Path(input_file_name).exists():
            # 存在チェックエラー
            raise FileNotFoundError("入力画像が見つかりません：" + str(input_file_name))

        # 推論結果画像が存在しない場合
        if len(predict_file_name) == 0 or not pathlib.Path(predict_file_name).exists():
            # 存在チェックエラー
            raise FileNotFoundError("推論結果画像が見つかりません：" + str(predict_file_name))

        # アノテーション画像
        if ann_flg and pathlib.Path(true_file_name).exists:
            ann_image = Image.open(true_file_name).convert("RGB")

            # 画像枚数カウント
            img_count = img_count + 1

        # 推論結果画像
        pre_image = Image.open(predict_file_name).convert("RGB")

        for class_name in classes:

            # アノテーションと推論結果の色(インデックスカラー)
            ann_color = pred_color = classes[class_name]

            # 背景はスキップ
            if ann_color == [0, 0, 0]:
                continue
            class_record[class_name] = {
                "sum_iou": 0,
                "count": 0,
                "true_count": 0,
                "true_area": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
            }

            # IoUの計算
            if ann_flg and pathlib.Path(true_file_name).exists:
                tp, fp, fn, iou = calc_iou(ann_image, pre_image, ann_color, pred_color)
            else:
                tp = 0
                fp = 0
                fn = 0
                iou = 0
            class_record[class_name]["tp"] = tp
            class_record[class_name]["fp"] = fp
            class_record[class_name]["fn"] = fn
            class_record[class_name]["iou"] = iou

            # result
            result_data_obj = {}
            result_data_obj["segmentation_image_path"] = str(
                pathlib.Path(
                    "../jpg/input", pathlib.Path(input_file_name).name
                ).as_posix()
            )

            if ann_flg is True:
                result_data_obj["true_class_path"] = str(
                    pathlib.Path(
                        "..", "jpg", "true", pathlib.Path(true_file_name).name
                    ).as_posix()
                )

            result_data_obj["predict_class_path"] = str(
                pathlib.Path(
                    "..", "jpg", "predict", pathlib.Path(predict_file_name).name
                ).as_posix()
            )

            # RGBからHTMLカラーコード取得
            color_code_r = "{:02x}".format(ann_color[0])
            color_code_g = "{:02x}".format(ann_color[1])
            color_code_b = "{:02x}".format(ann_color[2])
            color_code = "#{}{}{}".format(color_code_r, color_code_g, color_code_b)

            result_data_obj["class_color_code"] = color_code
            result_data_obj["class_name"] = class_name
            result_data_obj["segmentation_iou"] = iou
            result_data_obj["tp"] = tp
            result_data_obj["fp"] = fp
            result_data_obj["fn"] = fn

            # アノテーション面積
            true_area = tp + fn

            # 適合率
            if (tp + fp) != 0:
                presision = tp / (tp + fp)
            else:
                presision = 0
            result_data_obj["presision"] = presision

            # 再現率
            if (tp + fn) != 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
            result_data_obj["recall"] = recall
            result_list_data_json.append(result_data_obj)

            # IoU合計
            sum_record[class_name]["sum_iou"] = sum_record[class_name]["sum_iou"] + iou

            # アノテーション面積合計
            sum_record[class_name]["sum_true_area"] = (
                sum_record[class_name]["sum_true_area"] + true_area
            )

            # TP合計
            sum_record[class_name]["sum_tp"] = sum_record[class_name]["sum_tp"] + tp

            # FP合計
            sum_record[class_name]["sum_fp"] = sum_record[class_name]["sum_fp"] + fp

            # FN合計
            sum_record[class_name]["sum_fn"] = sum_record[class_name]["sum_fn"] + fn

            # 適合率合計
            sum_record[class_name]["sum_presision"] = (
                sum_record[class_name]["sum_presision"] + presision
            )

            # 再現率合計
            sum_record[class_name]["sum_recall"] = (
                sum_record[class_name]["sum_recall"] + recall
            )

            if ann_flg is True:
                # 記録用のCSVファイルを生成
                with open(
                    str(pathlib.Path(csv_dir) / pathlib.Path("all_predictions.csv")),
                    "a",
                    encoding="utf_8_sig",
                    newline="",
                ) as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                    writer.writerow(
                        [
                            predict_file_name,
                            class_name,
                            iou,
                            tp,
                            fp,
                            fn,
                            presision,
                            recall,
                            true_area,
                            predict_sec,
                        ]
                    )
            else:
                # 記録用のCSVファイルを生成
                with open(
                    str(pathlib.Path(csv_dir) / pathlib.Path("all_predictions.csv")),
                    "a",
                    encoding="utf_8_sig",
                    newline="",
                ) as f:
                    # 正解クラスは空文字""とし、CSVファイルとして成立させる
                    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                    writer.writerow(
                        [class_name, input_file_name, predict_file_name, ""]
                    )

    class_summary_list_json = []
    avg_row_iou = 0
    avg_row_true_count = 0
    avg_row_true_area = 0
    avg_row_tp = 0
    avg_row_fp = 0
    avg_row_fn = 0
    avg_row_presision = 0
    avg_row_recall = 0
    for class_name in classes:
        # 背景はスキップ
        if class_name == "_background_":
            continue

        record = sum_record[class_name]

        class_summary_obj = {"class_name": class_name, "avg_iou_pred": 0}

        if img_count != 0:
            # IoU平均
            class_summary_obj["avg_iou_pred"] = record["sum_iou"] / img_count
        else:
            class_summary_obj["avg_iou_pred"] = 0

        # アノテーション画像枚数
        class_summary_obj["true_count"] = img_count

        # アノテーション面積(TP + FN)
        class_summary_obj["true_area"] = record["sum_true_area"]

        # TP（合計）
        class_summary_obj["cls_tp"] = record["sum_tp"]

        # FP（合計）
        class_summary_obj["cls_fp"] = record["sum_fp"]

        # FN（合計）
        class_summary_obj["cls_fn"] = record["sum_fn"]

        # 適合率平均
        presision_denominator = record["sum_tp"] + record["sum_fp"]
        if presision_denominator != 0:
            class_summary_obj["avg_presision"] = (
                record["sum_tp"] / presision_denominator
            )
        else:
            class_summary_obj["avg_presision"] = 0

        # 再現率平均
        recall_denominator = record["sum_tp"] + record["sum_fn"]
        if recall_denominator != 0:
            class_summary_obj["avg_recall"] = record["sum_tp"] / (
                record["sum_tp"] + record["sum_fn"]
            )
        else:
            class_summary_obj["avg_recall"] = 0

        # IoU平均（平均行）
        avg_row_iou = avg_row_iou + float(class_summary_obj["avg_iou_pred"])

        # アノテーション画像枚数（平均行）
        avg_row_true_count = avg_row_true_count + float(class_summary_obj["true_count"])

        # アノテーション面積（平均行）
        avg_row_true_area = avg_row_true_area + float(class_summary_obj["true_area"])

        # TP（平均行）
        avg_row_tp = avg_row_tp + float(class_summary_obj["cls_tp"])

        # FP（平均行）
        avg_row_fp = avg_row_fp + class_summary_obj["cls_fp"]

        # FN（平均行）
        avg_row_fn = avg_row_fn + class_summary_obj["cls_fn"]

        # 適合率平均（平均行）
        avg_row_presision = avg_row_presision + class_summary_obj["avg_presision"]

        # 再現率平均（平均行）
        avg_row_recall = avg_row_recall + class_summary_obj["avg_recall"]

        class_summary_list_json.append(class_summary_obj)

    # result_summary_dataの作成
    result_summary_data_json = {"class_summary_list": [], "average": {}}
    result_summary_data_json["class_summary_list"] = class_summary_list_json

    # 平均行の算出
    result_summary_data_json["average"]["avg_row_iou"] = (
        avg_row_iou / index_data_json["class_count"]
    )
    result_summary_data_json["average"]["avg_row_true_count"] = (
        avg_row_true_count / index_data_json["class_count"]
    )
    result_summary_data_json["average"]["avg_row_true_area"] = (
        avg_row_true_area / index_data_json["class_count"]
    )
    result_summary_data_json["average"]["avg_row_tp"] = (
        avg_row_tp / index_data_json["class_count"]
    )
    result_summary_data_json["average"]["avg_row_fp"] = (
        avg_row_fp / index_data_json["class_count"]
    )
    result_summary_data_json["average"]["avg_row_fn"] = (
        avg_row_fn / index_data_json["class_count"]
    )
    result_summary_data_json["average"]["avg_row_presision"] = (
        avg_row_presision / index_data_json["class_count"]
    )
    result_summary_data_json["average"]["avg_row_recall"] = (
        avg_row_recall / index_data_json["class_count"]
    )

    result_data_json["result_list_data"] = result_list_data_json
    result_data_json["result_summary_data"] = result_summary_data_json

    # テンプレート準備
    if ann_flg is True:
        gzo_fw_common_template_path = pathlib.Path(
            __file__
        ).parent.parent.parent / pathlib.Path(
            "gzo_fw_common/templates/segmentation_evaluation"
        )
    else:
        gzo_fw_common_template_path = pathlib.Path(
            __file__
        ).parent.parent.parent / pathlib.Path(
            "gzo_fw_common/templates/segmentation_evaluation_only"
        )

    gzo_fw_common_template_docs = gzo_fw_common_template_path / pathlib.Path("docs")

    # テンプレートのコピー
    shutil.rmtree(docs_dir, ignore_errors=True)
    shutil.copytree(gzo_fw_common_template_docs, docs_dir)

    gzo_fw_common_template_path = pathlib.Path(
        __file__
    ).parent.parent.parent / pathlib.Path("gzo_fw_common/templates")
    # -------------------------------------------------------------------------------
    # テンプレートへの書き込み
    tpl = ""
    with open(
        gzo_fw_common_template_path / pathlib.Path("result_data.js.template"),
        "r",
        encoding=ENCODING_UTF8,
    ) as f_result_list_js_template:
        tpl = f_result_list_js_template.read()

    result_list_template = string.Template(tpl)
    result_data = json.dumps(result_data_json, indent=4)
    f_result_list_js_template.close()

    # js出力
    with open(
        str(docs_dir / pathlib.Path("results/result_data.js")),
        "w",
        encoding=ENCODING_UTF8,
    ) as f_result_list:
        f_result_list.write(result_list_template.substitute(result_data=result_data))


def _iou(bbox1, bbox2):
    """IOUを算出する

    Args:
        a ([int]): bbox1
        b ([int]): bbox2

    Returns:
        float: IOU値
    """

    ax_mn, ay_mn, ax_mx, ay_mx = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    bx_mn, by_mn, bx_mx, by_mx = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w * h

    iou = 0
    if (a_area + b_area - intersect) != 0:
        iou = intersect / (a_area + b_area - intersect)
    return iou


def draw_semantic_segmentation_results_image(
    image_file_path_list: list,
    output_dir: str,
    class_names: list,
    class_colors: dict = None,
):
    """
    SemanticSegmentation推論結果から結果画像を生成する。

    Args:
        imate_file_path_list: グレースケール画像のパスリスト (list[str])
        output_dir: 出力ディレクトリ名 (str)
        class_names: クラス名リスト (list[str])
            配列のk番目の要素である文字列が画像中の値kに対応する。
        class_colors: 色リスト (任意)
            dict((key,val)=(クラス名(str),色(int,3)))形式。
            入力されていない場合、common_configのcolorsを背景以外のクラスに順繰りに割り当てる。
            (背景は#000000で固定)

    Result:
        return output_image_path_list:
            変換後の画像のパスリスト。

    Raises:
        FileNotFoundError:
            outputディレクトリが存在しない場合。
            リスト中の画像ファイルパスに対応するファイルが存在しない場合。
        KeyError:
            対応する色が定義されていないピクセルが画像中に存在する場合。
        Exception:
            画像の読み込みに失敗した場合。
    """

    logger.info("SemanticSegmentation推論結果画像生成開始")

    # output用フォルダが存在するか判定
    if not pathlib.Path(output_dir).exists():
        raise FileNotFoundError("outputディレクトリが存在しません: " + output_dir)

    # クラス名と色のリスト化（クラス名から色を参照するため）
    color_dict = {}

    # 色設定が指定されているとき
    if class_colors:
        # class_namesのリストから1つずつ色を対応させる
        for class_idx, class_name in enumerate(class_names):
            try:
                color_dict[class_idx] = class_colors[class_name]
            except KeyError:
                raise KeyError("クラスに色が指定されていません: " + class_name)
    # 色設定が指定されていないとき
    else:
        # 共通設定から描画用の設定を取得
        colors = json.loads(common_config.get("draw_detection", "colors"))

        # backgroundは#000000で固定
        # それ以外はデフォルト色を順に指定する
        for color_idx, class_name in enumerate(class_names):
            if color_idx == 0:
                color = "#000000"
            else:
                color = colors[(color_idx - 1) % len(colors)]
            color_dict[color_idx] = getrgb(color)

    # 返り値の出力結果画像のパスリスト
    output_image_path_list = []

    # リスト中の画像パスに対して実行
    for image_file_path in image_file_path_list:
        # Pathオブジェクトの取得
        image_file = pathlib.Path(image_file_path)

        # 存在チェック
        if image_file.exists() is False:
            raise FileNotFoundError("ファイルが見つかりません：" + str(image_file))

        # 画像の読み込み
        gray_image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)

        # 画像の読み込みに失敗した場合
        if gray_image is None:
            raise Exception("画像の読み込みに失敗しました：" + str(image_file))

        height, width = gray_image.shape

        result_image = np.zeros((height, width, 3))
        result_image_file_path = pathlib.Path(output_dir) / image_file.name

        # 1ピクセルずつ適用する
        for x in range(height):
            for y in range(width):
                try:
                    result_image[x][y] = np.array(color_dict[gray_image[x][y]])
                except KeyError:
                    raise KeyError("対応する色が定義されていないピクセルが存在します: " + str(image_file))

        # openCVはBGR順なので順序を逆にして保存する
        cv2.imwrite(str(result_image_file_path), result_image[:, :, ::-1])

        output_image_path_list.append(str(result_image_file_path))

    return output_image_path_list


def coordinate_rect(img):
    """
    物体の輪郭から外接矩形の座標を取得して返却
    Args:
      img (ndarray): 画像データ(グレースケール)
    Return:
      min_x: 最小X座標
      min_y: 最小Y座標
      max_x: 最大X座標
      max_y: 最大Y座標
    """
    min_x, min_y, max_x, max_y = 0, 0, 0, 0

    # 輪郭の候補取得
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 煙の全体を取りたいため最大の外接矩形を取得
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    return min_x, min_y, max_x, max_y


def calc_iou(ann_image, pre_image, ann_color, pred_color):
    """
    ２つの画像データからIoUを計算して返却
    Args:
      ann_image (pil): アノテーション画像データ
      pre_image (pil): 推論結果画像データ
      ann_color (list):アノテーションカラー
      pred_color(list):推論のインデックスカラー
    Return:
      TP: 正しく認識できた箇所[px]
      FP: 誤認識箇所[px]
      FN: 未認識箇所[px]
      IoU: 評価指標[%]
    sample:
      calc_iou(
          アノテーションデータ,
          推論結果画像,
          [ind, ind, ind],

      )
    """

    # cv2.findContoursの入力はグレースケール
    ann_gray = cv2.cvtColor(np.array(ann_image), cv2.COLOR_BGR2GRAY)
    pre_gray = cv2.cvtColor(np.array(pre_image), cv2.COLOR_BGR2GRAY)
    # 処理負荷軽減のため、アノテーションと推論結果が存在する範囲座標を取得
    ann_min_x, ann_min_y, ann_max_x, ann_max_y = coordinate_rect(ann_gray)
    pre_min_x, pre_min_y, pre_max_x, pre_max_y = coordinate_rect(pre_gray)

    # 最大範囲を求める
    min_x = min(ann_min_x, pre_min_x)
    min_y = min(ann_min_y, pre_min_y)
    max_x = max(ann_max_x, pre_max_x)
    max_y = max(ann_max_y, pre_max_y)

    # 最大範囲内の全てのピクセルを処理
    tp, fp, fn = 0, 0, 0
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            ann_r, ann_g, ann_b = ann_image.getpixel((x, y))
            pre_r, pre_g, pre_b = pre_image.getpixel((x, y))

            # アノテーション色と推論結果色から一致・見逃し・過検知を判断
            if (
                ann_r == ann_color[0]
                and ann_g == ann_color[1]
                and ann_b == ann_color[2]
            ):
                if (
                    pre_r == pred_color[0]
                    and pre_g == pred_color[1]
                    and pre_b == pred_color[2]
                ):
                    # 一致
                    tp += 1
                else:
                    # 見逃し
                    fn += 1
            else:
                if (
                    pre_r == pred_color[0]
                    and pre_g == pred_color[1]
                    and pre_b == pred_color[2]
                ):
                    # 過検知
                    fp += 1

    # 未検出の場合は0割りになるため
    iou = 0
    if tp + fp + fn != 0:
        iou = tp / (tp + fp + fn)

    return tp, fp, fn, iou


def export_semantic_segmentation_results(
    image_dir_path_str: str,
    output_dir: str,
    dataset_format: str,
) -> None:
    """
    SemanticSegmentationアノテーション出力する

    Args:
        image_dir_path_str (str): 推論結果データパス
        output_dir (str): 出力ディレクトリパス
        dataset_format (str): データセットフォーマット (eiseg/labelme/mltable)

    Raise:
        FileNotFoundError: 引数に指定したファイルやディレクトリが見つからない場合に発生
        ImageReadError: 画像の読み込みに失敗した場合に発生
    """

    logger.info("開始:SemanticSegmentationアノテーション出力")

    image_dir_path = pathlib.Path(image_dir_path_str)

    classes_txt_path = Path(image_dir_path, "classes.txt")

    # input用フォルダが存在するか判定
    if Path(image_dir_path_str).exists() is False:
        raise FileNotFoundError("inputディレクトリが存在しません: " + image_dir_path_str)

    # 存在チェック
    if classes_txt_path.exists() is False:
        raise FileNotFoundError("クラスリストファイル(classes.txt)が存在しません。")

    # クラスファイルを読み込む
    with open(classes_txt_path, mode="r") as f:
        class_list = f.read().split("\n")

    # output用フォルダが存在するか判定
    if Path(output_dir).exists() is False:
        raise FileNotFoundError("outputディレクトリが存在しません: " + output_dir)

    # フォルダ内のファイルを取得
    images_list = []
    categories_list = []
    if dataset_format == "eiseg":
        for class_num, class_name in enumerate(class_list):
            if class_num == 0:
                continue
            categories = {
                "id": class_num,
                "name": class_name,
                "color": [0, 0, 0],
                "supercategory": "",
            }
            categories_list.append(categories)
    annotations_list = []
    # クラスの色別に画像の点群を取得してJsonファイルを作成
    for image_id, file_path in enumerate(image_dir_path.glob("**/*")):
        if file_path.suffix in IMAGE_SUFFIX:
            image_file_name = os.path.basename(file_path)
            image_url = image_dir_path / image_file_name
            # バリデーション
            if image_url.exists() is False:
                raise FileNotFoundError("ファイルが見つかりません：" + str(image_file_name))

            # 画像ファイルの読み込み
            img = cv2.imread(str(image_url), cv2.IMREAD_GRAYSCALE)
            base_img = cv2.imread(str(image_url))

            # 画像の読み込みに失敗した場合
            if img is None:
                raise ImageReadError("画像の読み込みに失敗しました：" + str(image_file_name))
            # 画像ファイルの高さ、幅を取得
            image_height, image_width = img.shape
            min_x, min_y, max_x, max_y = 0, 0, 0, 0
            # クラスリストからデータを取得
            if dataset_format == "mltable":
                label_list = []
                label_confidence_list = []
                for class_num, class_name in enumerate(class_list):
                    if class_num == 0:
                        continue
                    img_processed = np.where(img == class_num, 1, 0).astype("uint8")
                    # 輪郭の候補取得
                    contours, _ = cv2.findContours(
                        img_processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
                    )
                    for cnt in contours:
                        contour_points = []
                        for point in cnt:
                            x, y = point[0]
                            contour_points.append(float(x))
                            contour_points.append(float(y))

                        x, y, w, h = cv2.boundingRect(cnt)
                        min_x = min(float(min_x), x)
                        min_y = min(float(min_y), y)
                        max_x = max(float(max_x), x + w)
                        max_y = max(float(max_y), y + h)
                        bbox_list = [min_x, min_y, max_x, max_y]
                        label = {
                            "label": class_name,
                            "isCrowd": "false",
                            "polygon": contour_points,
                            "bbox": bbox_list,
                        }
                        label_list.append(label)
                        label_confidence_list.append(1.0)
                input_dict = {
                    "image_url": str(image_url),
                    "label": label_list,
                    "label_confidence": label_confidence_list,
                    "image_height": image_height,
                    "image_width": image_width,
                }
                input_json_file_path = pathlib.Path(
                    output_dir, image_file_name + ".json"
                )
            elif dataset_format == "labelme":
                shapes_list = []
                for class_num, class_name in enumerate(class_list):
                    if class_num == 0:
                        continue
                    img_processed = np.where(img == class_num, 1, 0).astype("uint8")
                    # 輪郭の候補取得
                    contours, _ = cv2.findContours(
                        img_processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
                    )
                    for cnt in contours:
                        contour_points = []
                        for point in cnt:
                            x, y = point[0]
                            contour_points.append([int(x), int(y)])

                        shapes = {
                            "label": class_name,
                            "points": contour_points,
                            "shape_type": "polygon",
                            "group_id": "null",
                            "description": "",
                            "flags": {},
                        }
                        shapes_list.append(shapes)
                # 画像ファイルをbase64文字列に変換
                with open(str(image_url), "rb") as image_file:
                    image_data = base64.b64encode(image_file.read())
                input_dict = {
                    "version": "5.2.1",
                    "flags": {},
                    "shapes": shapes_list,
                    "imagePath": str(image_url),
                    "imageData": "null",
                    "imageHeight": image_height,
                    "imageWidth": image_width,
                }
                input_json_file_path = pathlib.Path(
                    output_dir, image_file_name + ".json"
                )
            elif dataset_format == "eiseg":
                bbox_list = []
                for class_num, class_name in enumerate(class_list):
                    if class_num == 0:
                        continue
                    img_processed = np.where(img == class_num, 1, 0).astype("uint8")
                    # 輪郭の候補取得
                    contours, _ = cv2.findContours(
                        img_processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
                    )
                    for id_num, cnt in enumerate(contours, 1):
                        segmentation = []
                        area = cv2.contourArea(cnt)
                        for point in cnt:
                            x, y = point[0]
                            segmentation.append(int(x))
                            segmentation.append(int(y))
                        x, y, w, h = cv2.boundingRect(cnt)
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        max_x = max(max_x, x + w)
                        max_y = max(max_y, y + h)
                        bbox_list = [min_x, min_y, max_x, max_y]
                        annotations = {
                            "id": id_num,
                            "iscrowd": 0,
                            "image_id": image_id,
                            "category_id": class_num,
                            "segmentation": [segmentation],
                            "area": float(area),
                            "bbox": bbox_list,
                        }
                        annotations_list.append(annotations)
                    if len(contours) != 0:
                        x, y = contours[0][0][0]
                        color = base_img[y, x]
                        categories = {
                            "id": class_num,
                            "name": class_name,
                            "color": [int(color[0]), int(color[1]), int(color[2])],
                            "supercategory": "",
                        }
                        categories_list[class_num - 1] = categories
                images = {
                    "id": image_id,
                    "width": image_width,
                    "height": image_height,
                    "file_name": image_file_name,
                    "license": "",
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "",
                }
                images_list.append(images)
            if dataset_format == "labelme" or dataset_format == "mltable":
                with open(str(input_json_file_path), "wt") as json_file:
                    json.dump(input_dict, json_file, indent=2, ensure_ascii=False)
        if dataset_format == "eiseg":
            input_dict = {
                "categories": categories_list,
                "images": images_list,
                "annotations": annotations_list,
                "info": "",
                "licenses": [],
            }
            input_json_file_path = pathlib.Path(output_dir, "annotations.json")
            with open(str(input_json_file_path), "wt") as json_file:
                json.dump(input_dict, json_file, indent=2, ensure_ascii=False)

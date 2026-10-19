# paddle ocrを用いてスラブの文字認識を行う

from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
import yaml
import csv

class CudaOutOfMemoryError(Exception):
    """CUDAのメモリ不足が発生した際にスローされる例外"""
    pass
class ExceptionError(Exception):
    """予期せぬエラーが発生した際にスローされる例外"""
    pass


def predict_paddleocr(ndarray, config):
    '''Paddle OCRで推論
        Args:
            ndarray(ndarray): 画像配列
        Return:
            result[0](list): 画像認識の推論結果 box,test,score
            None(None): 推論失敗時にNoneを返す
    '''
    # ocr = PaddleOCR(use_angle_cls=True, lang='en')
    ocr = PaddleOCR(use_angle_cls=config.get('use_angle_cls'), lang=config.get('lang'))
    result = ocr.ocr(ndarray, cls=True)

    # 画像認識ができない場合 Noneを返す
    if result is None or len(result) == 0:
        return None

    return result[0]

def custom_draw_ocr(image, boxes, txts=None, scores=None, font_path='simfang.ttf', font_size=16):
    """
    OCR結果を描画するカスタム関数（テキストラベルを描画しないバージョン）
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    
    # バウンディングボックスを描画
    for box, score in zip(boxes, scores):
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        draw.polygon([tuple(pt) for pt in box], outline='red', width=2)
    
    return np.array(image)

def draw_csv(ndarray, result, filename, csv_path, font_path):
    '''推論結果を画像に描画
        Args:
            ndarray(ndarray): 画像配列
            result(list): 画像認識の推論結果 box,test,score
            font_path(str): フォントファイルのパス
        Return:
            im_show(ndarray): 推論結果の情報がある画像配列
            ndarray(ndarray): 元の画像配列
    '''

    # Noneの場合 元画像を出力
    try:
        if result is None:
            # file_exists=os.path.isfile(csv_path)
            # with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            #     writer=csv.writer(f)
            #     if not file_exists:
            #         writer.writerow(['ImageName', 'BoundingBoxNo', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'Text'])
            #         idx=0
            #         text=None
            #         x1, y1 = 0
            #         x2, y2 = 0
            #         x3, y3 = 0
            #         x4, y4 = 0
            #         writer.writerow([filename, idx, x1, y1, x2, y2, x3, y3, x4, y4, text])
            return ndarray
        file_exists=os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer=csv.writer(f)
            if not file_exists:
                writer.writerow(['ImageName', 'BoundingBoxNo', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'Text'])
            for idx, item in enumerate(result):
                box, info = item
                text, score = info
                x1, y1 = box[0]
                x2, y2 = box[1]
                x3, y3 = box[2]
                x4, y4 = box[3]
                writer.writerow([filename, idx, x1, y1, x2, y2, x3, y3, x4, y4, text])

        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        # im_show = draw_ocr(ndarray, boxes, font_path=font_path)
        # im_show = np.array(image)
        im_show = custom_draw_ocr(ndarray, boxes, txts, scores, font_path=font_path)

        return im_show
    except Exception as e:
            print(f"####drow_csv{e}####")
            raise



def paddleocr_main(input_dir, output_dir, run_id):

    src_dir = input_dir
    dst_dir = output_dir
    print(dst_dir)
    yaml_files = [f for f in input_dir.glob("*.yaml") if f.is_file()]
    font_path = 'PaddleOCR/doc/fonts/simfang.ttf'
    if not yaml_files:
        raise ValueError(f"No YAML configuration file found in the directory: {input_dir}")
    yaml_path = str(yaml_files[0])
    with open(yaml_path,'r') as stream:
        config=yaml.safe_load(stream)
    
    result_name=f"{run_id}_{config.get('lang')}_推論結果.csv"
    csv_file=os.path.join(output_dir,result_name)
    

    # src_dir内の画像ファイルを取得し、paddleOCRで推論をかける
    # 推論結果はdst_dir内に格納する
    try:
        for filename in os.listdir(src_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                frame = cv2.imread(os.path.join(src_dir, filename))
                    # predict by paddleOCR
                result = predict_paddleocr(frame, config)
                    # draw boxes,texts and scores
            # try:
                im_show=draw_csv(frame, result, filename, csv_file, font_path)
                cv2.imwrite(os.path.join(dst_dir , filename), im_show)
    except Exception as e:
        error_str = str(e)
        print(f"Error in predict: {error_str}")
        if 'CUDA' in error_str or 'out of memory' in error_str:
            raise CudaOutOfMemoryError(error_str) from e
        else:
            raise ExceptionError(error_str) from e
                

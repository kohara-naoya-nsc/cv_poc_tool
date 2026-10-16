# paddle ocrを用いてスラブの文字認識を行う

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import cv2
import os


def predict_paddleocr(ndarray):
    '''Paddle OCRで推論
        Args:
            ndarray(ndarray): 画像配列
        Return:
            result[0](list): 画像認識の推論結果 box,test,score
            None(None): 推論失敗時にNoneを返す
    '''
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(ndarray, cls=True)

    # 画像認識ができない場合 Noneを返す
    if result is None or len(result) == 0:
        return None

    return result[0]

def draw_img(ndarray, result, font_path):
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
    if result is None:
        return ndarray

    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]

    im_show = draw_ocr(ndarray, boxes, txts, scores, font_path=font_path)
    im_show = np.array(im_show)

    return im_show


if __name__ == '__main__':

    src_dir = './data/test/'
    dst_dir = './dst/test/'
    font_path = '../PaddleOCR/doc/fonts/simfang.ttf'
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    # src_dir内の画像ファイルを取得し、paddleOCRで推論をかける
    # 推論結果はdst_dir内に格納する
    for filename in os.listdir(src_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            frame = cv2.imread(os.path.join(src_dir, filename))
            # predict by paddleOCR
            result = predict_paddleocr(frame)
            # draw boxes,texts and scores
            im_show = draw_img(frame, result, font_path)

            cv2.imwrite(dst_dir + filename, im_show)


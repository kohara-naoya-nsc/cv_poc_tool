"""
PaddleOCRを使用して指定した複数の言語のOCRモデルを初期化するコード
言語コードを指定し、PaddleOCRのインスタンスを作成して使用することができる
関数:
    initialize_ocr(languages=['japan']):
        指定した言語モデルを初期化する関数。
使用例:
    from paddleocr import PaddleOCR
    languages = ['japan', 'english']
    ocr_models = initialize_ocr(languages)
"""

from paddleocr import PaddleOCR

def initialize_ocr(languages=['japan']):
    """
    指定した言語モデルを初期化する関数
    Args:
        languages (list of str): 使用する言語コードのリストを指定する（例：['ch', 'en', 'french', 'german', 'korean', 'japan']）。デフォルトは['japan']。
    Returns:
        dict: 指定した言語ごとのOCRモデルの辞書を返す
    """
    ocr_models = {}
    for language in languages:
        # 指定した言語でOCRモデルを初期化
        ocr_models[language] = PaddleOCR(use_angle_cls=True, lang=language)
    return ocr_models

# 言語コードを定義（例：'ch', 'en', 'french', 'german', 'korean', 'japan'）
languages = ['japan', 'en']

# OCRモデルを初期化
ocr_models = initialize_ocr(languages)


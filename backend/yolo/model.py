import os
from ultralytics import YOLO

def download_model(model_name, save_path, test_img):
    """
    指定されたモデル名のYOLOモデルをダウンロード
    Args:
        model_name(str): ダウンロードするモデルの名称
        save_path(str): モデルを保存するディレクトリのパス
        test_img(str): テスト用の画像ファイルのパス
    """
    # モデルのパスを作成
    model_path = os.path.join(save_path, model_name)
    # YOLOモデルをロード
    model = YOLO(model_path)
    # テスト画像に対する推論結果を取得
    result = model(source=test_img, verbose=False)


if __name__ == "__main__":
    # モデル保存ディレクトリの設定
    model_dir = "/app/models/"
    # テスト用画像ファイルのパス
    test_img = "/app/tests/dog.jpg"
    # 保存ディレクトリが存在しない場合は作成する
    os.makedirs(model_dir, exist_ok=True)

   # ダウンロードするモデル名の一覧
    model_names = [
        'yolov8n.pt',
        'yolov8m.pt',
        'yolov8n-cls.pt',
        'yolov8m-cls.pt',
    ]

    for model_name in model_names:
        # モデル処理のログ出力
        print(f"Processing model: {model_name}")
        # モデルをダウンロード
        download_model(model_name, model_dir, test_img)

# pytorchの公式イメージを使用
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# 非対話モードでパッケージをインストール
ARG DEBIAN_FRONTEND=noninteractive

# アプリケーションファイルを配置するディレクトリに移動
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-dev && apt-get install -y vim

# requirements.txtからPythonパッケージをインストール
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# YOLOv8のモデルをダウンロードするスクリプトをコピーし、実行
COPY model.py .
COPY tests /app/tests/
RUN python3 model.py

# FastAPIアプリをポート81で全IPからの接続を受け付けるモードで実行
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "81"]

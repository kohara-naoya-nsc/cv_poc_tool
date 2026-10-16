FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
# アプリケーションファイルを配置するディレクトリに移動
WORKDIR /app

RUN apt-get update && apt-get install -y \
    vim \
    libgl1 \
    libglu1 \
    libglib2.0-0\
    libsm6 \
    libxrender1 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# requirements.txtからPythonパッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# FastAPIアプリをポート82で全IPからの接続を受け付けるモードで実行
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "82"]

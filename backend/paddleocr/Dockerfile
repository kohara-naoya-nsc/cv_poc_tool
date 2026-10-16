
FROM paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5

# アプリケーションファイルを配置するディレクトリに移動
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8

# ベースのOSをアップデート
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; 
#     apt-get update; \
#     apt-get upgrade -y; \
#     apt-get autoremove -y

# RUN apt-get update && apt-get install -y vim

# requirements.txtからPythonパッケージをインストール
COPY requirements.txt ./
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install --ignore-installed PyYAML && \
    pip3 install -r requirements.txt

# 公式リポジトリをクローン
RUN git clone https://github.com/PaddlePaddle/PaddleOCR.git

# PaddleOCRのモデルをダウンロード
COPY model.py .
RUN python3 model.py

# FastAPIアプリをポート83で全IPからの接続を受け付けるモードで実行
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "83"]

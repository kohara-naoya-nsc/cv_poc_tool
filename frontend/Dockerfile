# Python 3.10.11を使用してStreamlitアプリケーションをビルドし、ポート8501で実行する環境を整備する

# Python 3.10.11の軽量版(bullseyeベース)のイメージを使用
# FROM python:3.10.11-slim-bullseye

# Python 3.10.11の軽量版(bookwormベース)のイメージを使用
FROM python:3.10-slim-bookworm

# アプリケーションファイルを配置するディレクトリに移動
WORKDIR /app

# 必要なシステムパッケージをインストールし、キャッシュを削除
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
# requirements.txtからPythonパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# Streamlitアプリが実行されるポート8501を開く
EXPOSE 8501

# ヘルスチェックとして、localhost:8501/_stcore/healthにアクセスする
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Streamlitアプリをポート8501で全IPからの接続を受け付けるモードで実行
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

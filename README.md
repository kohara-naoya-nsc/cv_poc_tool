# cv_PoC_tool
ツール名：画像解析WebUIツール<br>
案件名：画像AI及び生成AIに関する技術検討及びソフト試作<br>

## 概要
コードが書けない操業技術者でも画像解析のPoC(実証実験)を容易に行えるWebUIツールを提供する<br>
画像解析におけるモデルの学習から推論結果の確認までを当ツールで一括して行えるようにする<br>


## 使い方

### アプリ起動

```bash
# /home/csemi/cv_poc_tool
# 実行
docker compose up -d
# フォアグラウンドで実行
docker compose up
```


### アプリ停止

```bash
# /home/csemi/cv_poc_tool
# アプリ停止
docker compose down
```

### Makefileを使った起動と停止

```bash
# /home/csemi
# makefileを使ったアプリ起動　cv_annotation_toolも同時に起動
make up
```

```bash
# /home/csemi
# makefileを使ったアプリ停止　cv_annotation_toolも同時に停止
make down
```

#### Makefile

```
# cd /home/csemi

PROJECTS = cv_annotation_tool cv_poc_tool

help:
	@echo "次のコマンドを利用できます:"
	@echo ""
	@echo "make up                  - 全てのプロジェクトのコンテナ起動 (バックグラウンド)"
	@echo "make up-dino             - GroundingDINOを含むアノテーションツールのコンテナ起動 (フォアグラウンド)"
	@echo "make up-annotation       - アノテーションツールのコンテナ起動 (バックグラウンド)"
	@echo "make up-poc		 - 画像PoCツールのコンテナ起動 (バックグラウンド)"
	@echo "make down                - 全てのプロジェクトのコンテナ停止"
	@echo "make status              - 全てのプロジェクトの状態確認"
	@echo "make help		 - このヘルプを表示"

up:
	@for project in $(PROJECTS); do \
		echo "=== $$project 起動中 ==="; \
		cd $$project && docker compose up -d; \
		cd ../; \
	done

up-dino:
	echo "===  cv_annotation_tool 起動中 ==="; \
	cd cv_annotation_tool && docker compose --profile dino up; \

up-annotation:
	echo "===  cv_annotation_tool 起動中 ==="; \
	cd cv_annotation_tool && docker compose up -d; \

up-poc:
	echo "===  cv_poc_tool 起動中 ==="; \
	cd cv_poc_tool  && docker compose up -d; \

down:
	@for project in $(PROJECTS); do \
		echo "=== $$project 停止中 ==="; \
		cd $$project && docker compose down; \
		cd ../; \
	done

status:
	@for project in $(PROJECTS); do \
		echo "=== $$project 状態 ==="; \
		cd $$project && docker compose ps; \
		cd ../; \
	done

.PHONY: help up up-dino up-annotation up-poc down status

```

## ディレクトリ構成

```bash
# cv_poc_toolのディレクトリ構成
cv_poc_tool/
├── frontend/   # Streamlitアプリ
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .streamlit/            # Streamlit設定ファイルの格納ディレクトリ
│   ├── pages/                  # Streamlitページの格納ディレクトリ
│   ├── utils/                      # Streamlit内での共通関数
│   ├── assets/                  # 画像等の静的ファイルの格納ディレクトリ
│   ├── docs/                       # ドキュメントの格納ディレクトリ
│   └── app.py                  # メインアプリ
│
├── backend/    # 各モデルごとのバックエンドサービス
│   ├── yolo/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── models/             # ベースモデルの格納ディレクトリ
│   │   ├── tests/              # テストコードの格納ディレクトリ
│   │   ├── model.py            # モデルロード用のスクリプト
│   │   ├── yolo.py             # YOLOv8の処理スクリプト(推論処理,モデル学習)
│   │   └── app.py              # FastAPIのAPIエンドポイント
│   │
│   ├── patchcore/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── models/             # ベースモデルの格納ディレクトリ
│   │   ├── tests/              # テストコードの格納ディレクトリ
│   │   ├── model.py            # モデルロード用のスクリプト
│   │   ├── patchcore/          # PatchCoreの関数ディレクトリ
│   │   ├── patchcore.py        # PatchCoreの処理スクリプト(推論処理,モデル学習)
│   │   └── app.py              # FastAPIのAPIエンドポイント
│   │
│   └── paddleocr/
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── models/             # ベースモデルの格納ディレクトリ
│       ├── tests/              # テストコードの格納ディレクトリ
│       ├── model.py            # モデルロード用のスクリプト
│       ├── PaddleOCR/          # PaddleOCRの公式リポジトリ
│       ├── paddleocr.py        # PaddleOCRの処理スクリプト(推論処理)
│       └── app.py              # FastAPIのAPIエンドポイント
│
├── shared/     # 共有リソース
│   ├── predict/        # 推論処理時に使用するリソース 
│   │   └── yyyymmddhhmmss/     # 作成日時(run_id)ごとにディレクトリを作成
│   │       ├── pre_data/       # 推論処理の前に必要なデータを格納するディレクトリ (ex. 推論画像)
│   │       └── result_data/    # 推論処理の後に出力されるデータを格納するディレクトリ (ex. 結果画像,CSVファイル) 
│   │
│   └── train/          # 推論処理時に使用するリソース
│       └── yyyymmddhhmmss/     # 作成日時(run_id)ごとにディレクトリを作成
│           ├── pre_data/       # 学習処理の前に必要なデータを格納するディレクトリ (ex. 学習画像,検証画像,yamlファイル)
│           └── result_data/    # 学習処理の後に出力されるデータを格納するディレクトリ (ex. 学習済みモデル,runsの中身)
│
└── compose.yml  # 全コンテナを管理するメイン設定ファイル

```
import pytest
from fastapi.testclient import TestClient
from unittest.mock import  patch
from app import app

@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture()
def mock_yolo():
    with patch("app.yolo") as mock:
        yield mock 

@pytest.fixture()
def mock_path_exists():
    with patch("app.Path.exists", return_value=True) as mock:
        yield mock 

@pytest.fixture()
def mock_path_not_exists():
    with patch("app.Path.exists", return_value=False) as mock:
        yield mock 

#### /predict/ ####
# 入力データのディレクトリの確認
def test_predict_input_dir_success(client, mock_yolo, mock_path_exists):
    response = client.post("/predict/", data={"run_id": "vaild_run_id"})
    assert response.status_code == 200

def test_predict_input_dir_failure(client, mock_yolo, mock_path_not_exists):
    response = client.post("/predict/", data={"run_id": "invalid_run_id"})
    assert response.status_code == 404

# yolo.predictでエラー発生時のテスト
def test_predict_yolo_error(client, mock_yolo, mock_path_exists):
    # YOLOのpredictがエラーを送出するように設定
    mock_yolo.predict.side_effect = Exception("YOLO error")
    response = client.post("/predict/", data={"run_id": "vaild_run_id"})
    assert response.status_code == 500


#### /train/ ####
# 入力データのディレクトリの確認
def test_train_input_dir_success(client, mock_yolo, mock_path_exists):
    response = client.post("/train/", data={"run_id": "vaild_run_id"})
    assert response.status_code == 200

def test_train_input_dir_failure(client, mock_yolo, mock_path_not_exists):
    response = client.post("/train/", data={"run_id": "invalid_run_id"})
    assert response.status_code == 404

# yolo.predictでエラー発生時のテスト
def test_train_yolo_error(client, mock_yolo, mock_path_exists):
    # YOLOのpredictがエラーを送出するように設定
    mock_yolo.train.side_effect = Exception("YOLO error")
    response = client.post("/train/", data={"run_id": "vaild_run_id"})
    assert response.status_code == 500

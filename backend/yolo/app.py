import os
from fastapi import FastAPI, Form, HTTPException
from pathlib import Path
from yolo import YOLO, CudaOutOfMemoryError, ExceptionError

app = FastAPI()

@app.post("/predict/image/")
async def predict(run_id: str = Form(...)):
    # 入力データのディレクトリを確認
    input_dir = Path(f"shared/predict/{run_id}/pre_data")
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory does not exist: {input_dir}")
        
    # 出力データのディレクトリを確認
    output_dir = Path(f"shared/predict/{run_id}/result_data")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO()
    flag = True
    type="image"
    # YOLO推論       
    try:
        yolo.predict(input_dir, output_dir, run_id, flag, type)
    except CudaOutOfMemoryError:
        raise HTTPException(status_code=503, detail="CUDA out of memory")
    except ExceptionError:
        raise HTTPException(status_code=500)
    
@app.post("/predict/movie/")
async def predict(run_id: str = Form(...)):
    # 入力データのディレクトリを確認
    input_dir = Path(f"shared/predict/{run_id}/pre_data")
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory does not exist: {input_dir}")
        
    # 出力データのディレクトリを確認
    output_dir = Path(f"shared/predict/{run_id}/result_data")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO()
    flag = True
    type="movie"
    # YOLO推論
    try:
        yolo.predict(input_dir, output_dir, run_id, flag, type)
        # raise HTTPException(status_code=200)
    except CudaOutOfMemoryError:
        raise HTTPException(status_code=503, detail="CUDA out of memory")
    except ExceptionError:
        raise HTTPException(status_code=500)

@app.post("/predict_img/image/")
async def predict(
    run_id: str = Form(...),
    model_path: str = Form(...)
    ):
    ###############################
    # 各ディレクトリを確認
    ###############################
    # 入力データのディレクトリを確認
    input_dir = Path(f"shared/predict/{run_id}/pre_data")
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory does not exist: {input_dir}")
        
    # 出力データのディレクトリを確認
    output_dir = Path(f"shared/predict/{run_id}/result_data")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(model_path)
    flag = False
    type="image"

    ###############################
    # YOLO推論       
    ###############################
    try:
        yolo.predict(input_dir, output_dir, run_id, flag, type)
    except CudaOutOfMemoryError:
        raise HTTPException(status_code=503, detail="CUDA out of memory")
    except ExceptionError:
        raise HTTPException(status_code=500)
    
@app.post("/predict_img/movie/")
async def predict(
    run_id: str = Form(...),
    model_path: str = Form(...)
    ):
    ###############################
    # 各ディレクトリを確認
    ###############################
    # 入力データのディレクトリを確認
    input_dir = Path(f"shared/predict/{run_id}/pre_data")
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory does not exist: {input_dir}")
        
    # 出力データのディレクトリを確認
    output_dir = Path(f"shared/predict/{run_id}/result_data")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(model_path)
    flag = False
    type="movie"

    ###############################
    # YOLO推論       
    ###############################
    try:
        yolo.predict(input_dir, output_dir, run_id, flag, type)
    except CudaOutOfMemoryError:
        raise HTTPException(status_code=503, detail="CUDA out of memory")
    except ExceptionError:
        raise HTTPException(status_code=500)

@app.post("/train/")
async def train(run_id: str = Form(...)):
    ###############################
    # 入力データのディレクトリを確認
    ###############################
    input_dir = Path(f"shared/train/{run_id}/pre_data")
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory does not exist: {input_dir}")
        
    ###############################
    # 出力データのディレクトリを確認
    ###############################
    output_dir = Path(f"shared/train/{run_id}/result_data")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    ###############################
    # YOLO学習
    ###############################
    yolo = YOLO()
    flag = True
    try:
        yolo.train(input_dir, output_dir, run_id, flag)
    except CudaOutOfMemoryError:
        raise HTTPException(status_code=503, detail="CUDA out of memory")
    except ExceptionError:
        raise HTTPException(status_code=500)

@app.post("/train_img/")
async def train(
    run_id: str = Form(...),
    resume: bool = Form(...),
    use_model: str = Form(...),
    ):
    ###############################
    # 各ディレクトリを確認
    ###############################
    # 入力データのディレクトリを確認
    input_dir = Path(f"shared/train/{run_id}/pre_data")
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory does not exist: {input_dir}")
        
    # 出力データのディレクトリを確認
    output_dir = Path(f"shared/train/{run_id}/result_data")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    ###############################
    # YOLO学習
    ###############################
    # １から学習する場合
    if resume is False:
        pwd = os.getcwd()
        model_path = os.path.join(pwd, use_model)
        yolo = YOLO(model_path)

    # 再学習する場合
    else:
        yolo = YOLO(use_model)
    
    flag = False
    try:
        # YOLO学習
        yolo.train(input_dir, output_dir, run_id, flag)
    except CudaOutOfMemoryError:
        raise HTTPException(status_code=503, detail="CUDA out of memory")
    except ExceptionError:
        raise HTTPException(status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=81)

from fastapi import FastAPI,  Form, HTTPException
from pathlib import Path
from paddleocr_predict import paddleocr_main, CudaOutOfMemoryError, ExceptionError

app = FastAPI()
# patchcore = PatchCore()

@app.post("/predict/")
async def predict(run_id: str = Form(...)):
    # 入力データのディレクトリを確認
    input_dir = Path(f"shared/predict/{run_id}/pre_data")
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory does not exist: {input_dir}")
        
    # 出力データのディレクトリを確認
    output_dir = Path(f"shared/predict/{run_id}/result_data")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # paddleocr推論       
    try:
        paddleocr_main(input_dir, output_dir, run_id)
        # raise HTTPException(status_code=200)
    except CudaOutOfMemoryError:
        raise HTTPException(status_code=503, detail="CUDA out of memory")
    except ExceptionError:
        raise HTTPException(status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=83)

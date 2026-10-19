from fastapi import FastAPI,  Form, HTTPException
from pathlib import Path
from patchcore.patchcore import PatchCore
from patchcore.dataset import MVTecADImageOnly
import yaml
import os
import glob
import torch
import rich
import csv
import cv2
import shutil

app = FastAPI()
# patchcore = PatchCore()

@app.post("/predict/image/")
async def predict(run_id: str = Form(...)):
    # 入力データのディレクトリを確認
    try:
        input_dir = Path(f"shared/predict/{run_id}/pre_data")
        if not input_dir.exists():
            raise HTTPException(status_code=404, detail=f"Input directory does not exist: {input_dir}")

        # 出力データのディレクトリを確認
        output_dir = Path(f"shared/predict/{run_id}/result_data")
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        yaml_files = [f for f in input_dir.glob("*.yaml") if f.is_file()]
        yaml_path = str(yaml_files[0])
        with open(yaml_path,'r') as stream:
            config=yaml.safe_load(stream)

        dataset_arg={
        "load_shape" : (config.get('load_size'), config.get('load_size')),
        "input_shape" : (config.get('input_size'), config.get('input_size')),
        "im_mean" : (0.485, 0.456, 0.406),
        "im_std" : (0.229, 0.224, 0.225),
        }
        model_path=glob.glob(os.path.join(input_dir,"*.faiss"))
        patchcore = PatchCore(config.get('net_model'),config.get('repo'),config.get('device'),config.get('sampling_ratio'))
        patchcore.load(model_path[0])

        dataset = MVTecADImageOnly(input_dir, **dataset_arg)
        dloader=torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=config.get('num_workers'), pin_memory=True)

    # PatchCore推論       
    # try:
        rows=[]
        header=["filename", "anomaly_score"]
        output_csv=f"{output_dir}/{run_id}_{config.get('net_model')}_推論結果.csv"
        for x, gt, label, filepath, x_type in rich.progress.track(dloader, description="Processing..."):

            anomaly_map_rw, score=patchcore.predict(x, config.get('n_neighbors'))
            file_name=os.path.basename(filepath[0])

            rows.append([file_name, score])
            # print(f"rows:{rows}")
            # print(f"anomaly_map_rw:{anomaly_map_rw}")

            patchcore.save_anomaly_map(output_dir, anomaly_map_rw, filepath[0], x_type[0], contour=config.get('contour'))

        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer=csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)        

    except Exception as e:
        error_str = str(e)
        print(f"Error in predict: {error_str}")
        if 'CUDA' in error_str or 'out of memory' in error_str:
            raise HTTPException(status_code=503, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail=str(e))

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

    yaml_files = [f for f in input_dir.glob("*.yaml") if f.is_file()]
    yaml_path = str(yaml_files[0])
    with open(yaml_path,'r') as stream:
        config=yaml.safe_load(stream)

    result_name=f"{run_id}_{config.get('net_model')}_推論結果"
    dataset_arg={
        "load_shape" : (config.get('load_size'), config.get('load_size')),
        "input_shape" : (config.get('input_size'), config.get('input_size')),
        "im_mean" : (0.485, 0.456, 0.406),
        "im_std" : (0.229, 0.224, 0.225),
    }
    files = [f for f in input_dir.glob("*") if f.is_file() and f.suffix.lower() in ['.avi','.mp4']]
    for file in files:
        temp_dir='temp_frames'
        os.makedirs(temp_dir,exist_ok=True)
        temp_dir2='temp_frames2'
        os.makedirs(temp_dir2,exist_ok=True)
        file_path = str(file)
        file_name = file.name
        file_ext = file.suffix.lower()
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            print(f"Failed to open video: {file_name}")
            return 
    
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_idx=0
        try:
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break  

                frame_filename=os.path.join(temp_dir,f'frame_{frame_idx}.jpg')
                cv2.imwrite(frame_filename, frame)
                frame_idx+=1
        except Exception as e:
            print(f"###app:{e}###")
            raise HTTPException(status_code=500, detail=str(e))

        model_path=glob.glob(os.path.join(input_dir,"*.faiss"))
        patchcore = PatchCore(config.get('net_model'),config.get('repo'),config.get('device'),config.get('sampling_ratio'))
        patchcore.load(model_path[0])

        dataset = MVTecADImageOnly(temp_dir, **dataset_arg)
        dloader=torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=config.get('num_workers'), pin_memory=True)

        # PatchCore推論
        rows=[]
        header=["filename", "anomaly_score"]
        output_csv=f"{output_dir}/{run_id}_{config.get('net_model')}_推論結果.csv"    
    
        try:       
            for x, gt, label, filepath, x_type in rich.progress.track(dloader, description="Processing..."):

                anomaly_map_rw, score=patchcore.predict(x, config.get('n_neighbors'))

                frame_name=os.path.basename(filepath[0])
                rows.append([frame_name, score])

                patchcore.save_anomaly_map(temp_dir2, anomaly_map_rw, filepath[0], x_type[0], contour=config.get('contour'))
                # patchcore.predict(dataset, config.get('n_neighbors'))
            #     # raise HTTPException(status_code=200)
            with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
                writer=csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        except Exception as e:
            error_str = str(e)
            print(f"Error in predict: {error_str}")
            if 'CUDA' in error_str or 'out of memory' in error_str:
                raise HTTPException(status_code=503, detail=str(e))
            else:
                raise HTTPException(status_code=500, detail=str(e))
            

        FOURCC_MAP={
            '.mp4': 'mp4v',
            '.avi': 'XVID',
        }
        fourcc_code=FOURCC_MAP.get(file_ext,'mp4v')
        # print(f"fourcc_code:{fourcc_code}")
        fourcc=cv2.VideoWriter_fourcc(*fourcc_code)
        output_video_path = f"{output_dir}/{file_name}"
        # print(output_video_path)
        out=cv2.VideoWriter(output_video_path, fourcc, fps, (config.get('input_size'),config.get('input_size')))
        if not out.isOpened():
            print("エラー:VideoWriterを初期化できませんでした。")
            shutil.rmtree(temp_dir)
            return
        try:
            tmp_list=[]
            tmp_files=glob.glob(f"{temp_dir2}/*.jpg")
            for file in tmp_files:
                rename_file=file[len(temp_dir2)+3:][:-4]
                print(f"file:{file}")
                rename_path=f"{temp_dir2}/{rename_file}"
                print(f"rename_path:{rename_path}")
                print(f"os.path.join(temp_dir2,rename_file):{os.path.join(temp_dir2,rename_file)}")
                os.rename(file, os.path.join(temp_dir2,rename_file))
                tmp_list.append(rename_file)
            # print(os.listdir(temp_dir2))
            # print(glob.glob(f"{os.listdir(temp_dir2)}/*.jpg"))
            # print(tmp_list)
            image_files=sorted(
                [f for f in tmp_list if f.startswith('frame_') and f.endswith('.jpg')],
                key=lambda x: int(x.split('_')[1].split('.')[0])
            )
            # print(f"image_files:{image_files}")

            for image_file in image_files:
                print(f"frame:{image_file}")
                img_path=os.path.join(temp_dir2,image_file)
                img=cv2.imread(img_path)
                if img is not None:
                    out.write(img)
                else:
                    print(f"警告:画像{img_path}を読み込めませんでした。")
        finally:
            out.release()
            shutil.rmtree(temp_dir)
            shutil.rmtree(temp_dir2)

    

@app.post("/train/")
async def train(run_id: str = Form(...)):
    ##########################
    # 各ディレクトリを確認
    ##########################
    # 入力データのディレクトリを確認
    input_dir = Path(f"shared/train/{run_id}/pre_data")
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory does not exist: {input_dir}")
        
    # 出力データのディレクトリを確認
    output_dir = Path(f"/app/shared/train/{run_id}/result_data")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    yaml_files = [f for f in input_dir.glob("*.yaml") if f.is_file()]
    yaml_path = str(yaml_files[0])
    with open(yaml_path,'r') as stream:
        config=yaml.safe_load(stream)

    #######################
    # PatchCore学習
    #######################
    filepath=os.path.join(output_dir,f"{run_id}_{config.get('net_model')}.faiss")
    dataset_arg={
        "load_shape" : (config.get('load_size'), config.get('load_size')),
        "input_shape" : (config.get('input_size'), config.get('input_size')),
        "im_mean" : (0.485, 0.456, 0.406),
        "im_std" : (0.229, 0.224, 0.225),
    }
    patchcore = PatchCore(config.get('net_model'),config.get('repo'),config.get('device'),config.get('sampling_ratio'))
    dataset = MVTecADImageOnly(input_dir, **dataset_arg)
    try:
        patchcore.fit(dataset, config.get('batch'), config.get('num_workers'))
        patchcore.save(filepath)
    except Exception as e:
        error_str = str(e)
        print(f"Error in predict: {error_str}")
        if 'CUDA' in error_str or 'out of memory' in error_str:
            raise HTTPException(status_code=503, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=82)

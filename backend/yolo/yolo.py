import os
import shutil
import glob
import csv
import cv2
from datetime import datetime
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO as UltralyticsYOLO
from ultralytics import settings

class CudaOutOfMemoryError(Exception):
    """CUDAのメモリ不足が発生した際にスローされる例外"""
    pass
class ExceptionError(Exception):
    """予期せぬエラーが発生した際にスローされる例外"""
    pass


class YOLO:
    def __init__(self, model_path=None):
        """モデルの初期化
        Args:
            model_path: YOLOモデルのパス
        """
        # model_path = next((f for f in input_dir.glob("*.pt") if f.is_file() ), None)
        try:
            self.use_model = Path(model_path).stem
            self.model = UltralyticsYOLO(model_path)

        except Exception as e:
            error_str = str(e)
            raise ExceptionError(error_str) from e


    def predict(self, input_dir: Path, output_dir: Path, run_id: str, flag: bool, file_type: str):
        """推論処理
        Args:
            input_dir: 入力画像が格納されているディレクトリ
            output_dir: 推論結果が格納されるディレクトリ
        Returns:
            推論結果のオブジェクト
        Raises:
            ValueError: 入力ディレクトリ内に画像ファイルがない場合
        """
        # 入力ディレクトリ内の画像ファイルをリストアップ
        try:
            files = [f for f in input_dir.glob("*") if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png','.avi','.mp4']]
            pt_file = next((f for f in input_dir.glob("*.pt") if f.is_file() ), None)
            # pt_filename=os.path.basename(pt_files)
        
            if not files:
                raise ValueError(f"No images found in the directory: {input_dir}")

            if pt_file is None:
                raise ValueError(f"No model found in the directory: {input_dir}")

            result_name=f"{run_id}_{pt_file.name}_推論結果"
            # os.makedirs(os.path.join(output_dir,result_name))
        # try:
        # 推論実行
            rows=[]
            if flag == True and file_type == "image":
                # 画像の処理
                output_csv=f"{output_dir}/{result_name}/{result_name}.csv"
                print(f"output_csv:{output_csv}")
                header=["filename", "label", "confidence", "x1", "y1", "x2", "y2"]
                results = self.model.predict(source=str(input_dir), save=True, project=output_dir, name=result_name)
                # print(f"results:{results}")
                for result in results:
                    file_name=Path(result.path).name
                    for box in result.boxes:
                        class_id=int(box.cls[0].item())
                        class_name=self.model.names[class_id]
                        confidence=box.conf[0].item()
                        x1,y1,x2,y2=map(int,box[0].xyxy[0].tolist())
                        rows.append([file_name, class_name, confidence, x1, y1, x2, y2])

                    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
                        writer=csv.writer(f)
                        writer.writerow(header)
                        writer.writerows(rows)
            
        except Exception as e:
            error_str = str(e)
            print(f"Error in predict: {error_str}")
            if 'CUDA' in error_str or 'out of memory' in error_str:
                raise CudaOutOfMemoryError(error_str) from e
            else:
                raise ExceptionError(error_str) from e

        try:
            if flag == True and file_type == "movie":
                    # 動画の処理
                temp_dir='temp_frames'
                os.makedirs(temp_dir,exist_ok=True)
                os.makedirs(os.path.join(output_dir,result_name))
                for file in files:
                    file_path = str(file)
                    file_name = file.name
                    file_ext = file.suffix.lower()
                    
                    output_csv=f"{output_dir}/{result_name}/{result_name}.csv"
                    header=["filename", "label", "confidence", "x1", "y1", "x2", "y2"]
                    output_video_path = f"{output_dir}/{result_name}/{file_name}"

                    cap = cv2.VideoCapture(str(file_path))
                    if not cap.isOpened():
                        print(f"Failed to open video: {file_name}")
                        return 

                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    frame_idx=0
                    class_name="unknown"
                    confidence=0.0
                    x1=y1=x2=y2=0
                    while(cap.isOpened()):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        results = self.model(frame)
                        annotated_frame=results[0].plot()
                        frame_filename=os.path.join(temp_dir,f'frame_{frame_idx}.jpg')
                        cv2.imwrite(frame_filename, annotated_frame)
                        for box in results[0].boxes:
                            print(f"##box:{box}####")
                            if box.cls.shape[0] == 0:
                                continue
                            class_id=int(box.cls[0].item())
                            class_name=self.model.names[class_id] if class_id < len(self.model.names) else "unkown"
                            confidence=float(box.conf[0].item())
                            x1,y1,x2,y2=map(int,box[0].xyxy[0].tolist())
                        rows.append([file_name, class_name, confidence, x1, y1, x2, y2])
                        frame_idx+=1

                    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
                        writer=csv.writer(f)
                        writer.writerow(header)
                        writer.writerows(rows)

                    # out_video.release()

                    cap.release()

                    FOURCC_MAP={
                        ',mp4': 'mp4v',
                        '.avi': 'XVID',
                    }
                    fourcc_code=FOURCC_MAP.get(file_ext,'mp4v')
                    fourcc=cv2.VideoWriter_fourcc(*fourcc_code)
                    out=cv2.VideoWriter(output_video_path, fourcc, fps, (width,height))
                    if not out.isOpened():
                        print("エラー:VideoWriterを初期化できませんでした。")
                        shutil.rmtree(temp_dir)
                        return
                    try:
                        image_files=sorted(
                            [f for f in os.listdir(temp_dir) if f.startswith('frame_') and f.endswith('.jpg')],
                            key=lambda x: int(x.split('_')[1].split('.')[0])
                        )

                        for image_file in image_files:
                            # print(f"frame:{image_file}")
                            # print(f"type:{type(image_file)}")
                            img_path=os.path.join(temp_dir,image_file)
                            img=cv2.imread(img_path)
                            
                            if img is not None:
                                out.write(img)
                                print(f"img_shape:{img.shape}")
                                print(f"type:{type(img)}")

                            else:
                                print(f"警告:画像{img_path}を読み込めませんでした。")
                    finally:
                        out.release()
                        shutil.rmtree(temp_dir)
        except Exception as e:
            error_str = str(e)
            print(f"Error in predict: {error_str}")
            if 'CUDA' in error_str or 'out of memory' in error_str:
                raise CudaOutOfMemoryError(error_str) from e
            else:
                raise ExceptionError(error_str) from e

        try:
            if flag == False and file_type == "image":               
                # 画像の処理
                save_path=os.path.join(output_dir,result_name)
                os.makedirs(save_path)
                output_csv=f"{output_dir}/{result_name}/{result_name}.csv"
                header=["filename", "label", "confidence"]
                results = self.model.predict(source=str(input_dir), save=False, project=output_dir, name=result_name)
                for result in results:
                    file_name=Path(result.path).name
                    top1=result.probs.top1
                    top1_conf=result.probs.top1conf.item()
                    label_name=self.model.names[top1]
                    frame_filename=os.path.join(save_path,f'{file_name}')
                    rows.append([file_name, label_name, top1_conf])
                    # frame=cv2.imread(result.path)
                    frame=result.plot(txt_color=(0,255,0))
                    cv2.imwrite(os.path.join(save_path,file_name),frame)

                    # rows.append([file_name, label_name, top1_conf])

                with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
                    writer=csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(rows)
        except Exception as e:
            error_str = str(e)
            print(f"Error in predict: {error_str}")
            if 'CUDA' in error_str or 'out of memory' in error_str:
                raise CudaOutOfMemoryError(error_str) from e
            else:
                raise ExceptionError(error_str) from e

        try:
            if flag == False and file_type == "movie":   
                    # 動画の処理
                temp_dir='temp_frames'
                os.makedirs(temp_dir,exist_ok=True)
                os.makedirs(os.path.join(output_dir,result_name))
                for file in files:
                    file_path = str(file)
                    file_name = file.name
                    file_ext = file.suffix.lower()
                    
                    output_csv=f"{output_dir}/{result_name}/{result_name}.csv"
                    header=["flame", "label", "confidence"]
                    output_video_path = f"{output_dir}/{result_name}/{file_name}"

                    cap = cv2.VideoCapture(str(file_path))
                    if not cap.isOpened():
                        print(f"Failed to open video: {file_name}")
                        return 

                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    frame_idx=0
                    while(cap.isOpened()):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # print(f"frame_size:{frame.shape}")
                        results = self.model.predict(frame,verbose=False)
                        annotated_frame=results[0].plot(txt_color=(0,255,0))
                        # print(f"after_frame_size:{annotated_frame.shape}")
                        annotated_frame=annotated_frame.copy()
                        frame_filename=os.path.join(temp_dir,f'frame_{frame_idx}.jpg')
                        top1 = results[0].probs.top1
                        top1_conf = results[0].probs.top1conf.item()
                        label_name = self.model.names[top1]
                        rows.append([frame_idx,label_name,top1_conf])
                        cv2.imwrite(frame_filename, annotated_frame)

                        # top1 = results[0].probs.top1
                        # top1_conf = results[0].probs.top1conf.item()
                        # label_name = self.model.names[top1]
                        # rows.append([frame_idx,label_name,top1_conf])
                        frame_idx+=1

                    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
                        writer=csv.writer(f)
                        writer.writerow(header)
                        writer.writerows(rows)

                    cap.release()

                    FOURCC_MAP={
                        ',mp4': 'mp4v',
                        '.avi': 'XVID',
                    }
                    fourcc_code=FOURCC_MAP.get(file_ext,'mp4v')
                    fourcc=cv2.VideoWriter_fourcc(*fourcc_code)
                    out=cv2.VideoWriter(output_video_path, fourcc, fps, (width,height))
                    if not out.isOpened():
                        print("エラー:VideoWriterを初期化できませんでした。")
                        shutil.rmtree(temp_dir)
                        return
                    try:
                        image_files=sorted(
                            [f for f in os.listdir(temp_dir) if f.startswith('frame_') and f.endswith('.jpg')],
                            key=lambda x: int(x.split('_')[1].split('.')[0])
                        )

                        for image_file in image_files:
                            # print(f"frame:{image_file.shape}")
                            # print(f"type:{type(image_file)}")
                            img_path=os.path.join(temp_dir,image_file)
                            img=cv2.imread(img_path)
                            if img is not None:
                                out.write(img)
                            else:
                                print(f"警告:画像{img_path}を読み込めませんでした。")
                    finally:
                        out.release()
                        shutil.rmtree(temp_dir)

        #     return results
        except Exception as e:
            error_str = str(e)
            print(f"Error in predict: {error_str}")
            if 'CUDA' in error_str or 'out of memory' in error_str:
                raise CudaOutOfMemoryError(error_str) from e
            else:
                raise ExceptionError(error_str) from e
        

    def train(self, input_dir: Path, output_dir: Path, run_id: str, flag: bool):
        """学習処理
        Args:
            input_dir: 学習データが格納されているディレクトリ
            output_dir: 学習結果が格納されるディレクトリ
        Returns:
            str: 学習完了のメッセージ
        Raises:
            ValueError: 学習データのyamlファイルがない場合
        """
        # 学習データのyamlファイルをリストアップ
        yaml_files = [f for f in input_dir.glob("*.yaml") if f.is_file()]
        # pt_file = next((f for f in input_dir.glob("*.pt") if f.is_file() ), None)
        pt_file = Path(os.getcwd(), f"models/{self.use_model}").stem
        if not yaml_files:
            raise ValueError(f"No YAML configuration file found in the directory: {input_dir}")

        # if pt_file is None:
        #     raise ValueError(f"No model found in the directory: {input_dir}")

        result_name = f"{run_id}_{pt_file}_学習結果"
        yaml_path = str(yaml_files[0])
        yaml_para = str(yaml_files[1])
        cfg = yaml_para
        project_dir = output_dir.resolve()
        with open(cfg,'r') as stream:
            cfg_args=yaml.safe_load(stream)
        try:
            if flag == True:
                self.model.train(data=yaml_path, name=result_name, project=project_dir, **cfg_args)
            if flag == False:
                self.model.train(data=input_dir, name=result_name, project=project_dir, **cfg_args)
            
            return f"Training completed. Results saved in {output_dir / 'result_data'}"
        except Exception as e:
            error_str = str(e)
            print(f"Error in predict: {error_str}")
            if 'CUDA' in error_str or 'out of memory' in error_str:
                raise CudaOutOfMemoryError(error_str) from e
            else:
                raise ExceptionError(error_str) from e
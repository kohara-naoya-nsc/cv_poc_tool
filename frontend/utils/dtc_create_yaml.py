import os
import yaml
from tempfile import NamedTemporaryFile
import streamlit as st  
from . import dtc_model_train

def dtc_create_yaml(img_dir_path, val_dir_path, yaml_dir_path,class_text,epochs,batch,save,cache, device, workers, hsv_h, hsv_s, hsv_v, degrees, flipud, fliplr, result_name, run_id,translate,scale,shear,perspective,mosaic,mixup,copy_paste):
    #train = os.path.dirname(file_path)
    #val = os.path.dirname(text_path)
    yaml_dir_path=os.path.dirname(img_dir_path)

    lines = class_text.split('\n')
    filter_lines = [line for line in lines if line]
    # st.write(f"lines:{filter_lines}")
    line_count = len(filter_lines)
    yaml_data = {
        "path": f"/app/shared/train/{run_id}/pre_data/",
        "train": "train/images",
        "val": "val/images",
        "nc": line_count,
        "name": filter_lines,
    }
    yaml_para ={
        "epochs": epochs,
        "batch": batch,
        "save": save,
        "cache": cache,
        "device": device,
        "workers": workers,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "degrees": degrees,
        "flipud": flipud,
        "fliplr": fliplr,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "perspective": perspective,
        "mosaic": mosaic,
        "mixup": mixup,
        "copy_paste": copy_paste,
    }

    data_yaml_path=os.path.join(yaml_dir_path,"yaml_data.yaml")
    para_yaml_path=os.path.join(yaml_dir_path,"yaml_para.yaml")
    with open(data_yaml_path,"w") as data_yaml:
        yaml.dump(yaml_data,data_yaml)
    with open(para_yaml_path,"w") as para_yaml:
        yaml.dump(yaml_para,para_yaml)
    # with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_data_yaml:
    #     with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_para_yaml:
    #         yaml.dump(yaml_data, temp_data_yaml)
    #         temp_data_yaml.flush()
    #         # st.write(temp_data_yaml)
    #         temp_data_yaml_path = temp_data_yaml.name

    #         yaml.dump(yaml_para, temp_para_yaml)
    #         temp_para_yaml.flush()
    #         # st.write(temp_para_yaml)
    #         temp_para_yaml_path = temp_para_yaml.name

            #yolov8の学習実行
    # result = dtc_model_train.model_train(model,temp_data_yaml_path, temp_para_yaml_path, result_name)

    # return result


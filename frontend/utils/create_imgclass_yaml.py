import os
import yaml
import pandas
from tempfile import NamedTemporaryFile
import streamlit as st  
from . import cls_model_train

def create_imgclass_yaml(img_dir_path,val_dir_path,epochs,batch,save,cache, device, workers, hsv_h, hsv_s, hsv_v, degrees, flipud, fliplr, result_name,yaml_dir_path,translate,scale,shear,perspective,mosaic,mixup,copy_paste):
    #train = os.path.dirname(file_path)
    #val = os.path.dirname(text_path)
    # path = os.path.dirname(img_dir_path)
    # temp_data_path = os.path.basename(img_dir_path)
    # temp_para_path = os.path.basename(val_dir_path)
    # filter_lines = [line for line in csv_lavel if line]
    # st.write(f"lines:{filter_lines}")
    # line_count = len(filter_lines)
    yaml_data = {
        "path": f"/app/{yaml_dir_path}",
        # "train": "/train",
        # "val": "/val",
        # "train": f"/app/{img_dir_path}",
        # "val": f"/app/{val_dir_path}",
        # "nc": line_count,
        # "name": filter_lines,
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
            # print(f"temp_para_yaml_path:{temp_para_yaml_path}")
            #yolov8の学習実行
            # try:
            #     result = cls_model_train.cls_model_train(model, temp_data_yaml_path, temp_para_yaml_path, result_name, path)
            # except Exception as e:
            #     print(f"****Exception****:{e}")


    # return result

    # yaml_dir=f"{yaml_dir_path}/data.yaml"
    # with open(yaml_dir,'w') as file :
    #     yaml.dump(yaml_data, file, default_flow_style=False)
            

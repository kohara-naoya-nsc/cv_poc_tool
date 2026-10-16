import streamlit as st  
import yaml
from tempfile import NamedTemporaryFile
# from ultralytics import YOLO


# def train_model(model,temp_data_yaml_path,cfg_args,result_name,path):
#     # model.train(data=temp_data_yaml_path, name=result_name, **cfg_args)
#     # model.train(data="/home/itp0139/work/streamlit-app/app/test/class_20250422_101102", name=result_name, **cfg_args)
#     result = model.train(data=path, name=result_name, **cfg_args)
#     return result

def cls_model_train(model, temp_data_yaml_path, temp_para_yaml_path, result_name, path):
    # print("temp_yaml_path",temp_data_yaml_path)
    # Streamlitセッション状態の初期化
    # if 'logs' not in st.session_state:
    #     st.session_state.logs = []
    # if 'progress' not in st.session_state:
    #     st.session_state.progress = 0

    # progress_bar = st.progress(0)
    # log_text = st.empty()
    # dataset_path=path

    cfg = temp_para_yaml_path
    with open(cfg,'r') as stream:
        cfg_args=yaml.safe_load(stream)
        # print(f"cfg_args:{cfg_args}")
        result = model.train(data=path, name=result_name, **cfg_args)
    # train_thread = threading.Thread(target=train_model, args=(model,temp_data_yaml_path,cfg_args,result_name,path))
    # train_thread.start()

    # result = train_model(model,temp_data_yaml_path,cfg_args,result_name,path)

    # while train_thread.is_alive():
    #     epoch = len(st.session_state.logs) + 1
    #     total_epochs = temp_para_yaml_path.epochs
    #     progress = (epoch / total_epochs) * 100
    #     log_line = f"Epoch {epoch}/{total_epochs} - Progress: {progress:.2f}%"
    #     st.session_state.logs.append(log_line)
    #     st.session_state.progress = progress
    #     progress_bar.progress(int(st.session_state.progress))
    #     log_text.write("\n".join(st.session_state.logs))
        
    # st.session_state.progress >=100
    # progress_bar.progress(100)

    return result
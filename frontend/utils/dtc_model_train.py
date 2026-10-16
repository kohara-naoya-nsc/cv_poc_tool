# import streamlit as st  
# import threading
# import yaml
# from tempfile import NamedTemporaryFile
# from ultralytics import YOLO


# # def train_model(model,temp_data_yaml_path,cfg_args,result_name):
# #     model.train(data=temp_data_yaml_path, name=result_name, **cfg_args)

# def model_train(model, temp_data_yaml_path, temp_para_yaml_path, result_name):
#     # print("temp_yaml_path",temp_data_yaml_path)
#     # Streamlitセッション状態の初期化
#     # if 'logs' not in st.session_state:
#     #     st.session_state.logs = []
#     # if 'progress' not in st.session_state:
#     #     st.session_state.progress = 0

#     # progress_bar = st.progress(0)
#     # log_text = st.empty()

#     cfg = temp_para_yaml_path
#     with open(cfg,'r') as stream:
#         cfg_args=yaml.safe_load(stream)
#         result = model.train(data=temp_data_yaml_path, name=result_name, **cfg_args)
#         # train_thread = threading.Thread(target=train_model, args=(model,temp_data_yaml_path,cfg_args,result_name))
#         # train_thread.start()
        
#     return result
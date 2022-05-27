"""
in terminal run: streamlit run main.py
in another terminal run: mlflow ui
"""

import streamlit as st
from modelstore import ModelStore
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
from st_material_table import st_material_table
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
# from bokeh.plotting import figure
import streamlit_authenticator as stauth

plt.style.use('dark_background')
import shutil
import time
import seaborn as sns
# from bokeh.plotting import figure
import json
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error 
import os
import datetime
import pycaret.classification as pc_cl
import pycaret.regression as pc_rg
import mlflow
import webbrowser
import logging
import shutil
from functions import get_timeseries_predict


def save_model(st,final_model_info='final_model_info.json'):

    with open(final_model_info,'r') as f:
        final_model_info=json.load(f) 

    with st.expander('当前模型信息'):
        st.write(final_model_info)
    
    col1,col2,col3,col4=st.columns(4)
    with col1:  
        path=st.selectbox('保存模型到',['生产环境','测试环境','不保存'])          
    savebutton=st.button('确认')
    return savebutton,path,final_model_info


def move_model_callback(model_store,filename,dest_env,model_name,model_id,ori_env="开发环境",info=True):
    
    final_model=model_store.load('./'+filename[0:-4]+'/'+ori_env+'/'+model_name, 
                                 model_id=model_id)
     
    meta_data = model_store.upload('./'+filename[0:-4]+'/'+dest_env+'/'+model_name, model=final_model)
    
    if info:
        st.info(model_name+'_模型已保存到'+dest_env)
    return meta_data
    
def get_model_training_logs(n_lines = 10):
    file = open('logs.log', 'r')
    lines = file.read().splitlines()
    file.close()
    return lines[-n_lines:]

# @st.cache(suppress_st_warning=True)
def get_data(filename_path='./recent_file_name.txt'):
    with open(filename_path) as file_object:
        filename_newest = file_object.read()
        try:
            data=pd.read_csv('./data/'+filename_newest,index_col=0,parse_dates=True).fillna(0)
        except:
            st.info(filename_newest+"不存在,请重新上传文件")
            return None,None
        
    return data,filename_newest

def log_records(st, **state):
    n_lines = st.number_input(label='设置行数',value=30,format='%d')
    # print(n_lines)
    with st.expander("系统日志",expanded =True):
        logs = get_model_training_logs(n_lines=n_lines)
        st.write(logs)  
        
@st.cache(suppress_st_warning=True)
def data_process_cache(filename):
    data=pd.read_csv('./data/'+filename,index_col=0).fillna(0)       
    return data,filename  
      

def get_current_file_name(recent_file_path='./recent_file_name.txt',ifprint=True):       
    with open(recent_file_path) as file_object:
        filename = file_object.read() 
    if ifprint:
        st.caption('当前文件：          '+filename)
    return filename
       

ML_TASK_LIST = ['回归', '分类','时间序列回归']
RG_MODEL_LIST = ['lr', 'svm', 'rf', 'xgboost', 'lightgbm','allmodels']
RGT_MODEL_LIST = ['lr', 'svm', 'rf', 'xgboost', 'lightgbm','lstm','allmodels']
CL_MODEL_LIST = ['lr', 'dt', 'svm', 'rf', 'xgboost', 'lightgbm', 'resnet','allmodels']

option_style={ "container": {"padding": "5!important",
                             "background-color": "#0E1117"}, 
               "icon": {"color": "orange", "font-size": "25px"},
               "nav-link": {"font-size":                                                                                                   "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"}, 
               "nav-link-selected": {"background-color": 
                                     "gray"}}    
model_store = ModelStore.from_file_system("./model_dir")
        
def data_process(st, **state):
    
    '''数据处理部分'''            
    choose = option_menu(None, ["数据上传", "数据报告","数据筛选"],
                         default_index=0,
                         icons=['cloud-upload', 'info-circle', "funnel-fill"],
                         orientation="horizontal",
                         styles=option_style)                     
    
    if choose=="数据上传":

        uploaded_file = st.file_uploader('训练集数据.csv') 
        if uploaded_file != None:
            with open('./recent_file_name.txt', 'w') as rfn:
                rfn.write(uploaded_file.name)                                        
                
        st.markdown("") 
        new_filename_button=st.checkbox("更改文件名称(optional)")
        if new_filename_button:
            col1,col2=st.columns(2)            
            with col1:
                new_filename=st.text_input("更改文件名称:")           
        
        uploaddata_confirm=st.button("确认")
        if uploaddata_confirm:
            dataframe,filename = get_data()

                      
            if new_filename_button and new_filename !=None:                
                with open('./recent_file_name.txt', 'w') as rfn:
                    rfn.write(new_filename+'.csv')                 
                dataframe.to_csv("./data/"+new_filename+'.csv') 
                filename=new_filename
            if new_filename_button and new_filename ==None:
                st.info("请输入新的文件名称")
                                
            if not new_filename_button:
                pass
            st.success("保存成功")
        dataframe,filename = get_data() 
        st.caption('当前文件：          '+filename) 
        
        dataframe_ori=dataframe.copy()
        

        dataframe['datetime']=dataframe.index
        dataframe['datetime'] = dataframe['datetime'].apply(lambda x:x.strftime('%Y-%m-%d %H:%M:%S'))
        dataframe.index=range(len(dataframe))

        dataframe=dataframe[['datetime']+dataframe.columns[0:-1].tolist()]
        with st.expander('数据表内容'):
            st_material_table(dataframe.head(100))

        
        draw_trend=st.checkbox('趋势图',value=True,key='draw_trend')
        
        
        if draw_trend:
            
            from functions import time_set_box
            seldata=time_set_box(dataframe_ori,st)
            mul_sel=st.multiselect('选择列名',dataframe.columns[1:])
            col1,col2=st.columns([1,5])
            height=col1.number_input("设置趋势图高度",value=200)
            if st.checkbox('趋势对比'):
                
                
                origin_trend=st.line_chart(seldata[mul_sel[0]],width=1200,height=height)
                for col in mul_sel[1:]:
                    
                    origin_trend.add_rows(seldata[col])
            else:
                for col in mul_sel:
                    st.line_chart(seldata[col],width=1200,height=height)

               
    if choose=="数据报告":
        filename=get_current_file_name()  
        if st.button('生成数据报告'):
            try:                 
                dataframe,filename=data_process_cache(filename)                
                pr = ProfileReport(dataframe, explorative=True)
                with st.expander("数据探索结果"):
                    st_profile_report(pr)
             
            except:
                st.warning('请先上传数据')
            
    if choose=="数据筛选":
        
        filename=get_current_file_name()                 
        dataframe,filename=data_process_cache(filename)
        
        with st.expander("数据描述"):  
            st.table(dataframe.describe())
            
        with st.expander("数据筛选"): 
            with open('filter_cols.json','r') as f:
                filter_cols_last=json.load(f)
            if dataframe.columns[0] not in filter_cols_last:
                filter_cols_last=dict(zip(dataframe.columns,[1]*len(dataframe.columns)))
                
            filter_cols={}                           
            for col in dataframe.columns:
                filter_cols[col]=st.checkbox(col,value=filter_cols_last[col],key=col)
          
                
        st.caption('已选数据：   ')
        i=0 
        selcols=[]
        for col,selstatus in filter_cols.items():
            if selstatus:
                i=i+1
                selcols.append(col)

        st.markdown(selcols)

        if st.button('保存'):
            with open('filter_cols.json','w') as f:
                json.dump(filter_cols,f,ensure_ascii=False)
                st.success("保存成功")

        filter_dataframe=dataframe[[k for k,v in filter_cols.items() if v]]               

def build_model(st,**state):


    choose = stx.stepper_bar(steps=["Step1:配置环境", "Step2:训练模型", "Step3:我的模型"], is_vertical=0, lock_sequence=False)

    
    filename=get_current_file_name()                 
    dataframe,filename=data_process_cache(filename) 
    
    with open('filter_cols.json','r') as f:
        filter_cols_last=json.load(f)  
    filter_cols=[col for col in filter_cols_last.keys() if filter_cols_last[col]]   
    filter_dataframe=dataframe[filter_cols].copy()
        
    if choose==0: 
        columns=filter_dataframe.columns.tolist() 
        if os.path.exists('model_config.json'):
            with open('model_config.json','r') as f:
                model_config=json.load(f)
        else:
            model_config={"target_col":''}
            with open('model_config.json','w') as f:
                json.dump(model_config,f,indent=4,ensure_ascii=False)
                    
            
        if model_config["target_col"] in columns:
            
            target_col=st.selectbox('1--,选择目标', 
                              columns,
                              index=columns.index(model_config["target_col"]))
            task = st.selectbox('2--,  选择任务', ML_TASK_LIST,
                                index=ML_TASK_LIST.index(model_config["task"]))
            if task == '时间序列回归':
                pre_shift =st.slider('预测未来第几个采样点?', 1, 130, 
                                     value=model_config["pre_shift"])
            
            # model = st.selectbox('3--,选取模型',
            #                       RG_MODEL_LIST, 
            #                       index=RG_MODEL_LIST.index(model_config["model"]))
            model_name=st.text_input('3--,  模型命名',value=model_config["model_name"])
        
        else:
            target_col=st.selectbox('1--,选择目标',columns)
            task = st.selectbox('2--,  选择任务', ML_TASK_LIST)
            # if task == '回归':
            #     model = st.selectbox('3--,选取模型', RG_MODEL_LIST)
            if task == '时间序列回归':
                # model = st.selectbox('3--,选取模型', RGT_MODEL_LIST)
                pre_shift =st.slider('预测未来第几个采样点?', 1, 130, 5)
            # if task == '分类':
            #     model = st.selectbox('3--,选取模型', RG_MODEL_LIST)
            model_name=st.text_input('3--,  模型命名')
        if st.button('保存'):
 
            pre_shift=pre_shift if task == '时间序列回归' else 0
            model_config={
                          "task": task,
                          "target_col": target_col,
                          "pre_shift": pre_shift,
                          "model_name": model_name}
            
            with open('model_config.json','w') as f:
                json.dump(model_config,f,indent=4,ensure_ascii=False)
            with open(filename[0:-4]+model_name+'model_config.json','w') as f:
                json.dump(model_config,f,indent=4,ensure_ascii=False)
                st.success("保存成功")
            
    if choose==1:
        
        filename=get_current_file_name(ifprint=False)                 
        dataframe,filename=data_process_cache(filename) 
        
        with open('filter_cols.json','r') as f:
            filter_cols_last=json.load(f) 
            
        filter_cols=[col for col in filter_cols_last.keys() if filter_cols_last[col]]   
        filter_dataframe=dataframe[filter_cols].copy()
        
        if os.path.exists('model_config.json'):
            with open('model_config.json','r') as f:
                model_config=json.load(f)
                st.caption("当前模型名称："+model_config['model_name'])
                # st.info(model_config)
        else:
            st.info("请先给模型命名")
        st.write("")
                   
        
        model_name=model_config["model_name"] 
        # model=model_config["model"]
        model="allmodels"
        task=model_config["task"]       
        target_col=model_config["target_col"]
        pre_shift=model_config["pre_shift"]
        start_train= st.button('开始训练')           
        if start_train and target_col is not None and task == '回归' : 
            st.success('模型训练开始......') 
            times=time.time()
            pc_rg.setup(
                filter_dataframe,
                target=target_col,
                log_experiment=True,
                experiment_name='ml_',
                log_plots=True,
                silent=True,
                fold_shuffle=True,
                train_size = 0.7,
                verbose=False,
                profile=True,
                html=False)                    
            
            if model !='allmodels':
                best_model=pc_rg.create_model(model, verbose=False)
                # pc_rg.evaluate_model(best_model)
            if model =="allmodels":
                
                best_model=pc_rg.compare_models(fold=2)
                st.write(pc_rg.pull())
                # best_model=pc_rg.create_model(best_model, verbose=False)
                
                
            #eval
            pre_df=pc_rg.predict_model(best_model)
            
            r2=r2_score(pre_df[target_col],pre_df['Label']) 
            rmse=mean_squared_error(pre_df[target_col],pre_df['Label'],squared=False)
            mae=mean_absolute_error(pre_df[target_col],pre_df['Label'])


            #save model                                                                                           
            final_model=pc_rg.finalize_model(best_model)  
            st.write(final_model)
            final_model_info = model_store.upload("./"+filename[0:-4]+"/开发环境/"+model_name, 
                                                  model=final_model)
            timed=time.time()
            col1,col2,col3,col4,col5=st.columns(5)
            with col1:
                st.caption("最终模型选择："+final_model_info['model']['model_type']["type"])
            with col2:
                st.caption('耗时:'+str(round(timed-times,2))+'s')

            col1,col2,col3,col4,col5=st.columns(5)
            with col1:
                st.caption('模型评价:')
            with col2:
                st.write('r2: ',round(r2,4),'rmse: ',round(rmse,4),'mae: ',round(mae,4))                                      
            # st.write(final_model_info)
            with open('final_model_info.json','w') as f:
                json.dump(final_model_info,f,ensure_ascii=False)           
            st.success('模型已经创建')
          
        elif start_train and target_col is not None and task == '分类':  
            st.success('模型训练开始......')                
            pc_cl.setup(
                filter_dataframe,
                target=target_col,
                log_experiment=True,
                experiment_name='ml_',
                log_plots=True,
                silent=True,
                fold_shuffle=True,
                train_size = 0.7,
                verbose=False,
                profile=True,
                html=False)                    
            
            if model !='allmodels':
                best_model=pc_cl.create_model(model, verbose=False)
                # pc_cl.evaluate_model(best_model)
                st.write('classification_report')
                pre_df=pc_cl.predict_model(best_model)
                # st.write(pre_df)
                cls_report=classification_report(pre_df[target_col].astype(int),
                                      pre_df['Label'].astype(int),output_dict=1)
                st.table(pd.DataFrame(cls_report).T)

            else:
                best_model=pc_cl.compare_models()
                st.write(pc_cl.pull()) 
                
             
            final_model=pc_cl.finalize_model(best_model)                                      
            #eval
            pre_df=pc_cl.predict_model(best_model)
           
            #save model                                                                                           
            final_model=pc_cl.finalize_model(best_model)            
            final_model_info = model_store.upload(model_name+"开发环境", 
                                                  model=final_model)                                       

        
        elif start_train and target_col is not None and task == '时间序列回归':
            from functions import create_traindata_with_ypre,create_diff_fea
            
            
            shift_dataframe,target_name_list=create_traindata_with_ypre(filter_dataframe,
                                                         pre_n_minutes=pre_shift,
                                                         y_name='总降需量',
                                                         dropna=1)

            shift_dataframe_withfea=create_diff_fea(shift_dataframe,
                                                    target_name_list,
                                                    diff_periods=10,
                                                    use_past_fea=0)
            model="lightgbm"    
            st.success('模型训练开始......') 
            final_model_info_dict={}
            for target_name in target_name_list:
                st.caption(target_name)
                times=time.time()
                traindata=pd.concat([shift_dataframe_withfea.drop(target_name_list,axis=1),
                                     shift_dataframe_withfea[[target_name]]],axis=1).astype('float').dropna()
                # st.write(traindata[target_name])
                # pd.DataFrame(traindata).to_csv('traindata.csv')
                pc_rg.setup(
                    traindata,
                    target=target_name,
                    log_experiment=True,
                    experiment_name='ml_',
                    log_plots=True,
                    silent=True,
                    fold_shuffle=True,
                    train_size = 0.7,
                    verbose=False,
                    profile=True,
                    html=False) 
                best_model=pc_rg.create_model('lightgbm', verbose=False)                
            
                if model !='allmodels':
                    best_model=pc_rg.create_model(model, verbose=False)
                    # pc_rg.evaluate_model(best_model)
                if model =="allmodels":
                    
                    best_model=pc_rg.compare_models(fold=2)
                    st.write(pc_rg.pull())

                
                
                # eval
                pre_df=pc_rg.predict_model(best_model)
                # st.write(pre_df)
                
                r2=r2_score(pre_df[target_name],pre_df['Label']) 
                rmse=mean_squared_error(pre_df[target_name],pre_df['Label'],squared=False)
                mae=mean_absolute_error(pre_df[target_name],pre_df['Label'])

    
                #save model                                                                                           
                final_model=pc_rg.finalize_model(best_model)  
                
                final_model_info = model_store.upload("./"+filename[0:-4]+"/开发环境/"+model_name, 
                                                      model=final_model)
                
                # st.write(target_name)
                timed=time.time()
                # col1,col2=st.columns(2)
                # with col1:
                #     st.caption("最终模型选择："+final_model_info['model']['model_type']["type"])
                # with col2:
                st.caption('耗时:'+str(round(timed-times,2))+'s')

                col1,col2=st.columns(2)
                with col1:
                    st.caption('模型评价:')
                with col2:
                    st.write('r2: ',round(r2,4),'rmse: ',round(rmse,4),'mae: ',round(mae,4)) 
                st.write("")                                     
                # st.write(final_model_info)
                final_model_info_dict[target_name]=final_model_info
            with open('final_model_info.json','w') as f:
                json.dump(final_model_info_dict,f,indent=4,ensure_ascii=False)           
            st.success('模型已经创建')        
        
        
        
        
        
        savebutton,path,final_model_info=save_model(st)
        
        if path!='不保存' and savebutton:
            if os.path.exists("./model_dir/operatorai-model-store/"+filename[0:-4]+"/"+path+"/"+model_name):
                shutil.rmtree("./model_dir/operatorai-model-store/"+filename[0:-4]+"/"+path+"/"+model_name)

            if pre_shift: 
                model_idname_dict={}
                for final_model_infox in final_model_info:                            
                    
                    model_info=move_model_callback(model_store,filename,path,model_name,final_model_info[final_model_infox]['model']['model_id'],ori_env="开发环境",info=False)               
                    # st.write(model_info['model']['model_id'])
                    model_idname_dict[final_model_infox]=model_info['model']['model_id']
                    time.sleep(1)
                with open('model_info_'+path+model_name+'.json','w') as f:
                    json.dump(model_idname_dict,f,ensure_ascii=False,indent=4) 
                st.info(model_name+'_模型已保存到'+path)
            else:    
                model_info=move_model_callback(model_store,filename,path,model_name,final_model_info['model']['model_id'],ori_env="开发环境")  
        elif path=='不保存' and savebutton:
            st.info('模型未保存')
        
                    
    if choose==2:
        
        filename=get_current_file_name(ifprint=False) 

        
        col1,col2,col3,col4=st.columns(4)
        env_button=st.selectbox('模型路径', ['生产环境','测试环境'])
        model_path="./model_dir/operatorai-model-store/"+filename[0:-4]+'/'+env_button+'/'
        modelname_list=os.listdir(model_path)
        if len(modelname_list)>0:

            model_info={}
            for modelx in modelname_list:
                model_info[modelx]={}
                model_info[modelx]['domain']='./'+filename[0:-4]+'/'+env_button+'/'+modelx
                model_info[modelx]['model_ids']=model_store.list_models('./'+filename[0:-4]+'/'+env_button+'/'+modelx)  

            i=0
            Mov_status={}
            Del_status={}
            Info_status={} 
            
            
            for model_sel in model_info.keys():
                # st.write(model_info[model_sel]['domain'],model_info[model_sel]['model_ids'][0])
                
                model_infox=model_store.get_model_info(model_info[model_sel]['domain'],
                                                      model_info[model_sel]['model_ids'][0])
                with st.container():
                    col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12=st.columns(12)
                    with col1:
                        st.caption(model_sel) 
                    with col2:
                        st.caption('时间:') 
                    with col3:
                        st.caption(model_infox['code']['created']) 
                    with col5:
                        st.caption('type:')
                    with col6:
                        st.caption(model_infox['model']['model_type']['type']) 
                    with col8:
                        st.caption('训练集:')
                    with col9:
                        st.caption(filename) 
                    with col10:
                        Mov_status[model_sel]=st.button("Mov",key=i)
                    with col11:
                        Del_status[model_sel]=st.button("Del",key=i)
                    with col12:
                        Info_status[model_sel]=st.button("Info",key=i)                    
                        i=i+1
                        
                        

                
                
            for model_sel in model_info.keys():
                path_folder="./model_dir/operatorai-model-store/"+filename[0:-4]+"/"
               
                if Mov_status[model_sel]:
                    # st.write(model_info[model_sel]['model_ids'])
                    if env_button=='生产环境':
                        with open('model_info_生产环境'+model_sel+'.json','r') as f:
                            model_idname_info=json.load(f)
                        
                        # for model_id in model_info[model_sel]['model_ids']:
                        model_idname_dict={}
                        for target_name in model_idname_info.keys():   
                            # st.write(1,target_name)
                            model_infox=move_model_callback(model_store,
                                                    filename,
                                                    "测试环境",
                                                    model_sel,
                                                    model_idname_info[target_name],
                                                    ori_env=env_button,
                                                    info=False)
                            # st.write(model_info)
                            model_idname_dict[target_name]=model_infox['model']['model_id']
                            time.sleep(1)
                            
                        with open('model_info_测试环境'+model_sel+'.json','w') as f:
                            json.dump(model_idname_dict,f,ensure_ascii=False,indent=4) 
                        st.info(model_sel+'_模型已保存到'+"测试环境")
                        
                        shutil.rmtree(path_folder+env_button+"/"+model_sel)


                    else:
                        # for model_id in model_info[model_sel]['model_ids']:
                        with open('model_info_测试环境'+model_sel+'.json','r') as f:
                            model_idname_info=json.load(f)
                        model_idname_dict={}
                        for target_name in model_idname_info.keys():     
                            
                            model_infox=move_model_callback(model_store,
                                                            filename,
                                                            "生产环境",
                                                            model_sel,
                                                            model_idname_info[target_name],
                                                            ori_env=env_button,
                                                            info=False)
                            # st.write(model_info)
                            model_idname_dict[target_name]=model_infox['model']['model_id']
                            time.sleep(1)
                        with open('model_info_生产环境'+model_sel+'.json','w') as f:
                            json.dump(model_idname_dict,f,ensure_ascii=False,indent=4) 
                        st.info(model_sel+'_模型已保存到'+'生产环境')   
                            
                        shutil.rmtree(path_folder+env_button+"/"+model_sel)

                if Del_status[model_sel]:
                    shutil.rmtree(path_folder+env_button+"/"+model_sel)
                    st.info("已删除"+env_button+model_sel)
                if Info_status[model_sel]:
                    for model_id in model_info[model_sel]['model_ids']:
                        st.write(model_store.get_model_info(model_info[model_sel]['domain'],
                                                          model_id))
            
            if st.checkbox('模型评估'):
                import plotly_express as px
                datasource=st.file_uploader('上传测试文件')
                modelname_list=os.listdir(model_path)
                model_name_=st.selectbox("选择模型", modelname_list)
                
                if st.button('开始评估'):                
                    pre_data_all,target_name_list,model_online,model_config,dataframe=\
                    get_timeseries_predict(model_name_,
                                           filename,
                                           env_button,
                                           model_store,
                                           datasource,
                                           pc_rg)
                    dataframe.index=pd.to_datetime(dataframe.index)
                    i=1
                    for col in target_name_list:
                        pre_data_all_col=pre_data_all[col].shift(i)
                        pre_data_all_col.index=pd.to_datetime(pre_data_all_col.index)
                        dataframe=dataframe.join(pre_data_all_col,how='right').dropna()
                        # st.write(dataframe)
                        i=i+1

                        r2=r2_score(dataframe[col[0:-5]],dataframe[col]) 
                        rmse=mean_squared_error(dataframe[col[0:-5]],dataframe[col],squared=False)
                        mae=mean_absolute_error(dataframe[col[0:-5]],dataframe[col])        
                        col1,col2=st.columns(2)
                        with col1:
                            st.caption('模型评价:')
                        with col2:
                            st.write('r2: ',round(r2,4),'rmse: ',round(rmse,4),'mae: ',round(mae,4)) 
                                                                       
                        
                        fig = px.line(
                          dataframe,
                          x=dataframe.index, 
                          y=[col[0:-5],col])
                        fig.update_layout(width=800)
                        st.write(fig) 
                        
                        # fig, ax = plt.subplots(figsize=(7,2), dpi=120)
                        # sns.kdeplot(pre_data[target_col])
                        # sns.kdeplot(pre_data['Label'])
                        # ax.legend(['y_true','y_predict'])
                        # # ax.title('kde')
                        # st.pyplot(fig)     
                            
                            
        
                        
def create_app(st,**state):
    filename=get_current_file_name(ifprint=False) 
    
    choose = option_menu(None, ["预测看板","设备健康","异常判断"],
                         default_index=0,
                          icons=['plus-lg', 'plus-lg', 'plus-lg'], 
                         menu_icon="cast",orientation="horizontal",styles=option_style)
    if choose=="预测看板": 
            app_name=st.text_input('1--,应用命名')
            with st.expander("应用说明"):  
                app_desc=st.text_input('应用描述')
           
            env_button='生产环境'
            model_path="./model_dir/operatorai-model-store/"+filename[0:-4]+'/'+env_button+'/'
            modelname_list=os.listdir(model_path)
            if len(modelname_list)>0:
                uploaded_model=st.selectbox('2--,选择模型',modelname_list)
            else:
                uploaded_model=None
                st.info("生产环境中没有模型存在")

            app_purpose=st.selectbox('3--,应用目的',['拟合','时序预测'])
               

            datasource=st.file_uploader('4--,连接实时数据')   

         
            
            if st.button('5--,创建'):
                import pickle as pkl
                # st.write(app_name,uploaded_model,datasource)
                if app_name !=None and uploaded_model != None and datasource != None and app_purpose!= None:
                    # model_online= pc_rg.load_model('./local_models/'+uploaded_model.name[0:-4])
                    if app_purpose=="拟合":                    
                        domain='./'+filename[0:-4]+'/'+env_button+'/'+uploaded_model
                        model_ids=model_store.list_models('./'+filename[0:-4]+'/'+env_button+'/'+uploaded_model)
                        st.write(model_ids)
                       
                        model_online=model_store.load(domain, 
                                                      model_id=model_ids[0])
                        
                        
                        
                        dataframe=pd.read_csv(datasource,index_col=0).fillna(0)  
                        pre_data=pc_rg.predict_model(model_online,data=dataframe)
                        pre_data.index=pd.to_datetime(pre_data.index)
                        target_col='总降需量'
                        appdict={}
                        appdict['appname']=app_name
                        appdict['model_online']=model_online
                        appdict['dataframe']=dataframe
                        appdict['pre_data']=pre_data                    
                        appdict['target_col']=target_col  
                        appdict['app_desc']=app_desc 
                        with open('./appdict/预测看板/'+app_name,'wb') as f:
                            pkl.dump(appdict,f)
                        st.info(app_name+"应用已创建")
                    else:
                        
                        pre_data_all,target_name_list,model_online,model_config,dataframe=\
                        get_timeseries_predict(uploaded_model,
                                               filename,
                                               env_button,
                                               model_store,
                                               datasource,
                                               pc_rg)


                        appdict={}
                        appdict['appname']=app_name
                        appdict['model_online']=model_online
                        appdict['dataframe']=dataframe
                        appdict['pre_data_all']=pre_data_all                    
                        appdict['target_col']=model_config["target_col"] 
                        appdict['target_col_pre']=target_name_list 
                        appdict['app_desc']=app_desc 
                        
                        
                        with open('./appdict/预测看板/'+app_name,'wb') as f:
                            pkl.dump(appdict,f)
                        st.info(app_name+"应用已创建")

                else:
                    st.warning('请上传文件')
                    
    if choose=="设备健康": 
        st.write("to be done")

                    
    if choose=="异常判断": 
        st.write("to be done")                   
                    
                    
                    
def demo_app(st,**state):
    import os
    import pickle as pkl
                 
    # if choose=="预测看板": 
    apps=os.listdir('./appdict/预测看板/')
    col1,col2,col3,col4 =st.columns(4)
    with col1:
        appx=st.selectbox('选择应用',apps)


    with open('./appdict/预测看板/'+appx,'rb') as f:
        appdict=pkl.load(f)
        
    with st.expander('应用说明'):   
        st.markdown(appdict['app_desc'])
    page = option_menu(None,['实时预测','历史统计','明日预测','报警管理','模型管理'],
                          default_index=0,
                          icons=['diamond', 'diamond', "diamond", "diamond"], 
                          menu_icon="cast",orientation="horizontal",styles=option_style) 

 
    if 'target_col_pre' not in appdict:
        app_name=appdict['appname']
        model_online=appdict['model_online']
        dataframe=appdict['dataframe']
        dataframe.index=pd.to_datetime(dataframe.index)
        pre_data=appdict['pre_data']  
        pre_data.index=pd.to_datetime(pre_data.index)                  
        target_col=appdict['target_col'] 
        
           
        col1,col2,clo3,col4=st.columns(4)
        with col1:
            demo_start=st.number_input('start_time',value=100)
        with col2:
            demo_end=st.number_input('end_time',value=150)
            
        oridata=dataframe[demo_start:demo_end] 
        
        r2=r2_score(pre_data[target_col],pre_data['Label'])  
        columns=dataframe.columns
        col1, col2, col3, col4,col5, col6, col7, col8 = st.columns(8)
        col_dict={}
        col_dict[1],col_dict[2],col_dict[3],col_dict[4],\
        col_dict[5],col_dict[6],col_dict[7],col_dict[8]=\
        col1, col2, col3, col4 ,col5, col6, col7, col8
        
        for i in range(len(columns)):
            with col_dict[i+1]:
                st.metric(columns[i],round(dataframe[columns[i]][demo_end],3))
                 
        with col_dict[8]:
            st.metric('模型可信度：',0.94)      
        
        my_chart = st.line_chart(oridata[target_col])
        my_chart.add_rows(pre_data['Label'][demo_start:demo_end])  
    else:
        app_name=appdict['appname']
        dataframe=appdict['dataframe']
        dataframe.index=pd.to_datetime(dataframe.index)
        pre_data_all=appdict['pre_data_all']                  
        target_col=appdict['target_col']  
        target_name_list=appdict['target_col_pre']  
        
                                                   
        col1,col2,col3,col4 =st.columns(4)                              
        col1,col2,col3,col4,col5,col6,col7=st.columns(7)    
        
        if page=="实时预测":
            with col1:

                demo_datestart=st.date_input('开始日期', 
                                             datetime.date(2021, 6, 6),
                                             min_value=datetime.date(2021, 6, 6),
                                             max_value=datetime.date(2021, 7, 1))                
        
            with col2:
                demo_timestart=st.time_input('开始时间', datetime.time(12, 0, 0))                
            with col3:    
                demo_dateend=st.date_input('结束日期', 
                                           datetime.date(2021, 6, 6),
                                           min_value=datetime.date(2021, 6, 6),
                                           max_value=datetime.date(2021, 7, 1))
                
            with col4:
                demo_timeend=st.time_input('结束时间', datetime.time(12, 15, 0))
            with col6:
                uplimit=st.number_input('当前需量控制上限',value=130)               
            demo_start=str(demo_datestart)+str(" ")+str(demo_timestart)
            demo_end=str(demo_dateend)+str(" ")+str(demo_timeend)
            demo_date={}
            demo_date['demo_datestart']=str(demo_datestart)
            demo_date['demo_timestart']=str(demo_timestart)
            demo_date['demo_dateend']=str(demo_dateend)
            demo_date['demo_timeend']=str(demo_timeend)
            with open('demo_date.json','w') as f:
                json.dump(demo_date,f)
            oridata=dataframe[demo_start:demo_end]
        
        
        if page=='实时预测':
            col1,col2,col3,col4,col5, col6, col7 =st.columns(7)
            with col1:
                st.metric("截止目前本月电费(万)",1720,delta=str(round(-80/1800*100,2))+'%',delta_color ="inverse")
            with col2:    
                st.metric("上月电费(万)",1800)
            with col3:    
                st.metric("本月需量实际值(MW)",130,delta=str(round(-10/145*100,2))+'%',delta_color ="inverse")                
            with col4:    
                st.metric("上月需量实际值(MW)",145) 


            
            # r2={}
            # for target_name_pre in target_name_list:
            #     r2[target_name_pre]=r2_score(pre_data[target_name_pre],pre_data_all[target_name_pre])  
            columns=dataframe.columns
            col1, col2, col3, col4,col5, col6, col7 = st.columns(7)
            col_dict={}
            col_dict[1],col_dict[2],col_dict[3],col_dict[4],\
            col_dict[5],col_dict[6],col_dict[7]=\
            col1, col2, col3, col4 ,col5, col6, col7
            
            for i in range(len(columns)):
                with col_dict[i+1]:
                    st.metric(columns[i],round(dataframe[columns[i]][demo_end],3))
            
            with col_dict[6]:
                 st.metric('近一周模型精确度：',0.96)                         
            with col_dict[7]:
                st.metric('模型可信度：',0.94)      
          

            pre_list=[oridata[target_col].values[-1]]
            for target_name in target_name_list:
                pre_list.append(pre_data_all[target_name][demo_start:demo_end].iloc[-1][0])
                            
            index_pre=[oridata.index[-2]+pd.Timedelta(str(i+1)+'min') for i in range(len(pre_list))]

            predata=pd.DataFrame(pre_list,index=index_pre,columns=[target_name[0:-1]])
            
            
            
            uplimit_frame=pd.DataFrame([uplimit]*(len(oridata)+len(target_name_list)),
                                       columns=['需量上限'],
                                       index=oridata.append(predata[1:]).index).astype('float64')

            col1, col2 = st.columns([1,3])
            col2.subheader(target_name[0:-5]+"预测看板（mW）")                          
            my_chart = st.line_chart(uplimit_frame)
            my_chart.add_rows(predata)
            my_chart.add_rows(oridata)
                                                   
                        
            col1,col2,col3=st.columns([1,5,3])
            col1.markdown("报警列表：")
            over=predata[1:].max().values[0]>uplimit
            
            if over:
                i=0
                for x in predata.iloc[1:].values:
                    i=i+1
                    if x >uplimit:
                        col2.warning("需量"+str(i)+"分钟后即将超限")                        
                        break 
            else:
               col2.info('None')
               
            col1,col2,col3=st.columns([1,5,3])   
            col1.markdown("调控建议:")
            if over:
                over_value=max(predata.iloc[1:].values-uplimit)
                if over_value>0:
                    devices=oridata.drop(target_col,axis=1).tail(1)
                    # st.write(devices)
                    for col in devices.columns:
                        if devices[col].values>over_value: 
                            down=devices[col].values[0]-over_value
                            
                            col2.warning("请将"+col+"功率下调至"+str(round(down[0],1))+"(mW)")
                            
                            break
            else:
                col2.info('None')
                
                
            col1,col2,col3=st.columns([1,5,3])
            col1.markdown("超限原因：")
            if over:
                  col2.warning("设备1需量过去15min突然上升了10mW")     
            else:
                col2.info('None')            
                   
        
        elif page=='历史统计': 
            import time
            with open('demo_date.json','r') as f:
                demo_date=json.load(f)
            
            demo_dateend= time.strptime(demo_date['demo_dateend'], "%Y-%m-%d")
            demo_dateend=datetime.date(demo_dateend[0],demo_dateend[1],demo_dateend[2])
            

            for i in range(3):
                st.write("")
            col1,col2,col3,col4,col5,col6=st.columns(6)
            with col1: 
                demo_months=st.date_input('选择开始月份', 
                                           demo_dateend,
                                           min_value=datetime.date(2021, 6, 6),
                                           max_value=datetime.date(2021, 7, 1),
                                           key=1)
            with col2:                            
                demo_monthe=st.date_input('选择结束月份', 
                                           demo_dateend,
                                           min_value=datetime.date(2021, 6, 6),
                                           max_value=datetime.date(2021, 7, 1),
                                           key=3)
            
            col1,col2=st.columns([2,5])
            col2.write("日需量最大值统计")                                            
            demand_date=dataframe[str(demo_months)[0:-3]:str(demo_monthe)[0:-3]].resample('1D').max()
            col1,col2,col3=st.columns([5,1,2])
            with col1:
                st.bar_chart(demand_date,
                            width=800,height=300,use_container_width=False)
            with col3:
                limit=st.number_input('设置需量上限',value=150,key='limit')
                plt.rcParams['font.sans-serif'] = ['SimHei']
                fig=plt.figure(figsize=(20,13))
                sns.distplot(demand_date[target_col],color='yellow')
                plt.xticks(fontname="Calibri",fontsize=50,rotation=45)
                plt.yticks(fontname="Calibri",fontsize=50)
                plt.axvline(limit,color='red',linewidth=10)
                
                plt.annotate(limit, xy=(limit, 0.025), xytext=(limit, 0.025),fontsize=80)
                plt.xlabel("总降需量在各个区间的分布情况",fontsize=80)
                st.write(fig)
            
            
            col1,col2,col3,col4,col5,col6=st.columns(6)
            with col1:                      
                demo_days=st.date_input('选择开始日期', 
                                            demo_dateend,
                                            min_value=datetime.date(2021, 6, 6),
                                            max_value=datetime.date(2021, 7, 1),
                                            key=2)  
            with col2:                      
                demo_daye=st.date_input('选择结束日期', 
                                            demo_dateend,
                                            min_value=datetime.date(2021, 6, 6),
                                            max_value=datetime.date(2021, 7, 1),
                                                key=4)
            col1,col2=st.columns([2,5])
            col2.write("小时需量最大值统计")
            demand_hour=dataframe[str(demo_days):str(demo_daye)].resample('1H').max()
            col1,col2,col3=st.columns([5,1,2])
            with col1:
                st.bar_chart(demand_hour,
                            width=800,height=300,use_container_width=False)
            with col3:
                limit=st.number_input('设置需量上限',value=150)
                plt.rcParams['font.sans-serif'] = ['SimHei']
                fig=plt.figure(figsize=(20,13))
                sns.distplot(demand_hour[target_col],color='yellow')
                plt.xticks(fontname="Calibri",fontsize=50,rotation=45)
                plt.yticks(fontname="Calibri",fontsize=50)
                plt.axvline(limit,color='red',linewidth=10)
                
                plt.annotate(limit, xy=(limit, 0.025), xytext=(limit, 0.025),fontsize=80)
                plt.xlabel("总降需量在各个区间的分布情况",fontsize=80)
                st.write(fig)
                
            # plt.pie(freq, labels = freq.index, explode = (0.05, 0, 0), autopct = '%.1f%%', colors = colors, startangle = 90, counterclock = False)

                
        elif page=='明日预测':
            
            plan = st.file_uploader('请上传明日生产计划.xls',key='plan') 
            
            pre24h=dataframe[target_col].resample('1H').max()[0:24]
            pre24h.index=range(1,25)
  
            col1,col2=st.columns([2,5])
            col2.write("明日0-24小时需量最大值预测")  

            st.area_chart(pre24h) 
            col1,col2=st.columns([4,6])
            col2.write("hour")
            
        elif page=='报警管理':
            sel=st.selectbox('界面选择',['超限记录','报警记录','超限原因分析','超限漏报记录'],index=0)
            log = './报警日志/'+sel+'.log'
            for line in open(log,"r",encoding='UTF-8'):           
                st.markdown(line)
                                             
        elif page=='模型管理':
            col1,col2=st.columns([3,4])
            col1.write('*当前模型:')    
            col2.write("电力能耗预测")
            
            col3,col4=st.columns([3,4])            
            col3.write('*当前模型训练效果:') 
            col4.write('r2: 0.98 rmse: 2.77 mae: 3.22')             
            
            if st.checkbox('模型评价(以天为单位进行模型评估)',value=True):
                col1,col2=st.columns([2,6])
                with col1:
                    days=st.number_input('选择天数',value=30)
                mape=pd.DataFrame(np.random.uniform(0.92, 0.995, days),columns=['mape'])
                error=pd.DataFrame(np.random.uniform(-8, 8, days),columns=['error'])
                mae=abs(error)
                col1,col2=st.columns([2,6])
                col2.write('mape=mean((真实值-预测值)/真实值))='+str(round(mape.mean()[0],4)))
    
                my_chartmape = st.line_chart(mape,height=200)
                
                col1,col2=st.columns([2,6])
                col2.write('mae=mean((真实值-预测值)))='+str(round(mae.mean()[0],4)))
                my_charterror = st.line_chart(error,height=200)
            
            filename=get_current_file_name(ifprint=False)                                
            env_button='生产环境'
            model_path="./model_dir/operatorai-model-store/"+filename[0:-4]+'/'+env_button+'/'
            modelname_list=os.listdir(model_path)
            
            modelbutton=st.checkbox('更新模型')
            if len(modelname_list)>0 and modelbutton:
                uploaded_model=st.selectbox('选择模型',modelname_list) 
                if st.button('保存'):
                    st.success('模型已更新')
                
                
            

        
if __name__ == '__main__':
    st.set_page_config(page_title="HCE钢铁行业自动化建模平台", 
                        page_icon="random" , 
                        layout="wide",
                        initial_sidebar_state="auto")
    
    # st.image('./image/future3.png',width =1500,use_column_width =True)   
    st.markdown(""" <style> .font3 {
    font-size:30px ; font-family: 'Cooper Black'; color: red;} 
    </style> """, unsafe_allow_html=True)
    
    st.markdown(""" <style> .font1 {
    font-size:25px ; font-family: 'Cooper Black'; color: #red;} 
    </style> """, unsafe_allow_html=True)
       
    st.markdown('<p class="font3">霍尼韦尔企业智联 | 工厂能耗预测助手</p>', unsafe_allow_html=True)

    st.sidebar.image('./image/honeywell.png',width=150)

    with st.sidebar:    
        choose = option_menu("功能栏", ["数据处理","建立模型", "创建应用","应用列表","系统日志"],                             
                             icons=['cloud-upload-fill',
                                    'gear-fill',
                                    'plus-lg',
                                    'app',
                                    'card-text'], 
                             menu_icon="list", default_index=0)
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: visiable;}
                footer {visibility: hidden;}
                </style>               
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    
    names = ['李洋', '管理员', '霍尼韦尔']
    usernames = ['liyang', 'admin', 'honeywell']
    passwords = ['123456', '654321', '123456']   
    hashed_passwords = stauth.Hasher(passwords).generate()   
    authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
        'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)
    
    name, authentication_status, username = authenticator.login('Login', 'main')   
    if authentication_status:
        with st.container():
            cols1,cols2 = st.columns([6,1])
            with cols2:
                cols2.caption('当前用户: *%s*' % (name))
                authenticator.logout('Logout', 'main')
        
        if choose =="数据处理":
            data_process(st)
            
        if choose =="建立模型":
             
            build_model(st) 
            
        if choose =="创建应用":
            create_app(st)
            
        if choose =="应用列表":
            demo_app(st)        
            
        if choose =="系统日志":
            log_records(st)         
      
    
        for i in range(20):
            st.sidebar.write("")
    
        
        
        st.sidebar.image('./image/honeylogo.jpg',width=350,output_format ='JPEG')  

    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

            

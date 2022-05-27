# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:25:55 2022

@author: PC
"""

import pandas as pd
import json
def create_traindata_with_ypre(dataframe,
                                 pre_n_minutes=6,
                                 y_name='220KV站矿钢线进线(272)P（有功功率）',
                                 dropna=1):

    """构建训练数据,生成ypre1，ypre2...等预测目标

    输入参数：
    ----------
       dataframe : dataframe,原始时序数据,包含特征和目标                               
    pre_n_minutes: int, 预测到未来几分钟，默认=6
           y_name: str, 预测目标的字段名，比如：'220KV站矿钢线进线(272)P（有功功率）'    
           dropna: bool,是否剔除包含nan值的行，默认=1剔除
    返回值：
    -------       
    traindata : 每一行包含了ypre1，ypre2...等预测目标，以及当前时刻的x和y作为特征值"""  
    
    ypre_total=pd.DataFrame()
    
    for i in range(1,pre_n_minutes+1):
        ypre=dataframe[[y_name]].shift(-i).rename(columns={y_name:y_name+'ypre'+str(i)})
        ypre_total=pd.concat([ypre_total,ypre],axis=1)  

    if dropna:
        traindata=pd.concat([ypre_total,dataframe],axis=1).dropna()
    else:
        traindata=pd.concat([ypre_total,dataframe],axis=1)

    return traindata,[y_name+'ypre'+str(i) for i in range(1,pre_n_minutes+1)]

def create_diff_fea(train_set,
                    target_name_list,
                    diff_periods=10,
                    use_past_fea=0):

    """创建差分特征，当前时刻的特征值依次减去上1，2，3... diff_periods时刻的值作为当前样本的新特征  

    输入参数：
    ----------
           train_set: dataframe, 包含所有预测目标ypre1......ypre6,和特征值的数据集,create_traindata_withy的输出结果                    
    target_name_list: str or str_list, 目标变量的名称[ypre1,......ypre6]
        diff_periods：int 构建差分特征，当前特征依次减去上1，2，3...diff_periods分钟的特征值作为差分特征计算差值的最大间隔
        use_past_fea：bool,是否使用历史原始数据，作为当前样本的特征，默认0：不使用

    返回值：
    -------       
    train_set：dataframe，包含预测目标和原始特征值，差分特征值的数据集"""

    train_data=train_set.drop(target_name_list,axis=1)
    
    for i in range(1,diff_periods):
        fea_diff=train_data.diff(periods=i).rename(lambda x: x+'_diff'+str(i),axis=1)
        train_set=pd.concat([train_set,fea_diff],axis=1)
    if use_past_fea:
        for i in range(1,diff_periods):
            fea_past=train_data.shift(i).rename(lambda x: x+'_past'+str(i),axis=1)
            train_set=pd.concat([train_set,fea_past],axis=1)            
    return train_set 
        

def time_set_box(dataframe,st):
    import datetime
    col1,col2,col3,col4,col5=st.columns(5)    
    
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
    demo_start=str(demo_datestart)+str(" ")+str(demo_timestart)
    demo_end=str(demo_dateend)+str(" ")+str(demo_timeend)   
    seldata=dataframe[demo_start:demo_end]                       
    return seldata

def get_timeseries_predict(uploaded_model,filename,env_button,model_store,datasource,pc_rg):
    with open('model_info_'+env_button+uploaded_model+'.json','r') as f:
        model_idname_info=json.load(f)
        
    with open(filename[0:-4]+uploaded_model+'model_config.json','r') as f:
        model_config=json.load(f, encoding="utf8")
    domain='./'+filename[0:-4]+'/'+env_button+'/'+uploaded_model

    model_online={}
    for target_name, model_id in model_idname_info.items():
        model_online[target_name]=model_store.load( domain, 
                                      model_id=model_id)

    
    dataframe=pd.read_csv(datasource,index_col=0).fillna(0)
    shift_dataframe,target_name_list=create_traindata_with_ypre(dataframe,
                                                 pre_n_minutes=model_config["pre_shift"],
                                                 y_name=model_config["target_col"],
                                                 dropna=1)

    shift_dataframe_withfea=create_diff_fea(shift_dataframe,
                                            target_name_list,
                                            diff_periods=10,
                                            use_past_fea=0)
    
    
    pre_data_all={}
    for target_name in target_name_list:                                                                       
    
        pre_data=pd.DataFrame()
        pre_data=pc_rg.predict_model(model_online[target_name],data=shift_dataframe_withfea)[['Label']]
       
        pre_data.index=pd.to_datetime(pre_data.index)
        pre_data.columns=[target_name]                            
        pre_data_all[target_name]=pre_data.copy()
    return pre_data_all,target_name_list,model_online,model_config,dataframe








                 # with col1:
     
                 #     demo_start=st.number_input('start_time',value=100,key=1)
                 #     demo_end=st.number_input('end_time',value=150,key=2)
                 #     uplimit=st.number_input('当前需量控制上限',value=130,key=3)
                    
                     
                     # col1, col2, col3, col4,col5, col6, col7, col8 = st.columns(8)
                     # col_trend={}
                     # col_trend[1],col_trend[2],col_trend[3],col_trend[4],\
                     # col_trend[5],col_trend[6],col_trend[7],col_trend[8]=\
                     # col1, col2, col3, col4 ,col5, col6, col7, col8
                     # with col_trend[1]:
                     #     st.write("趋势线选择：")
                     # for i in range(len(columns)):
                     #     with col_trend[i+2]:
                     #         st.checkbox(columns[i],key=columns[i])
                     # for session in columns:
                     #     if st.session_state[session]:
                     #         my_chart.add_rows(oridata[session])           
                # import plotly_express as px
                # oridata_px=oridata.copy()           
                # ani_frame=[0]*len(oridata_px)
                # cut_bin=15
                # for i in range(len(oridata_px)):
                #     if i%10==0:
                #         ani_frame[i:i+10]=[i]*10            
                # oridata_px['ani_frame']=ani_frame[:len(oridata_px)]
                # oridata_px['time']=pd.to_datetime(oridata_px.index)
                # st.table(oridata_px.head(10))
                # fig = px.scatter(
                #   oridata_px,
                #   x='time', 
                #   y='总降需量',
                #   animation_frame='ani_frame',
                #   # animation_group=target_col,
                #   range_y=[0,160])
                # fig.update_layout(width=800)
                # st.write(fig) 
                
                # if choose=="设备健康": 
                #     apps=os.listdir('./appdict/预测看板/')
                #     appbutton={}
                #     for appx in apps:            
                #         appbutton[appx]=st.button(appx)
                    
                #     for buttons in appbutton.keys():
                #         if appbutton[buttons]:
                #             with open('./appdict/预测看板/'+buttons,'rb') as f:
                #                 appdict=pkl.load(f)
                        
                #         app_name=appdict['appname']
                #         model_online=appdict['model_online']
                #         dataframe=appdict['dataframe']
                #         pre_data=appdict['pre_data']                    
                #         target_col=appdict['target_col']  
                        
                #         r2=r2_score(pre_data[target_col],pre_data['Label']) 
                #         rmse=mean_squared_error(pre_data[target_col],pre_data['Label'],squared=False)
                #         mae=mean_absolute_error(pre_data[target_col],pre_data['Label'])
                #         st.write('r2: ',round(r2,4),'rmse: ',round(rmse,4),'mae: ',round(mae,4))
                        
                #         pre_data['index']=pd.to_datetime(pre_data.index) 
                        
                #         p = figure(title='y_true VS y_predict',x_axis_type="datetime",height=350, width=500)
                #         p.line(x='index',
                #                y=target_col,
                #                legend_label='True', 
                #                line_width=2,
                #                source=pre_data,
                #                color='black')
                #         p.line(x='index',
                #                 y='Label', 
                #                 legend_label='y_pre', 
                #                 line_width=2,
                #                 source=pre_data,
                #                 color='red')        
                #         st.bokeh_chart(p, use_container_width=True)
                        
                #         fig, ax = plt.subplots(figsize=(7,2), dpi=120)
                #         sns.kdeplot(pre_data[target_col])
                #         sns.kdeplot(pre_data['Label'])
                #         ax.legend(['y_true','y_predict'])
                #         # ax.title('kde')
                #         st.pyplot(fig)
                
        # import streamlit.components.v1 as components
        # # components.html("""我们要使用的HTML代码""", weight=500, height=300, scrolling=True)
        # def draw_table(df, theme, table_height):
        #    columns = df.columns
        #    thead1="""<thead><th scope="col"></th>"""
        #    thead_temp = []
        #    for k in range(len(list(columns))):
        #        thead_temp.append("""<th scope="col" class="text-white">"""+str(list(columns)[k])+"""</th>""")
        #    header = thead1+"".join(thead_temp)+"""</tr></thead>"""
        #    rows = []
        #    rows_temp = []
        #    for i in range(df.shape[0]):
        #        rows.append("""<th scope="row">"""+str(i+1)+"""</th>""")
        #        rows_temp.append(df.iloc[i].values.tolist())
        #    td_temp = []
        #    for j in range(len(rows_temp)):
        #        for m in range(len(rows_temp[j])):
        #            td_temp.append("""<td class="text-white">"""+str(rows_temp[j][m])+"""</td>""")
        #    td_temp2 = []
        #    for n in range(len(td_temp)):
        #        td_temp2.append(td_temp[n:n+df.shape[1]])
        #    td_temp3 = []
        #    for x in range(len(td_temp2)):
        #        if int(x % (df.shape[1])) == 0:
        #            td_temp3.append(td_temp2[x])
        #    td_temp4 = []
        #    for xx in range(len(td_temp3)):
        #        td_temp4.append("".join(td_temp3[xx]))
        #    td_temp5 = []
        #    for v in range(len(td_temp4)):
        #        td_temp5.append("""<tr><th scope="row" class="text-white">"""+str(v+1)+"""</th>"""+str(td_temp4[v])+"""</tr>""")
        #    table_html = """<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">"""+\
        #    """<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p" crossorigin="anonymous"></script>"""+\
        #    """<table class="table text-center table-bordered """+str(theme)+'"'+">""" + \
        #    header+"""<tbody>"""+"".join(td_temp5)+"""</tbody></table>"""
        
        #    return components.html(table_html,height=table_height, scrolling=True)                
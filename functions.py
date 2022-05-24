# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:25:55 2022

@author: PC
"""

import pandas as pd
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
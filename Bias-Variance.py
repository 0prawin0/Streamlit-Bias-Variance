import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler ,StandardScaler 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor,export_graphviz,plot_tree
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,r2_score

st.set_page_config(
    page_title="Bias-Variance variation for different algorithms",
    page_icon="📊",
    layout="centered",  # or "centered"
    initial_sidebar_state="expanded",  # or "collapsed"
)

st.markdown("<style>div.css-9s0hzh {width: 100% !important;}</style>", unsafe_allow_html=True)


# st.title("Bias-Variance")

st.sidebar.header("Bias-Variance for different algorithms for Bike rental Dataset")


bike=pd.read_csv('bike data.csv')
df_bike = bike.copy()

df_bike.drop(columns=['Date'],inplace=True)

df_bike.drop(['Casual Users','Registered Users'],inplace=True,axis = 1)

df_bike.drop_duplicates(inplace=True)

num_cols_bike=df_bike[['Temperature F','Temperature Feels F','Humidity','Wind Speed']]
Q1 = num_cols_bike.quantile(0.25)
Q3 = num_cols_bike.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

y_main_bike = df_bike['Total Users']
df_bike.drop(columns=['Total Users'],inplace=True,axis=1)

X_num_bike = df_bike[['Temperature F', 'Temperature Feels F', 'Humidity', 'Wind Speed']]
ss=MinMaxScaler()
scaled_bike = ss.fit_transform(X_num_bike)

scaled_bike = pd.DataFrame(scaled_bike,columns=X_num_bike.columns,index = X_num_bike.index)

X_main_bike = pd.concat([scaled_bike,df_bike[['Season','Hour','Holiday','Day of the Week','Working Day','Weather Type']]],axis=1)

X_train_bike, X_test_bike, y_train_bike,y_test_bike = train_test_split(X_main_bike,y_main_bike,test_size=0.3)

model_selected = st.sidebar.selectbox('Select the algorithm',['Random Forest','Decision Tree','KNN'])

# fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig,ax = plt.subplots(1, 1, figsize=(12, 6))



# Random forest

if model_selected == 'Random Forest':

    bias_list = []
    variance_list = []

    model_num = st.slider('Select the number of Bootstrap models:',2,30,1)

    for i in [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]:
        for j in range(model_num):
            y_pred_df = pd.DataFrame()
            train_set_X = X_train_bike.sample(frac=0.9)                      
            index = train_set_X.index                                        
            train_set_y = y_train_bike[index]

            rf = DecisionTreeRegressor(max_depth = i)
            model = rf.fit(train_set_X,train_set_y)
            y_pred_test = model.predict(X_test_bike)
            y_pred_df[f'yhat{j}'] = y_pred_test
                        
        y_pred_df.index=list(y_test_bike.index)
        y_pred_df['mean'] = 0

        y_pred_df['mean'] = y_pred_df.sum(axis=1)/(j+1)
        y_pred_df['var'] = 0
        for ab in y_pred_df.columns:
            if ab not in ['var','mean']:
                y_pred_df['var'] = y_pred_df['var'] + (y_pred_df['mean']-y_pred_df[ab])**2/(j+1)

        y_pred_df['bias'] = 0
        y_pred_df['bias'] = ((y_test_bike - y_pred_df['mean'])**2)/(j+1)

        bias_list.append(y_pred_df['bias'].mean())
        variance_list.append(y_pred_df['var'].mean())



    df_var_bias=pd.DataFrame({'Depth':list(np.arange(2,32,2)),'Variance':variance_list,'Bias':bias_list})  

    mm = MinMaxScaler()
    df_var_bias['Variance']=mm.fit_transform(df_var_bias['Variance'].values.reshape(-1,1))
    df_var_bias['Bias']=mm.fit_transform(df_var_bias['Bias'].values.reshape(-1,1))

    st.subheader('Bias Variance Chart')
    # fig,ax = plt.subplots()
    ax.plot(df_var_bias['Depth'],df_var_bias['Bias'],marker='*',label='Bias-sqr')
    ax.plot(df_var_bias['Depth'],df_var_bias['Variance'],marker='o',label='Variance',color = 'r')
    ax.legend()
    ax.set_xlabel('Depth')
    ax.set_ylabel('Bias-Variance')


    # ax2.plot(df_var_bias['Depth'],df_var_bias['Variance'],marker='o',label='Variance',color = 'r')
    # ax2.set_xlabel('Depth')
    # ax2.set_ylabel('Variance')
    st.pyplot(fig)

    # st.subheader('Variance Chart')
    # fig2,ax2 = plt.subplots()
    # ax2.plot(df_var_bias['Depth'],df_var_bias['Variance'],marker='o',label='Variance')
    # ax2.set_xlabel('Depth')
    # ax2.set_ylabel('Variance')
    # st.pyplot(fig2)

elif model_selected == 'Decision Tree':
    bias_list = []
    var_list = []

    for x in range(1,50,1):
        dtc = DecisionTreeRegressor(max_depth=x)
        model = dtc.fit(X_train_bike,y_train_bike)
        y_pred = model.predict(X_test_bike)
        variance = y_pred.var(ddof=0)
        bias = np.sum((y_pred - y_test_bike)**2)
        bias_list.append(bias)
        var_list.append(variance)
    df_var_bias=pd.DataFrame({'Depth':np.arange(1,50,1),'Variance':var_list,'Bias':bias_list})  
    
    mm = MinMaxScaler()
    df_var_bias['Variance']=mm.fit_transform(df_var_bias['Variance'].values.reshape(-1,1))
    df_var_bias['Bias']=mm.fit_transform(df_var_bias['Bias'].values.reshape(-1,1))

    st.subheader('Bias Variance Chart')
    # fig,ax = plt.subplots()
    ax.plot(df_var_bias['Depth'],df_var_bias['Bias'],marker='*',label='Bias-sqr')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Bias-Variance')
    ax.plot(df_var_bias['Depth'],df_var_bias['Variance'],marker='o',label='Variance',color = 'r')
    ax.legend()


    # ax2.plot(df_var_bias['Depth'],df_var_bias['Variance'],marker='o',label='Variance',color = 'r')
    # ax2.set_xlabel('Depth')
    # ax2.set_ylabel('Variance')
    st.pyplot(fig)

    # st.subheader('Variance Chart')
    # fig2,ax2 = plt.subplots()
    # ax2.plot(df_var_bias['Depth'],df_var_bias['Variance'],marker='o',label='Variance')
    # ax2.set_xlabel('Depth')
    # ax2.set_ylabel('Variance')
    # st.pyplot(fig2)

else:

    bias_list = []
    variance_list = []

    for i in [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]:
        for j in range(5):
            y_pred_df = pd.DataFrame()
            train_set_X = X_train_bike.sample(frac=0.9)                      
            index = train_set_X.index                                        
            train_set_y = y_train_bike[index]

            knn = KNeighborsRegressor(n_neighbors=i)
            model = knn.fit(train_set_X,train_set_y)
            y_pred_test = model.predict(X_test_bike)
            y_pred_df[f'yhat{j}'] = y_pred_test
                        
        y_pred_df.index=list(y_test_bike.index)
        #y_pred_df.columns = np.arange(1,j+1)    
        y_pred_df['mean'] = 0

        y_pred_df['mean'] = y_pred_df.sum(axis=1)/(j+1)
        y_pred_df['var'] = 0
        for ab in y_pred_df.columns:
            if ab not in ['var','mean']:
                y_pred_df['var'] = y_pred_df['var'] + (y_pred_df['mean']-y_pred_df[ab])**2/(j+1)

        y_pred_df['bias'] = 0
        y_pred_df['bias'] = ((y_test_bike - y_pred_df['mean'])**2)/(j+1)

        bias_list.append(y_pred_df['bias'].mean())
        variance_list.append(y_pred_df['var'].mean())
    df_var_bias=pd.DataFrame({'Neighbors':list(np.arange(2,32,2)),'Variance':variance_list,'Bias':bias_list})  

    st.subheader('Bias Variance Chart')
    mm = MinMaxScaler()
    df_var_bias['Variance']=mm.fit_transform(df_var_bias['Variance'].values.reshape(-1,1))
    df_var_bias['Bias']=mm.fit_transform(df_var_bias['Bias'].values.reshape(-1,1))
    # fig,ax = plt.subplots()
    ax.plot(df_var_bias['Neighbors'],df_var_bias['Bias'],marker='*',label='Bias-sqr')
    ax.plot(df_var_bias['Neighbors'],df_var_bias['Variance'],marker='o',label='Variance',color = 'r')
    ax.legend()
    ax.set_xlabel('Neighbors')
    ax.set_ylabel('Bias-Variance')
    st.pyplot(fig)

    # ax2.plot(df_var_bias['Neighbors'],df_var_bias['Variance'],marker='o',label='Variance',color = 'r')
    # ax2.set_xlabel('Neighbors')
    # ax2.set_ylabel('Variance')

    # st.subheader('Variance Chart')
    # fig2,ax2 = plt.subplots()
    # ax2.plot(df_var_bias['Depth'],df_var_bias['Variance'],marker='o',label='Variance')
    # ax2.set_xlabel('Depth')
    # ax2.set_ylabel('Variance')
    # st.pyplot(fig2)  







import math
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from os import listdir, walk
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tsfresh import extract_features,select_features,extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
import tsfresh
from sklearn import cluster
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn import cross_validation, metrics, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


# ### function

# In[3]:


def modelfit(alg, X , Y,useTrainCV=True, cv_folds=5, early_stopping_rounds=50,cv=0):
    
    
    x_train = X.drop([cv])
    y_train = Y.drop([cv])
    x_val = X.iloc[cv,:].to_frame().T
    y_val = Y.iloc[cv,:].to_frame().T


    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(x_train, y_train)

    #Predict training set:
    dtrain_predictions = alg.predict(x_val)
    rmse=mean_squared_error(y_val,dtrain_predictions)**(.5)
    #Print model report:
#     print ("\nModel Report")
#     print ("RMSE :",rmse)

#     feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
#     plt.figure(figsize=(20,10))
#     feat_imp.plot(kind='bar', title='Feature Importances')
#     plt.ylabel('Feature Importance Score')
    return(dtrain_predictions,rmse)


# ### Import data

# In[4]:


path='C:\\Users\\DALab\\Desktop\\data contest\\preliminary data'
data=pd.DataFrame(np.nan,index=range(300000), columns=['id','time','std','range']) ; y=[]
index=0
for file in listdir(path):
    temp=pd.read_excel(path+'\\'+file,header=None)
    y.append(np.float32(temp.iloc[7500,0][9:]))
    data.iloc[(index*7500):((index+1)*7500),0]=index
    data.iloc[(index*7500):((index+1)*7500),1]=range(1,7501)
    data.iloc[(index*7500):((index+1)*7500),2]=temp.loc[:7499,:].std(axis=1).values
    data.iloc[(index*7500):((index+1)*7500),3]=(temp.loc[:7499,:].max(axis=1)-temp.loc[:7499,:].min(axis=1)).values
    index+=1
data['log_std']=np.log(data['std']) ; data['log_range']=np.log(data['range']) 


# ### Scale X

# In[649]:


min_max_scaler = preprocessing.MinMaxScaler()
data.iloc[:,2:5]=min_max_scaler.fit_transform(data.iloc[:,2:5])


# In[839]:


extracted_features = extract_features(data, column_id="id", column_sort="time")


# ### Choose one way to get final features

# In[2]:


features=pd.read_csv('C:\\Users\\DALab\\Desktop\\data contest\\Features.csv').iloc[:,:3176]
#features_filtered = select_features(features,np.array(y),fdr_level=.6)


# In[8]:


model= xgb.XGBRegressor(
     learning_rate =0.16,
     n_estimators=1000,
     max_depth=5,
     min_child_weight=5,
     gamma=0,
     subsample=0.77,
     colsample_bytree=0.7,
     objective= 'reg:linear',
     nthread=4,
     alpha=.1,
     lamda=1)


# ### Build xgboost model

# In[21]:


xgb_param = model.get_xgb_params()
xgtrain = xgb.DMatrix(features_filtered, label=y)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=5, early_stopping_rounds=50)
model.set_params(n_estimators=cvresult.shape[0])
model.fit(features_filtered,y)
model.predict(features_filtered)


# In[30]:


path='C:\\Users\\DALab\\Desktop\\data contest\\test data'
test=pd.DataFrame(np.nan,index=range(75000), columns=['id','time','1st','2nd','3rd','4th']) 
index=0
for file in listdir(path):
    temp=pd.read_excel(path+'\\'+file,header=None)
    test.iloc[(index*7500):((index+1)*7500),2:6]=temp.loc[:7499,:].values
    test.iloc[(index*7500):((index+1)*7500),0]=index
    test.iloc[(index*7500):((index+1)*7500),1]=range(1,7501)
    index+=1


# In[31]:


min_max_scaler = preprocessing.MinMaxScaler()
test.iloc[:,2:]=min_max_scaler.fit_transform(test.iloc[:,2:6])


# In[32]:


temp_feature=extract_features(test, column_id="id", column_sort="time")


# In[65]:


test_features=temp_feature[list(features_filtered.columns)]


# In[53]:


test_features['y']=model.predict(test_features)


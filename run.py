import pandas as pd
import numpy as np
import glob
from tsfresh import extract_features
# from util import *
import os
import tsfresh as ts
from sklearn.preprocessing import MinMaxScaler


def read_file_name(path = './data/data_odiginal/test'):
    file_name = []
    for file in os.listdir(path):
        if file.endswith(".xls"):
            file_name.append(file)
            print(file)
    return file_name

def read_xls(path = './data/data_odiginal'):
    all_files = glob.glob(path + "/*.xls")
    # print(all_files)
    ans = []
    file_list = []
    AR_list = []
    for file in all_files:
        print(file)
        df = pd.read_excel(file,header=None)
        ans_ = df.iloc[-1,0]
        df = df[:-1]
        AR = df[9:].values - df[:-9].values
        AR_list.append(AR)
        file_list.append(df)
        ans.append(ans_)
    return file_list, AR_list, ans



def jacky_feature(path = './data/data_odiginal/test', file_format='xls', num_files=40, Trainable=True):
    all_files = glob.glob(path + "/*.xls")
    # print(all_files)
    ans = []
    file_list = []
    for file in all_files:
        print(file)
        df = pd.read_excel(file, header=None)
        if Trainable:
            ans_ = df.iloc[-1,0]
            ans.append(ans_)
            df = df[:-1]
            file_list.append(df)
        else:
            file_list.append(df)



    df_con = pd.concat(file_list, ignore_index=True)
    df_con = df_con.astype('float32')
    df_con.columns = ['1st', '2nd', '3rd', '4th']
    df_con['id'] = pd.Series(np.repeat(np.arange(num_files), 7500), index=df_con.index)
    df_con['time'] = pd.Series(np.tile(np.arange(7500), num_files), index=df_con.index)


    f = []
    for file in os.listdir(path):
        if file.endswith(file_format):
            print(file)
            f.append(file)

    min_max_scaler = MinMaxScaler()
    df_con[['1st', '2nd', '3rd', '4th']] = min_max_scaler.fit_transform(df_con[['1st', '2nd', '3rd', '4th']])
    print('feature extract.')
    df_feature = extract_features(df_con, column_id='id', column_sort='time')
    features_filtered = ['id', '2nd__fft_coefficient__coeff_76__attr_"imag"', '3rd__fft_coefficient__coeff_24__attr_"real"', '2nd__fft_coefficient__coeff_94__attr_"real"', '2nd__partial_autocorrelation__lag_3',
                         '3rd__fft_coefficient__coeff_62__attr_"abs"', '2nd__fft_coefficient__coeff_99__attr_"imag"', '3rd__fft_coefficient__coeff_57__attr_"real"', '2nd__fft_coefficient__coeff_96__attr_"real"',
                         '1st__energy_ratio_by_chunks__num_segments_10__segment_focus_1', '2nd__fft_coefficient__coeff_11__attr_"angle"', '3rd__fft_coefficient__coeff_73__attr_"imag"',
                         '1st__fft_coefficient__coeff_41__attr_"imag"', '2nd__fft_coefficient__coeff_81__attr_"real"']
    print('feature select.')
    df_feature = df_feature[features_filtered]
    df_feature = df_feature.iloc[:, 1:]

    print(f)

    return df_feature, f, ans

 # def xgb_data(dataframe=None):
 #    df_jacky_xgb = dataframe.copy()
 #    X_jacky_input = df_jacky_xgb.iloc[:, 1:]
 #    return X_jacky_input



def model_train(features_filtered, y):

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

    xgb_param = model.get_xgb_params()
    xgtrain = xgb.DMatrix(features_filtered, label=y)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=5, early_stopping_rounds=50)
    model.set_params(n_estimators=cvresult.shape[0])
    print('ready to train.')
    model.fit(features_filtered,y)
    print('Done!')


    return model


if __name__ == '__main__':
    df_feature_train, file_name_train, train_ans= jacky_feature(path = './data/data_odiginal', file_format='xls', num_files=40, Trainable=True)	
    file_list_train, AR_list_train, ans_train = read_xls(path = './data/data_odiginal')
    model = model_train(features_filtered=df_feature_train, y=ans_train)


    # test
    df_feature_test, file_name_test, _ = jacky_feature(path = './data/data_odiginal/test', file_format='xls', num_files=10, Trainable=False)	
    answer = model.predict(df_feature_test)
    df_ans = ataFrame({'file_name':file_name_test, 'answer':answer})
    print(df_ans)
    print('finish!')
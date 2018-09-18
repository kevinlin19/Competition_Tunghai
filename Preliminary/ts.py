import pandas as pd
import numpy as np
import glob
from tsfresh import extract_features
import tsfresh as ts


# Torr
path = './data'
all_files = glob.glob(path + "/*.xls")
ans = []
file_list = []
AR_list = []
for file in all_files:
    df = pd.read_excel(file, header=None)
    ans_ = df.iloc[-1,0]
    df = df[:-1]
    AR = df[9:].values - df[:-9].values
    AR_list.append(AR)
    file_list.append(df)
    ans.append(ans_)

# X: df_con
df_con = pd.concat(file_list, ignore_index=True)
df_con.columns = ['col_1', 'col_2', 'col_3', 'col_4']
df_con = df_con.astype(np.float32)
df_con['id'] = pd.Series(np.repeat(np.arange(40), 7500), index=df_con.index)
df_con['time'] = pd.Series(np.tile(np.arange(7500), 40), index=df_con.index)
df = df_con[['id', 'time', 'col_1', 'col_2', 'col_3', 'col_4']].astype(np.float32)
df_feature = extract_features(df_con, column_id='id', column_sort='time', n_jobs=8)

# df_feature.to_csv('./feature_mean.csv')

AR_list = np.asarray(AR_list)
df_empty = pd.DataFrame(columns = ['col_1', 'col_2', 'col_3', 'col_4'])
for i in range(AR_list.shape[0]):
    df_ar = pd.DataFrame(AR_list[i], columns = ['col_1', 'col_2', 'col_3', 'col_4'])
    df_empty = df_empty.append(df_ar, ignore_index=True)
df_ar_con = df_empty.copy()
df_ar_con = df_ar_con.astype(np.float32)
df_ar_con['id'] = pd.Series(np.repeat(np.arange(40), 7491), index=df_ar_con.index)
df_ar_con['time'] = pd.Series(np.tile(np.arange(7491), 40), index=df_ar_con.index)
df_ar_con = df_ar_con[['id', 'time', 'col_1', 'col_2', 'col_3', 'col_4']].astype(np.float32)
df_feature_ar = extract_features(df_ar_con, column_id='id', column_sort='time', n_jobs=8)
df_feature_ar.to_csv('./feature_ar.csv')

# create X: X_original, X_log, X_abs
X_original = np.zeros(shape=[40, 7500, 6])
X_log = np.zeros(shape=[40, 7500, 6])
for i in range(40):
    X_original[i] = df_con[i*7500:(i+1)*7500]
for i in range(40):
    X_log[i] = df_con_log[i*7500:(i+1)*7500]
X_abs = np.abs(X_log)
X_ar = np.array(AR_list)
X_10_ar = X_ar * 10e+5 +5
x_scale = ((X_ar - np.percentile(X_ar, 1)) * 10)/(np.percentile(X_ar, 99)-np.percentile(X_ar, 1)) - 5
# Y: ans
ans_final = []
for an in ans:
    ans_final.append(an[9:])
ans_final = np.array(ans_final).astype(np.float32)
ans_final_original = ans_final.reshape(-1, 1)
ans_final_log = np.log(ans_final_original)
ans_final_abs10 = np.abs(ans_final_log) * 10

import pandas as pd
import numpy as np
import glob
from tsfresh import extract_features


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
df_con = df_con.astype('float32')
df_con.columns = ['col_1', 'col_2', 'col_3', 'col_4']
# df_con[['col_1_o', 'col_2_o', 'col_3_o', 'col_4_o']] = df_con
# df_con[['col_1', 'col_2', 'col_3', 'col_4']] = np.log(df_con[['col_1', 'col_2', 'col_3', 'col_4']])
# df_con['total_MEAN'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].mean(axis=1)
# df_con['total_STD'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].std(axis=1)
df_con['one_two_MEAN'] = df_con[['col_1', 'col_2']].mean(axis=1)
df_con['one_three_MEAN'] = df_con[['col_1', 'col_3']].mean(axis=1)
# df_con['one_four_MEAN'] = df_con[['col_1', 'col_4']].mean(axis=1)
# df_con['one_four_MEAN'] = df_con[['col_1', 'col_4']].mean(axis=1)
# df_con['two_three_MEAN'] = df_con[['col_2', 'col_3']].mean(axis=1)
# df_con['two_four_MEAN'] = df_con[['col_2', 'col_4']].mean(axis=1)
# df_con['three_four_MEAN'] = df_con[['col_3', 'col_4']].mean(axis=1)
# df_con['one_two_three_MEAN'] = df_con[['col_1', 'col_2', 'col_3']].mean(axis=1)
# df_con['one_two_four_MEAN'] = df_con[['col_1', 'col_2', 'col_4']].mean(axis=1)
# df_con['four_two_three_MEAN'] = df_con[['col_4', 'col_2', 'col_3']].mean(axis=1)
# df_con['one_two_STD'] = df_con[['col_1', 'col_2']].std(axis=1)
# df_con['one_three_STD'] = df_con[['col_1', 'col_3']].std(axis=1)
# df_con['one_four_STD'] = df_con[['col_1', 'col_4']].std(axis=1)
# df_con['one_four_STD'] = df_con[['col_1', 'col_4']].std(axis=1)
# df_con['two_three_STD'] = df_con[['col_2', 'col_3']].std(axis=1)
# df_con['two_four_STD'] = df_con[['col_2', 'col_4']].std(axis=1)
# df_con['three_four_STD'] = df_con[['col_3', 'col_4']].std(axis=1)
# df_con['one_two_three_STD'] = df_con[['col_1', 'col_2', 'col_3']].std(axis=1)
# df_con['one_two_four_STD'] = df_con[['col_1', 'col_2', 'col_4']].std(axis=1)
# df_con['four_two_three_STD'] = df_con[['col_4', 'col_2', 'col_3']].std(axis=1)
# df_con['total_RANGE'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].max(axis=1) - df_con[['col_1', 'col_2', 'col_3', 'col_4']].min(axis=1)
# df_con['MIN'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].min(axis=1)
# df_con['MAX'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].max(axis=1)
# log ----
# df_con['total_MEAN_log'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].mean(axis=1)
df_con['total_STD_log'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].std(axis=1)
# df_con['one_two_MEAN_log'] = df_con[['col_1', 'col_2']].mean(axis=1)
# df_con['one_three_MEAN_log'] = df_con[['col_1', 'col_3']].mean(axis=1)
# df_con['one_four_MEAN_log'] = df_con[['col_1', 'col_4']].mean(axis=1)
# df_con['one_four_MEAN_log'] = df_con[['col_1', 'col_4']].mean(axis=1)
# df_con['two_three_MEAN_log'] = df_con[['col_2', 'col_3']].mean(axis=1)
# df_con['two_four_MEAN_log'] = df_con[['col_2', 'col_4']].mean(axis=1)
# df_con['three_four_MEAN_log'] = df_con[['col_3', 'col_4']].mean(axis=1)
# df_con['one_two_three_MEAN_log'] = df_con[['col_1', 'col_2', 'col_3']].mean(axis=1)
# df_con['one_two_four_MEAN_log'] = df_con[['col_1', 'col_2', 'col_4']].mean(axis=1)
# df_con['four_two_three_MEAN_log'] = df_con[['col_4', 'col_2', 'col_3']].mean(axis=1)
# df_con['one_two_STD_log'] = df_con[['col_1', 'col_2']].std(axis=1)
# df_con['one_three_STD_log'] = df_con[['col_1', 'col_3']].std(axis=1)
# df_con['one_four_STD_log'] = df_con[['col_1', 'col_4']].std(axis=1)
# df_con['one_four_STD_log'] = df_con[['col_1', 'col_4']].std(axis=1)
# df_con['two_three_STD_log'] = df_con[['col_2', 'col_3']].std(axis=1)
# df_con['two_four_STD_log'] = df_con[['col_2', 'col_4']].std(axis=1)
# df_con['three_four_STD_log'] = df_con[['col_3', 'col_4']].std(axis=1)
# df_con['one_two_three_STD_log'] = df_con[['col_1', 'col_2', 'col_3']].std(axis=1)
# df_con['one_two_four_STD_log'] = df_con[['col_1', 'col_2', 'col_4']].std(axis=1)
# df_con['four_two_three_STD_log'] = df_con[['col_4', 'col_2', 'col_3']].std(axis=1)
df_con['total_RANGE_log'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].max(axis=1) - df_con[['col_1', 'col_2', 'col_3', 'col_4']].min(axis=1)
# df_con['MIN_log'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].min(axis=1)
# df_con['MAX_log'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].max(axis=1)
# df_con['total_RANGE'] = df_con[['col_1_o', 'col_2_o', 'col_3_o', 'col_4_o']].max(axis=1) - df_con[['col_1_o', 'col_2_o', 'col_3_o', 'col_4_o']].min(axis=1)
# df_con['MIN'] = df_con[['col_1_o', 'col_2_o', 'col_3_o', 'col_4_o']].min(axis=1)
# df_con['MAX'] = df_con[['col_1_o', 'col_2_o', 'col_3_o', 'col_4_o']].max(axis=1)

# ts feature
# df_con['id'] = pd.Series(np.repeat(np.arange(40), 7500), index=df_con.index)
# df_con['time'] = pd.Series(np.tile(np.arange(7500), 40), index=df_con.index)
# df_feature = extract_features(df_con, column_id='id', column_sort='time', n_jobs=8)
# df_feature.to_csv('./feature_mean_log_total.csv')

df_con = df_con[['one_two_MEAN', 'one_three_MEAN', 'total_STD_log', 'total_RANGE_log']]
df_con = df_con.astype(np.float32)
df_con_log = np.log(df_con)
# create X: X_original, X_log, X_abs
X_original = np.zeros(shape=[40, 7500, 4])
X_log = np.zeros(shape=[40, 7500, 4])
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

#Y describe
# data_y = pd.DataFrame({'Ori':ans_final_original.reshape(40,),
#                       'Log':ans_final_log.reshape(40,),
#                       'Abs':ans_final_abs10.reshape(40,)},
#                       index=np.arange(40))
# '''
# Out[191]:
#              Abs        Log        Ori
# count  40.000000  40.000000  40.000000
# mean    5.923230  -0.583191   0.588633
# std     3.108662   0.328098   0.200278
# min     0.710175  -1.184170   0.306000
# 25%     3.713607  -0.823144   0.439125
# 50%     5.811889  -0.581189   0.559250
# 75%     8.231436  -0.371361   0.689850
# max    11.841702   0.111631   1.118100
# '''





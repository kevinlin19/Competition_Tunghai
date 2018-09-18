import pandas as pd
import numpy as np
import glob
from tsfresh import extract_features

path = './data/test'
all_files = glob.glob(path + "/*.xls")
file_list = []
AR_list = []
for file in all_files:
    df = pd.read_excel(file, header=None)
    AR = df[9:].values - df[:-9].values
    AR_list.append(AR)
    file_list.append(df)

# X: df_con
df_con = pd.concat(file_list, ignore_index=True)
df_con = df_con.astype('float32')
df_con.columns = ['col_1', 'col_2', 'col_3', 'col_4']
df_con['total_MEAN'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].mean(axis=1)
df_con['total_STD'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].std(axis=1)
df_con['one_two_MEAN'] = df_con[['col_1', 'col_2']].mean(axis=1)
df_con['one_three_MEAN'] = df_con[['col_1', 'col_3']].mean(axis=1)
df_con['one_four_MEAN'] = df_con[['col_1', 'col_4']].mean(axis=1)
df_con['one_four_MEAN'] = df_con[['col_1', 'col_4']].mean(axis=1)
df_con['two_three_MEAN'] = df_con[['col_2', 'col_3']].mean(axis=1)
df_con['two_four_MEAN'] = df_con[['col_2', 'col_4']].mean(axis=1)
df_con['three_four_MEAN'] = df_con[['col_3', 'col_4']].mean(axis=1)
df_con['one_two_three_MEAN'] = df_con[['col_1', 'col_2', 'col_3']].mean(axis=1)
df_con['one_two_four_MEAN'] = df_con[['col_1', 'col_2', 'col_4']].mean(axis=1)
df_con['four_two_three_MEAN'] = df_con[['col_4', 'col_2', 'col_3']].mean(axis=1)
df_con['one_two_STD'] = df_con[['col_1', 'col_2']].std(axis=1)
df_con['one_three_STD'] = df_con[['col_1', 'col_3']].std(axis=1)
df_con['one_four_STD'] = df_con[['col_1', 'col_4']].std(axis=1)
df_con['one_four_STD'] = df_con[['col_1', 'col_4']].std(axis=1)
df_con['two_three_STD'] = df_con[['col_2', 'col_3']].std(axis=1)
df_con['two_four_STD'] = df_con[['col_2', 'col_4']].std(axis=1)
df_con['three_four_STD'] = df_con[['col_3', 'col_4']].std(axis=1)
df_con['one_two_three_STD'] = df_con[['col_1', 'col_2', 'col_3']].std(axis=1)
df_con['one_two_four_STD'] = df_con[['col_1', 'col_2', 'col_4']].std(axis=1)
df_con['four_two_three_STD'] = df_con[['col_4', 'col_2', 'col_3']].std(axis=1)
df_con['total_RANGE'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].max(axis=1) - df_con[['col_1', 'col_2', 'col_3', 'col_4']].min(axis=1)
df_con['MIN'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].min(axis=1)
df_con['MAX'] = df_con[['col_1', 'col_2', 'col_3', 'col_4']].max(axis=1)
# log ----
df_con['total_MEAN_log'] = np.log(df_con[['col_1', 'col_2', 'col_3', 'col_4']]).mean(axis=1)
df_con['total_STD_log'] = np.log(df_con[['col_1', 'col_2', 'col_3', 'col_4']]).std(axis=1)
df_con['one_two_MEAN_log'] = np.log(df_con[['col_1', 'col_2']]).mean(axis=1)
df_con['one_three_MEAN_log'] = np.log(df_con[['col_1', 'col_3']]).mean(axis=1)
df_con['one_four_MEAN_log'] = np.log(df_con[['col_1', 'col_4']]).mean(axis=1)
df_con['one_four_MEAN_log'] = np.log(df_con[['col_1', 'col_4']]).mean(axis=1)
df_con['two_three_MEAN_log'] = np.log(df_con[['col_2', 'col_3']]).mean(axis=1)
df_con['two_four_MEAN_log'] = np.log(df_con[['col_2', 'col_4']]).mean(axis=1)
df_con['three_four_MEAN_log'] = np.log(df_con[['col_3', 'col_4']]).mean(axis=1)
df_con['one_two_three_MEAN_log'] = np.log(df_con[['col_1', 'col_2', 'col_3']]).mean(axis=1)
df_con['one_two_four_MEAN_log'] = np.log(df_con[['col_1', 'col_2', 'col_4']]).mean(axis=1)
df_con['four_two_three_MEAN_log'] = np.log(df_con[['col_4', 'col_2', 'col_3']]).mean(axis=1)
df_con['one_two_STD_log'] = np.log(df_con[['col_1', 'col_2']]).std(axis=1)
df_con['one_three_STD_log'] = np.log(df_con[['col_1', 'col_3']]).std(axis=1)
df_con['one_four_STD_log'] = np.log(df_con[['col_1', 'col_4']]).std(axis=1)
df_con['one_four_STD_log'] = np.log(df_con[['col_1', 'col_4']]).std(axis=1)
df_con['two_three_STD_log'] = np.log(df_con[['col_2', 'col_3']]).std(axis=1)
df_con['two_four_STD_log'] = np.log(df_con[['col_2', 'col_4']]).std(axis=1)
df_con['three_four_STD_log'] = np.log(df_con[['col_3', 'col_4']]).std(axis=1)
df_con['one_two_three_STD_log'] = np.log(df_con[['col_1', 'col_2', 'col_3']]).std(axis=1)
df_con['one_two_four_STD_log'] = np.log(df_con[['col_1', 'col_2', 'col_4']]).std(axis=1)
df_con['four_two_three_STD_log'] = np.log(df_con[['col_4', 'col_2', 'col_3']]).std(axis=1)
df_con['total_RANGE_log'] = np.log(df_con[['col_1', 'col_2', 'col_3', 'col_4']]).max(axis=1) - np.log(df_con[['col_1', 'col_2', 'col_3', 'col_4']]).min(axis=1)
df_con['MIN_log'] = np.log(df_con[['col_1', 'col_2', 'col_3', 'col_4']]).min(axis=1)
df_con['MAX_log'] = np.log(df_con[['col_1', 'col_2', 'col_3', 'col_4']]).max(axis=1)


# ts feature
df_con['id'] = pd.Series(np.repeat(np.arange(10), 7500), index=df_con.index)
df_con['time'] = pd.Series(np.tile(np.arange(7500), 10), index=df_con.index)
df_feature = extract_features(df_con, column_id='id', column_sort='time', n_jobs=8)
df_feature.to_csv('./feature_test.csv')
df_con.shape
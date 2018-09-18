import pandas as pd
import numpy as np
import glob
from tsfresh import extract_features
# from util import *
import os
import tsfresh as ts
from sklearn.preprocessing import MinMaxScaler


# read xls
def read_xls(path = './data/data_odiginal'):
    all_files = glob.glob(path + "/*.xls")
    # print(all_files)
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
    return file_list, AR_list, ans

# return dataframe
def statistic_feature(file_list, num_files=40):
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
    # log ----------------------------------------------------------------------------------------
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
    df_con['id'] = pd.Series(np.repeat(np.arange(num_files), 7500), index=df_con.index)
    df_con['time'] = pd.Series(np.tile(np.arange(7500), num_files), index=df_con.index)
    return df_con
# Y: ans
def ans_dataframe(ans):
    ans_final = []
    for an in ans:
        ans_final.append(an[9:])
    ans_final = np.array(ans_final).astype(np.float32)
    df_ans = pd.DataFrame(ans_final, columns=['ans'])
    return df_ans

def jacky_feature(path = './data/data_odiginal/test', file_format='xls'):
    test = pd.DataFrame(np.nan, index=range(75000), columns=['id', 'time', '1st', '2nd', '3rd', '4th'])
    index = 0
    for file in os.listdir(path):
        if file.endswith(file_format):
            temp = pd.read_excel(path + '\\' + file, header=None)
            test.iloc[(index * 7500):((index + 1) * 7500), 2:6] = temp.loc[:7499, :].values
            test.iloc[(index * 7500):((index + 1) * 7500), 0] = index
            test.iloc[(index * 7500):((index + 1) * 7500), 1] = range(1, 7501)
            index += 1
    min_max_scaler = MinMaxScaler()
    test.iloc[:, 2:6] = min_max_scaler.fit_transform(test.iloc[:, 2:6])
    df_feature = extract_features(test, column_id='id', column_sort='time')
    features_filtered = ['id', '2nd__fft_coefficient__coeff_76__attr_"imag"', '3rd__fft_coefficient__coeff_24__attr_"real"', '2nd__fft_coefficient__coeff_94__attr_"real"', '2nd__partial_autocorrelation__lag_3',
                         '3rd__fft_coefficient__coeff_62__attr_"abs"', '2nd__fft_coefficient__coeff_99__attr_"imag"', '3rd__fft_coefficient__coeff_57__attr_"real"', '2nd__fft_coefficient__coeff_96__attr_"real"',
                         '1st__energy_ratio_by_chunks__num_segments_10__segment_focus_1', '2nd__fft_coefficient__coeff_11__attr_"angle"', '3rd__fft_coefficient__coeff_73__attr_"imag"',
                         '1st__fft_coefficient__coeff_41__attr_"imag"', '2nd__fft_coefficient__coeff_81__attr_"real"']
    df_feature = df_feature[features_filtered]
    df_feature.to_csv('./data/xgb_feature_new.csv')


if __name__ == '__main__':
    # kai feature
    file_list, AR_list, ans = read_xls()
    df = statistic_feature(file_list=file_list, num_files=40)
    df_ans = ans_dataframe(ans)
    df_feature = extract_features(df, column_id='id', column_sort='time', n_jobs=8)
    df_feature.to_csv('./data/feature_new.csv')
    df_sel = ts.feature_selection.selection.select_features(df_feature, df_ans.ans, fdr_level=0.5)
    df_sel.to_csv('./data/feature_select_new.csv')
    # jacky feature
    # jacky_feature()


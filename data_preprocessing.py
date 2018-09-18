from sklearn.preprocessing import MinMaxScaler
import pandas as pd



# kai data
def kai_input_data(feature_path='./data/feature_select_new.csv', ans_path='./data/ans.csv' ):
    df = pd.read_csv(feature_path, header=0)
    df_ans = pd.read_csv(ans_path, names=['ans'])
    # Scale
    x_scale = MinMaxScaler((0, 2))
    X_train = y_scale.fit_transform(df)
    y_scale = MinMaxScaler()
    Y_train = y_scale.fit_transform(df_ans.ans.values.reshape(-1, 1))
    return x_scale, y_scale, X_train, Y_train, df, df_ans

# xgb data
def xgb_data(path='./data/xgb_feature_new.csv'):
    df_jacky_xgb = pd.read_csv(path, header=0)
    X_jacky_input = df_jacky_xgb.iloc[:, 1:]
    return X_jacky_input

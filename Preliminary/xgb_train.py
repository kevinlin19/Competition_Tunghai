import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
# from combine import *

# util
def RMSE(y_actual, y_predicted):
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms

rmse = []

# load data
df = pd.read_csv('./final_feature.csv', header=0)
df_ans = pd.read_csv('./ans.csv', names=['ans'])
# Scale
scale = MinMaxScaler()
X_train = scale.fit_transform(df)
y_scale = MinMaxScaler()
Y_train = y_scale.fit_transform(df_ans.ans.values.reshape(-1, 1))


# -------------------------Model Grid Search-------------------------
# params
# cv_params = {'n_estimators': [725, 750, 700, 675, 450]} #[400, 500, 600, 700, 800] [350, 375, 400, 425, 450]
# cv_params = {'n_estimators': [500, 475, 450, 425, 400]} #[400, 500, 600, 700, 800][350, 375, 400, 425, 450]
# cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]} # 4, 4
# cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]} # 0.6
# cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]} # 0.8, 0.8
# cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]} # 3, 0.05
# cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]} # 0.1
# --------------------------------------------------------------------
other_params = {'learning_rate': 0.1, 'n_estimators': 725, 'max_depth': 3, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.9, 'colsample_bytree': 0.6, 'gamma': 0.1, 'reg_alpha': 0.05, 'reg_lambda': 1}
# Model Search
# model_search = xgb.XGBRegressor(**other_params)
# optimized_GBM = GridSearchCV(estimator=model_search, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
# optimized_GBM.fit(x_scale, df_ans.ans.values)
# # Score
# evalute_result = optimized_GBM.grid_scores_
# print('每輪迭代運行结果:{0}'.format(evalute_result))
# print('参數的最佳取值：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

# Model
for i in range(40):
    x_train = np.delete(X_train, i, axis=0)
    x_test = X_train[i].reshape(1, -1)
    y_train = np.delete(Y_train, i, axis=0)
    y_test = Y_train[i].reshape(1, -1)
    model = xgb.XGBRegressor(**other_params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    ans = RMSE(y_test, y_pred)
    print('rmse:', ans)
    rmse.append(ans)
    del model
np.mean(rmse)
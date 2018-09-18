from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt


def RMSE(y_actual, y_predicted):
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms

def rmse(preds, train_data):
    labels = train_data.get_label()
    return 'error',  np.sqrt(np.mean((preds-labels)**2)), False

def scoring(reg, x, y):
    pred = reg.predict(x)
    return RMSE(y, pred)

def get_statistic_feature(data):
    mean_ = np.mean(data, axis=1)
    median_ = np.median(data, axis=1)
    max_ = np.max(data, axis=1)
    sum_ = np.sum(data, axis=1)
    min_ = np.min(data, axis=1)
    var_ = np.var(data, axis=1)
    std_ = np.std(data, axis=1)
    ans = np.hstack((mean_, median_, max_, sum_, min_, var_, std_))
    ans = ans.reshape(-1, 7)
    return ans


model_param = {'lr': 0.01, 'depth': 10, 'tree': 5000, 'leaf': 400, 'sample': 0.9, 'seed': 3}
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'max_depth': model_param['depth'],
    'num_leaves': model_param['leaf'],
    'min_data_in_leaf': 20,
    'learning_rate': model_param['lr'],
    'feature_fraction': 1,
    'bagging_fraction': model_param['sample'],
    'bagging_freq': 1,
    'bagging_seed': model_param['seed'],
    'verbose': 0
}

model_param1 = {'lr': 0.05, 'depth': 10, 'tree': 1500, 'leaf': 400, 'sample': 0.9, 'seed': 3}
params1 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'max_depth': model_param1['depth'],
    'num_leaves': model_param1['leaf'],
    'min_data_in_leaf': 20,
    'learning_rate': model_param1['lr'],
    'feature_fraction': 1,
    'bagging_fraction': model_param1['sample'],
    'bagging_freq': 1,
    'bagging_seed': model_param1['seed'],
    'verbose': 0
}

model_param2 = {'lr': 0.1, 'depth': 10, 'tree': 1000, 'leaf': 200, 'sample': 0.9, 'seed': 3}
params2 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'max_depth': model_param2['depth'],
    'num_leaves': model_param2['leaf'],
    'min_data_in_leaf': 20,
    'learning_rate': model_param2['lr'],
    'feature_fraction': 1,
    'bagging_fraction': model_param2['sample'],
    'bagging_freq': 1,
    'bagging_seed': model_param2['seed'],
    'verbose': 0
}

model_param3 = {'lr': 0.05, 'depth': 20, 'tree': 2000, 'leaf': 500, 'sample': 0.9, 'seed': 3}
params3 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'max_depth': model_param3['depth'],
    'num_leaves': model_param3['leaf'],
    'min_data_in_leaf': 20,
    'learning_rate': model_param3['lr'],
    'feature_fraction': 1,
    'bagging_fraction': model_param3['sample'],
    'bagging_freq': 1,
    'bagging_seed': model_param3['seed'],
    'verbose': 0
}

model_param4 = {'lr': 0.1, 'depth': 15, 'tree': 1500, 'leaf': 500, 'sample': 0.9, 'seed': 3}
params4 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'max_depth': model_param4['depth'],
    'num_leaves': model_param4['leaf'],
    'min_data_in_leaf': 20,
    'learning_rate': model_param4['lr'],
    'feature_fraction': 1,
    'bagging_fraction': model_param4['sample'],
    'bagging_freq': 1,
    'bagging_seed': model_param4['seed'],
    'verbose': 0
}

cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=2)

x_train, x_test, y_train, y_test = train_test_split(df, df_ans.ans, train_size=.8)
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

model_param = {'lr': 0.005, 'depth': 10, 'tree': 1000, 'leaf': 400, 'sample': 0.9, 'seed': 3}
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'max_depth': model_param['depth'],
    'num_leaves': model_param['leaf'],
    'min_data_in_leaf': 20,
    'learning_rate': model_param['lr'],
    'feature_fraction': 1,
    'bagging_fraction': model_param['sample'],
    'bagging_freq': 1,
    'bagging_seed': model_param['seed'],
    'verbose': 0,
    'min_data': 1,
    'min_data_in_bin': 1
}

cv_results = lgb.train(
    params,
    lgb_train,
    num_boost_round=5000,
    valid_sets=lgb_eval,
    feval=rmse,
    early_stopping_rounds=30,
    verbose_eval=True)

model_param['tree'] = cv_results.best_iteration


import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))

def rmse(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

df = pd.read_csv('./data/feature_select_new.csv', header=0)
df_ans = pd.read_csv('./data/ans.csv', names=['ans'])
x_train, x_valid, y_train, y_valid = train_test_split(df, df_ans, test_size=0.10, random_state=7)


lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_valid, y_valid,
                       # reference=lgb_train
                       )

model_param = {'lr': 0.005, 'depth': 10, 'tree': 500, 'leaf': 10, 'sample': 0.9, 'seed': 3}
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
    'min_data_in_leaf': 1,
    'min_data_in_bin': 1
}

model = lgb.train(params, lgb_train, num_boost_round=5000, #tree
                  valid_sets=lgb_eval,
                  feval=rmse,
                  early_stopping_rounds=30,
                  verbose_eval=True
                  )

preds = model.predict(x_valid)
print(rmse(y, preds))
#
# # print("調參1：提高準確率")
# min_merror = float('Inf')
# for num_leaves in range(20, 450, 100):
#     for max_depth in range(5, 16, 5):
#         params['num_leaves'] = num_leaves
#         params['max_depth'] = max_depth
#         cv_results = lgb.cv(
#             params,
#             lgb_train,
#             seed=42,
#             nfold=3,
#             early_stopping_rounds=10,
#             verbose_eval=True
#         )
#         mean_merror = pd.Series(cv_results['l2-mean']).min()
#         if mean_merror < min_merror:
#             min_merror = mean_merror
#             best_params['num_leaves'] = num_leaves
#             best_params['max_depth'] = max_depth
# params['num_leaves'] = best_params['num_leaves']
# params['max_depth'] = best_params['max_depth']
# # overfitting
# print("調参2：降低overfit")
# for max_bin in range(1, 155, 25):
#     for min_data_in_leaf in range(10, 101, 10):
#         params['max_bin'] = max_bin
#         params['min_data_in_leaf'] = min_data_in_leaf
#
#         cv_results = lgb.cv(
#             params,
#             lgb_train,
#             seed=42,
#             nfold=3,
#             early_stopping_rounds=3,
#             # verbose_eval=True
#         )
#
#         mean_merror = pd.Series(cv_results['l2-mean']).min()
#
#         if mean_merror < min_merror:
#             min_merror = mean_merror
#             best_params['max_bin'] = max_bin
#             best_params['min_data_in_leaf'] = min_data_in_leaf
#
# params['min_data_in_leaf'] = best_params['min_data_in_leaf']
# params['max_bin'] = best_params['max_bin']
# print params
# print("調参3：降低overfit")
# for feature_fraction in [i / 10.0 for i in range(0, 11, 2)]:
#     for bagging_fraction in [i / 10.0 for i in range(0, 11, 2)]:
#         for bagging_freq in range(0, 50, 5):
#             params['feature_fraction'] = feature_fraction
#             params['bagging_fraction'] = bagging_fraction
#             params['bagging_freq'] = bagging_freq
#
#             cv_results = lgb.cv(
#                 params,
#                 lgb_train,
#                 seed=42,
#                 nfold=3,
#                 early_stopping_rounds=3,
#                 # verbose_eval=True
#             )
#
#             mean_merror = pd.Series(cv_results['l2-mean']).min()
#             boost_rounds = pd.Series(cv_results['l2-mean']).argmin()
#
#             if mean_merror < min_merror:
#                 min_merror = mean_merror
#                 best_params['feature_fraction'] = feature_fraction
#                 best_params['bagging_fraction'] = bagging_fraction
#                 best_params['bagging_freq'] = bagging_freq
#
# params['feature_fraction'] = best_params['feature_fraction']
# params['bagging_fraction'] = best_params['bagging_fraction']
# params['bagging_freq'] = best_params['bagging_freq']
# print params
# print("調参4：降低overfit")
# for lambda_l1 in [i / 10.0 for i in range(0, 11, 2)]:
#     for lambda_l2 in [i / 10.0 for i in range(0, 11, 2)]:
#         for min_split_gain in [i / 10.0 for i in range(0, 11, 2)]:
#             params['lambda_l1'] = lambda_l1
#             params['lambda_l2'] = lambda_l2
#             params['min_split_gain'] = min_split_gain
#
#             cv_results = lgb.cv(
#                 params,
#                 lgb_train,
#                 seed=42,
#                 nfold=3,
#                 early_stopping_rounds=3,
#                 # verbose_eval=True
#             )
#
#             mean_merror = pd.Series(cv_results['l2-mean']).min()
#
#             if mean_merror < min_merror:
#                 min_merror = mean_merror
#                 best_params['lambda_l1'] = lambda_l1
#                 best_params['lambda_l2'] = lambda_l2
#                 best_params['min_split_gain'] = min_split_gain
#
# params['lambda_l1'] = best_params['lambda_l1']
# params['lambda_l2'] = best_params['lambda_l2']
# params['min_split_gain'] = best_params['min_split_gain']
#
# print(model_param)
# print(params)

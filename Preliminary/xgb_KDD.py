from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit

params = {
    'max_depth': range(7, 12, 2),
    'min_child_weight': [2],
    'gamma': [i / 10.0 for i in range(8, 9)],
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)],
    'reg_alpha': [0, 0.001, 0.001],
    'learning_rate': [0.001, 0.01, 0.1],
    'n_estimators': [1000, 2000, 3000]
}
cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=2)
CLF = GridSearchCV(
    estimator=xgb.XGBRegressor(learning_rate=0.001, n_estimators=3000, max_depth=10, min_child_weight=2,
                               reg_alpha=0.001, gamma=0.6, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
                               seed=27), param_grid=params, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv, verbose=6)



CLF.fit(df_sel.values, ans_final_original)
print('best params:\n', CLF.best_params_)
mean_scores = np.array(CLF.cv_results_['mean_test_score'])
print('mean score', mean_scores)
print('best score', CLF.best_score_)
return CLF.best_params_


best_param = {'colsample_bytree': 0.6,
 'gamma': 0.8,
 'learning_rate': 0.1,
 'max_depth': 7,
 'min_child_weight': 2,
 'n_estimators': 3000,
 'reg_alpha': 0,
 'subsample': 0.7}
reg = xgb.XGBRegressor(**best_param)
reg.fit(df_sel.values, ans_final_original, verbose=100)
reg.predict(df_sel.values)
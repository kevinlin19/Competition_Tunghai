import pandas as pd
import tsfresh as ts

df_ans = pd.read_csv('./ans.csv', names=['ans'])
df_ar = pd.read_csv('./feature_ar.csv', header=0)
df_col = pd.read_csv('./feature_me.csv', header=0)
df_mean = pd.read_csv('./feature_mean.csv', header=0)
df_mean_log_total = pd.read_csv('./feature_mean_log_total.csv', header=0)


# df_rel = ts.feature_selection.relevance.calculate_relevance_table(df_ar, df_ans.ans, ml_task='regression')
df_sel_1 = ts.feature_selection.selection.select_features(df_ar, df_ans.ans, fdr_level=0.5) #0.5
df_sel_2 = ts.feature_selection.selection.select_features(df_col, df_ans.ans, fdr_level=0.5) #0.4
df_sel_3 = ts.feature_selection.selection.select_features(df_mean, df_ans.ans, fdr_level=0.2) #0.2
df_sel_4 = ts.feature_selection.selection.select_features(df_mean_log_total, df_ans.ans)
df = pd.concat([df_sel_1, df_sel_2, df_sel_3, df_sel_4], axis=1)

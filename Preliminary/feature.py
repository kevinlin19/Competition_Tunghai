import pandas as pd
import tsfresh as ts
from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler()
feature_scale = scale.fit_transform(df_feature)
pd.DataFrame(feature_scale)
df_rel = ts.feature_selection.relevance.calculate_relevance_table(pd.DataFrame(feature_scale), pd.Series(ans_final), ml_task='regression', fdr_level=0.1)
df_sel = ts.feature_selection.selection.select_features(pd.DataFrame(feature_scale), pd.Series(ans_final), fdr_level=0.1)
df_rel = ts.feature_selection.relevance.calculate_relevance_table(df_feature, pd.Series(ans_final), ml_task='regression', fdr_level=0.6)
df_sel = ts.feature_selection.selection.select_features(df_feature, pd.Series(ans_final), fdr_level=0.6)
df_rel[df_rel.relevant == True]
df_rel[df_rel.relevant == True]

df_sel.columns
'''
                                                    type   p_value  relevant  
feature                                                                       
one_two_MEAN__fft_coefficient__coeff_94__attr_"...  real  0.000018      True  
one_three_MEAN__fft_coefficient__coeff_36__attr...  real  0.000022      True  
two_three_MEAN__fft_coefficient__coeff_76__attr...  real  0.000041      True  
one_two_four_MEAN__fft_coefficient__coeff_94__a...  real  0.000045      True  
one_three_MEAN__fft_coefficient__coeff_8__attr_...  real  0.000075      True  
 
'''
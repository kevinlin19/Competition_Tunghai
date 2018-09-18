import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import *
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            center=0,
            annot=True)
plt.show()

plot_acf(df_con.iloc[:2000,0].values)
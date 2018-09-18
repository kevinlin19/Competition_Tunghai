import glob
import pandas as pd
import numpy as np
from tsfresh.feature_extraction.feature_calculators import fft_coefficient
from scipy.fftpack import fft, rfft, irfft

path='./data/data_odiginal/test'
all_files = glob.glob(path + '/*.xls')
test=pd.DataFrame(np.nan,index=range(75000), columns=['id','time','1st','2nd','3rd','4th'])
index=0
for file in all_files:
    temp=pd.read_excel(file,header=None)
    test.iloc[(index*7500):((index+1)*7500),2:6]=temp.loc[:7499,:].values
    test.iloc[(index*7500):((index+1)*7500),0]=index
    test.iloc[(index*7500):((index+1)*7500),1]=range(1,7501)
    index+=1

fft_coefficient(test[test.id==0]['2nd'], param={'coeff':76 , 'attr': "imag"})
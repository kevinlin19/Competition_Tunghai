import pickle
from keras.models import load_model
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# load kai & hang data
df = pd.read_csv('./data/final_feature.csv', header=0)
df_ans = pd.read_csv('./data/ans.csv', names=['ans'])
# scale
scale = MinMaxScaler((0, 2))
X_train = scale.fit_transform(df)
y_scale = MinMaxScaler()
Y_train = y_scale.fit_transform(df_ans.ans.values.reshape(-1, 1))


def custom_sigmoid(x):
    return 2*K.sigmoid(x)
get_custom_objects().update({'custom_sigmoid': Activation(custom_sigmoid)})
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


model_jacky = pickle.load(open("./model_save/xgboost.pickle.dat", "rb"))
model_kai = load_model('C:/Users/S.K.LIN/Desktop/Preliminary/kai.h5', custom_objects={'root_mean_squared_error':root_mean_squared_error})
model_hang = load_model('./model_save/hang.h5')


ans_kai = model_kai.predict(X_train)
y_pred = y_scale.inverse_transform(ans_kai)
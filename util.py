from sklearn.metrics import mean_squared_error
from math import sqrt
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import tsfresh as ts


def RMSE(y_actual, y_predicted):
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms

def custom_sigmoid(x):
    return 2*K.sigmoid(x)
get_custom_objects().update({'custom_sigmoid': Activation(custom_sigmoid)})

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


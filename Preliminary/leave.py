# ----------------------------------------Data------------------------------------------
import pandas as pd
import numpy as np
import glob

# load data
path = './data'
all_files = glob.glob(path + "/*.xls")
ans = []
file_list = []
AR_list = []
for file in all_files:
    df = pd.read_excel(file, header=None)
    ans_ = df.iloc[-1,0]
    df = df[:-1]
    AR = df[9:].values - df[:-9].values
    AR_list.append(AR)
    file_list.append(df)
    ans.append(ans_)

# X_input
X_ar = np.array(AR_list)
X = X_ar * 10e+5 +5

# Y
ans_final = []
for an in ans:
    ans_final.append(an[9:])
ans_final = np.array(ans_final).astype(np.float32)
Y = ans_final.reshape(-1, 1)

# ---------------------------------------------Model--------------------------------------------
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, BatchNormalization, Conv1D
from keras.layers import Dropout
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model

# Customize
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def custom_tanh(x):
    return 10*K.sigmoid(x)
get_custom_objects().update({'custom_tanh': Activation(custom_tanh)})

#ans
ANS_list = []
Original = []
# Build
for i in range(40):
    x_test = X[i].reshape(1, 7491,4)
    x_train = np.delete(X, i, 0)
    y_test = Y[i].reshape(1, 1)
    y_train = np.delete(Y, i, 0)
    model = Sequential()
    model.add(LSTM(input_shape=(7491, 4), units=10, activation=custom_tanh, return_sequences=True))
    model.add(Conv1D(filters=16, strides=2, kernel_size=10))
    model.add(BatchNormalization())
    model.add(LSTM(units=10, activation=custom_tanh, return_sequences=True))
    model.add(Conv1D(filters=8, strides=2, kernel_size=10))
    model.add(BatchNormalization())
    model.add(LSTM(units=5, activation=None, return_sequences=True))
    model.add(Conv1D(filters=4, strides=1, kernel_size=5, activation=custom_tanh))
    model.add(BatchNormalization())
    model.add(LSTM(units=5, activation=custom_tanh, return_sequences=True))
    model.add(Conv1D(filters=2, strides=1, kernel_size=5, activation=custom_tanh))
    model.add(Flatten())
    model.add((Dense(320)))
    model.add(Dropout(0.5))
    model.add(Dense(80))
    model.add(BatchNormalization())
    model.add(Dense(20))
    model.add((Dense(1, activation='sigmoid')))
    # model.summary()
    # tarining
    filepath = 'weights.h5'
    chackpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    callbacks = [chackpoint, reduce_lr]
    model.compile(optimizer='rmsprop', loss=root_mean_squared_error)
    model.fit(x=x_train, y=y_train, epochs=150, batch_size=4, validation_data=[x_test, y_test], callbacks=callbacks)

    del model
    model = load_model('./weights.h5', custom_objects={'root_mean_squared_error':root_mean_squared_error})
    ANS_list.append(model.predict(x_test))
    Original.append(y_test)
    del model
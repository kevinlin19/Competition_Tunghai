import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
# keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, BatchNormalization, Conv1D, Dropout
import keras.backend as K
from keras.optimizers import RMSprop
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import regularizers
import glob

rmse = []
def RMSE(y_actual, y_predicted):
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms


# load data
import pandas as pd
import numpy as np
import glob
from tsfresh import extract_features


regularizer = 0.00004

# Torr
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

# X: df_con
df_con = pd.concat(file_list, ignore_index=True)
df_con = df_con.astype('float32')
df_con.columns = ['col_1', 'col_2', 'col_3', 'col_4']
df_con['one_two_MEAN'] = df_con[['col_1', 'col_2']].mean(axis=1)
df_con['one_three_MEAN'] = df_con[['col_1', 'col_3']].mean(axis=1)
df_con['total_STD_log'] = np.log(df_con[['col_1', 'col_2', 'col_3', 'col_4']].std(axis=1))
df_con['total_RANGE_log'] = np.log(df_con[['col_1', 'col_2', 'col_3', 'col_4']].max(axis=1) - df_con[['col_1', 'col_2', 'col_3', 'col_4']].min(axis=1))
df_con = df_con[['one_two_MEAN', 'one_three_MEAN', 'total_STD_log', 'total_RANGE_log']]
df_con = df_con.astype(np.float32)

x_scale = MinMaxScaler((0, 1))
X = x_scale.fit_transform(df_con)
X_train = np.zeros(shape=[40, 7500, 4])
for i in range(40):
    X_train[i] = X[i*7500:(i+1)*7500]

# Y_train
ans_final = []
for an in ans:
    ans_final.append(an[9:])
ans_final = np.array(ans_final).astype(np.float32)
Y_train = ans_final.reshape(-1, 1)

# X_train_scale = (X_train - X_train.min()) / (X_train.max() - X_train.min())
Y_train_scale = (Y_train - Y_train.min()) / (Y_train.max() - Y_train.min())

# model custom
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def custom_tanh(x):
    return 10*K.sigmoid(x)
get_custom_objects().update({'custom_tanh': Activation(custom_tanh)})

for i in range(40):
    x_train = np.delete(X_train, i, axis=0)
    x_test = X_train[i].reshape(1, 7500, 4)
    y_train = np.delete(Y_train_scale, i, axis=0)
    y_test = Y_train_scale[i].reshape(1, -1)
    # model
    model = Sequential()
    model.add(LSTM(input_shape=(7500, 4), units=15, activation='sigmoid', return_sequences=True, kernel_regularizer=regularizers.l2(regularizer)))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=16, strides=1, kernel_size=60, activation='elu'))
    model.add(BatchNormalization(beta_regularizer=regularizers.l2(regularizer), gamma_regularizer=regularizers.l2(regularizer)))
    model.add(LSTM(units=10, activation='sigmoid', return_sequences=True, kernel_regularizer=regularizers.l2(regularizer)))
    model.add(Conv1D(filters=8, strides=1, kernel_size=30, activation='elu'))
    model.add(BatchNormalization(beta_regularizer=regularizers.l2(regularizer),
                           gamma_regularizer=regularizers.l2(regularizer)))
    model.add(LSTM(units=5, activation='sigmoid', return_sequences=True, kernel_regularizer=regularizers.l2(regularizer)))
    model.add(Conv1D(filters=4, strides=1, kernel_size=15, activation='elu'))
    model.add(BatchNormalization(beta_regularizer=regularizers.l2(regularizer),
                           gamma_regularizer=regularizers.l2(regularizer)))
    model.add(LSTM(units=5, activation='sigmoid', return_sequences=True, kernel_regularizer=regularizers.l2(regularizer)))
    model.add(Conv1D(filters=2, strides=1, kernel_size=5, activation='elu'))
    model.add(Flatten())
    model.add((Dense(320,
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer), activation='elu')))
    model.add(Dropout(0.5))
    model.add(Dense(80,
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer), activation='elu'))
    model.add(BatchNormalization(beta_regularizer=regularizers.l2(regularizer),
                           gamma_regularizer=regularizers.l2(regularizer)))
    model.add(Dropout(0.5))
    model.add(Dense(20,kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer), activation='elu'))
    model.add((Dense(1, activation='sigmoid')))

    # Compile
    filepath = 'weights.h5'
    chackpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00000001)
    callbacks = [chackpoint, reduce_lr]
    model.compile(optimizer=RMSprop(lr=0.0001), loss=root_mean_squared_error)
    his = model.fit(x=x_train, y=y_train, epochs=100, batch_size=4, validation_data=[x_test, y_test], callbacks=callbacks)
    model.save('./model_{}.h5'.format(i))

    # rmse
    a = model.predict(x_test)
    a = a * (Y_train.max() - Y_train.min()) + Y_train.min()
    ans = RMSE(Y_train[i], a)
    print('rmse:', ans)
    rmse.append(ans)
    del model
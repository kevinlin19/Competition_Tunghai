from keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from keras.optimizers import RMSprop
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from sklearn.model_selection import train_test_split

import keras.backend as k
from keras.layers import (Input, Concatenate, BatchNormalization, Dropout, Dense)
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

y = []
y_prediction = []
rmse = []
# load data
df = pd.read_csv('./final_feature.csv', header=0)
df_ans = pd.read_csv('./ans.csv', names=['ans'])
# Scale
scale = MinMaxScaler((0, 2))
X_train = scale.fit_transform(df)
y_scale = MinMaxScaler()
Y_train = y_scale.fit_transform(df_ans.ans.values.reshape(-1, 1))
# custom model
def RMSE(y_actual, y_predicted):
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms

def custom_sigmoid(x):
    return 2*K.sigmoid(x)
get_custom_objects().update({'custom_sigmoid': Activation(custom_sigmoid)})

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
# model function
def get_model(input_dim, output_dim):
    # dropout = 0.5
    # regularizer = 0.00004
    # main_input = Input(shape=(input_dim,), dtype='float32', name='main_input')
    #
    # x = Dense(32, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(main_input)
    # x = Dropout(dropout)(x)
    # x = concatenate([main_input, x])
    # x = Dense(32, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    # x = BatchNormalization(beta_regularizer=regularizers.l2(regularizer),
    #                        gamma_regularizer=regularizers.l2(regularizer)
    #                        )(x)
    # x = Dropout(dropout)(x)
    # x = Dense(16, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    # x = Dropout(dropout)(x)
    #
    # x = Dense(8, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    # x = Dropout(dropout)(x)
    # x = Dense(output_dim, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    # ...................................................................................................
    # dropout = 0.5
    # regularizer = 0.00004
    # main_input = Input(shape=(input_dim,), dtype='float32', name='main_input')
    #
    # x = Dense(32, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(main_input)
    # x1 = Dropout(dropout)(x)
    # x = concatenate([main_input, x1])
    # x = Dense(32, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    # x = BatchNormalization(beta_regularizer=regularizers.l2(regularizer),
    #                        gamma_regularizer=regularizers.l2(regularizer)
    #                        )(x)
    # x2 = Dropout(dropout)(x)
    # x = concatenate([x1, x2])
    # x = Dense(16, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    # x = Dropout(dropout)(x)
    # x = Dense(8, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    # x = Dropout(dropout)(x)
    # x = Dense(output_dim, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    # ....................................................................................................
    # dropout = 0.5
    # regularizer = 0.00004
    # main_input = Input(shape=(input_dim,), dtype='float32', name='main_input')
    #
    # x = Dense(64, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(main_input)
    # x = Dropout(dropout)(x)
    # x = concatenate([main_input, x])
    # x = Dense(32, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    # x = BatchNormalization(beta_regularizer=regularizers.l2(regularizer),
    #                        gamma_regularizer=regularizers.l2(regularizer)
    #                        )(x)
    # x = Dropout(dropout)(x)
    # x = Dense(16, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    # x = Dropout(dropout)(x)
    # x = Dense(8, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    # x = Dropout(dropout)(x)
    # x = Dense(output_dim, activation='sigmoid',
    #           kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    # ....................................................................................................
    dropout = 0.5
    regularizer = 0.00004
    main_input = Input(shape=(input_dim,), dtype='float32', name='main_input')

    x = Dense(32, activation=custom_sigmoid,
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(main_input)
    x = Dropout(dropout)(x)
    x = concatenate([main_input, x])
    x = Dense(32, activation=custom_sigmoid,
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = BatchNormalization(beta_regularizer=regularizers.l2(regularizer),
                           gamma_regularizer=regularizers.l2(regularizer)
                           )(x)
    x = Dropout(dropout)(x)
    x = Dense(16, activation=custom_sigmoid,
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = Dropout(dropout)(x)
    x = Dense(8, activation=custom_sigmoid,
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = Dropout(dropout)(x)
    x = Dense(output_dim, activation='sigmoid',
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    model = Model(inputs=[main_input], outputs=[x])
    return model

# model
# for i in range(40):
#     x_train = np.delete(X_train, i, axis=0)
#     x_test = X_train[i].reshape(1, -1)
#     y_train = np.delete(Y_train, i, axis=0)
#     y_test = Y_train[i].reshape(1, -1)
#     model = get_model(14, 1)
#     filepath = 'weights.h5'
#     chackpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False,
#                                  mode='min', period=1)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00000001)
#     callbacks = [chackpoint, reduce_lr]
#     model.compile(optimizer=RMSprop(lr=0.0001), loss='mse')
#     model.fit(x=x_train, y=y_train, epochs=1200, batch_size=4, validation_data=[x_test, y_test], callbacks=callbacks, shuffle=True)
#     # predict
#     a = model.predict(x_test)
#     y_pred = y_scale.inverse_transform(a)
#     # rmse
#     ans = RMSE(df_ans.ans.values[i].reshape(-1, 1), y_pred)
#     y.append(df_ans.ans.values[i].reshape(-1, 1))
#     y_prediction.append(y_pred)
#     rmse.append(ans)
#     del model
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=.8)


model = get_model(14, 1)
filepath = 'weights.h5'
chackpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00000001)
callbacks = [chackpoint, reduce_lr]
model.compile(optimizer=RMSprop(lr=0.0001), loss=root_mean_squared_error)
model.fit(x=X_train, y=Y_train, epochs=250, batch_size=4, callbacks=callbacks, shuffle=True)
model.save('./kai.h5')
a = model.predict(X_train)
y_pred = y_scale.inverse_transform(a)
# rmse
ans = RMSE(df_ans.ans.values.reshape(-1, 1), y_pred)
ans

df_test = pd.read_csv('./test_feature_selection_kai.csv', header=0)
X_test = scale.transform(df_test)
pd.DataFrame(y_scale.inverse_transform(model.predict(X_test))).to_csv('./test_kai.csv')


x_train, x_test, y_train, y_test = train_test_split(df.values, df_ans.ans.values.reshape(-1, 1), train_size=.8)
# def get_keras_model(input_dim: int, dropout_rate: float=0.5, l2_strength: float=0.0001, units: list=(64, 64, 32, 16, 8), lr: float=0.00005):
#     inp = Input(shape=(input_dim,), dtype='float32')
#     x = Dense(units=units[0], activation='selu', kernel_initializer='lecun_uniform',
#               kernel_regularizer=l2(l2_strength))(inp)
#     x = Dropout(dropout_rate)(x)
#     x = Concatenate()([inp, x])
#     x = Dense(units=units[1], activation='selu', kernel_initializer='lecun_uniform',
#               kernel_regularizer=l2(l2_strength))(x)
#     x = BatchNormalization(beta_regularizer=l2(l2_strength), gamma_regularizer=l2(l2_strength))(x)
#     x = Dropout(dropout_rate)(x)
#     x = Dense(units=units[2], activation='selu', kernel_initializer='lecun_uniform',
#               kernel_regularizer=l2(l2_strength))(x)
#     x = Dropout(dropout_rate)(x)
#     x = Dense(units=units[3], activation='selu', kernel_initializer='lecun_uniform',
#               kernel_regularizer=l2(l2_strength))(x)
#     x = Dropout(dropout_rate)(x)
#     x = Dense(units=1, activation='linear', kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_strength))(x)
#     model = Model(inputs=inp, outputs=x)
#     optimizer = Adam(lr=lr, clipvalue=1, clipnorm=1)
#     loss = root_mean_squared_error
#     model.compile(loss=loss, optimizer=optimizer)
#     return model


model_ = get_keras_model(input_dim=23)
model_.fit(x=df.values, y=df_ans.ans.values.reshape(-1, 1), epochs=10000)
RMSE(df_ans.ans.values.reshape(-1, 1), model_.predict(df.values))

p = []
fot i in range(40):
    model_ = get_keras_model(input_dim=23)
    model_.fit(x=np.delete(df.values, i, axis=0), y=df_ans.ans.values[i].reshape(-1, 1), epochs=10000)
    r = RMSE(df_ans.ans.values[i].reshape(-1, 1), model_.predict(np.delete(df.values, i, axis=0)))
    pre = model_.predict(np.delete(df.values, i, axis=0))
    p.append(pre)

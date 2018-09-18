from keras.models import Sequential
from keras.layers import MaxPooling1D, LSTM, Dense, ConvLSTM2D, Flatten, BatchNormalization, Conv2D, Conv1D, GlobalMaxPooling2D, Reshape
from keras.layers import LeakyReLU, Dropout
import keras.backend as K
from keras.optimizers import Adam, RMSprop
import numpy as np
from data_preprocessing import *
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras import regularizers
from keras.initializers import glorot_normal
from keras.constraints import max_norm

regularizer = 0.00004

rmse = []
y = []
yp = []
# x_train, x_test, y_train, y_test = train_test_split(X_10_ar, ans_final_original, test_size=0.2, shuffle=True)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# def custom_tanh(x):
#     return 10*K.tanh(x)
def custom_tanh(x):
    return 10*K.sigmoid(x)
get_custom_objects().update({'custom_tanh': Activation(custom_tanh)})

# CRNN_train = np.zeros(shape=(32, 7500-100+1, 1, 100, 6), dtype=np.float32)
# for i in range(32):
#     for j in range(7500-100+1):
#         CRNN_train[i][j][0] = x_train_NN[i][j: j+100]
#
# x_train_NN.shape
'''
l1 = model_auto.add(Conv2D(input_shape=(7500, 6, 1), filters=3, kernel_size=(6, 6), strides=(4,1)))
flatten1 = model_auto.add(Flatten())
l2 = model_auto.add(Dense(units=2400, activation='relu'))
auto = model_auto.add(Dense(units=800, activation='relu'))
l_2 = model_auto.add(Dense(units=2400, activation='relu'))
l_3 = model_auto.add(Dense(units=5622, activation='relu'))
decoder = model_auto.add(Dense(units=7500*6, activation='relu'))
l_1 = model_auto.add(Reshape((7500, 6, 1)))
model_auto.summary()
model_auto.compile(optimizer='adam', loss='mse')
his = model_auto.fit(x=x_train_NN[:8].reshape(8, 7500, 6, 1), y=x_train_NN[:8].reshape(8, 7500, 6, 1), batch_size=4, epochs=10, validation_data=[x_test_NN.reshape(8, 7500, 6, 1), x_test_NN.reshape(8, 7500, 6, 1)])
'''
'''
model.add(ConvLSTM2D(input_shape=(CRNN_train.shape[1:]), activation='relu', filters=2, kernel_size=(10,6), strides=(1, 1), padding='valid', data_format='channels_first', return_sequences=True))
model.add(ConvLSTM2D(activation='relu', filters=4, kernel_size=(10,1), strides=(1, 1), padding='valid', data_format='channels_first', return_sequences=True))
model.add(BatchNormalization())
model.add(ConvLSTM2D(activation='relu', filters=2, kernel_size=(10,1), strides=(1, 1), padding='valid', data_format='channels_first'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='relu'))
'''
for i in range(40):
    x_train = np.delete(X_10_ar, i, axis=0)
    x_test = X_10_ar[i].reshape(1, 7491, 4)
    y_train = np.delete(ans_final_original, i, axis=0)
    y_test = ans_final_original[i].reshape(-1, 1)
    model = Sequential()
    model.add(LSTM(input_shape=(7491, 4), units=10, return_sequences=True))
    model.add(Conv1D(filters=16, strides=2, kernel_size=10))
    model.add(BatchNormalization(beta_regularizer=regularizers.l2(regularizer), gamma_regularizer=regularizers.l2(regularizer)))
    model.add(LSTM(units=10, activation=custom_tanh, return_sequences=True))
    model.add(Conv1D(filters=8, strides=2, activation=custom_tanh, kernel_size=10))
    model.add(BatchNormalization(beta_regularizer=regularizers.l2(regularizer), gamma_regularizer=regularizers.l2(regularizer)))
    model.add(LSTM(units=5, activation=custom_tanh, return_sequences=True))
    model.add(Conv1D(filters=4, strides=1, activation=custom_tanh, kernel_size=5))
    model.add(BatchNormalization(beta_regularizer=regularizers.l2(regularizer), gamma_regularizer=regularizers.l2(regularizer)))
    model.add(LSTM(units=5, activation=custom_tanh, return_sequences=True))
    model.add(Conv1D(filters=2, strides=1, activation=custom_tanh, kernel_size=5))
    model.add(Flatten())
    model.add((Dense(320)))
    model.add(Dropout(0.5))
    model.add(Dense(80, activation='linear', kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer)))
    model.add(BatchNormalization(beta_regularizer=regularizers.l2(regularizer), gamma_regularizer=regularizers.l2(regularizer)))
    # model.add(Dropout(0.5))
    model.add(Dense(20, activation='linear'))
    model.add((Dense(1, activation='sigmoid')))
    model.summary()

    '''
    model.add(Conv2D(input_shape=(7500, 6, 1), filters=4, kernel_size=(100, 6), activation='relu'))
    model.add(Reshape((7401, 4, 1)))
    model.add(Conv2D(activation='relu', filters=8, kernel_size=(1000, 4)))
    model.add(Reshape((6402, 8, 1)))
    model.add(Conv2D(activation='elu', filters=4, kernel_size=(3000, 8)))
    model.add(BatchNormalization())
    model.add(Reshape((3403, 4, 1)))
    model.add(Conv2D(activation='relu', filters=2, kernel_size=(3000, 4)))
    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(units=4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='relu'))
    '''

    '''
    model.add(Conv1D(input_shape=(7491, 4), filters=8, strides=5, kernel_size=4))
    model.add(LeakyReLU())
    model.add(Conv1D(filters=16, strides=2, kernel_size=8))
    model.add(LeakyReLU())
    model.add(MaxPooling1D())
    model.add(Conv1D(filters=8, strides=1, kernel_size=16, activation=custom_tanh))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=4, strides=1, kernel_size=8))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(units=256))
    model.add(LeakyReLU())
    model.add(Dense(units=128, activation=custom_tanh))
    model.add(BatchNormalization())
    model.add(Dense(units=32))
    model.add(LeakyReLU())
    model.add(Dense(units=8))
    model.add(Dense(units=1, activation='softmax'))
    '''
    '''
    model.add(Conv1D(input_shape=(7491, 4), filters=8, strides=5, kernel_size=4, activation=custom_tanh))
    model.add(Conv1D(filters=16, strides=2, kernel_size=8, activation=custom_tanh))
    model.add(MaxPooling1D())
    model.add(Conv1D(filters=8, strides=1, kernel_size=16, activation=custom_tanh))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=4, strides=1, kernel_size=8, activation=custom_tanh))
    model.add(Flatten())
    model.add(Dense(units=256, activation=custom_tanh))
    model.add(Dense(units=128, activation=custom_tanh))
    model.add(BatchNormalization())
    model.add(Dense(units=32, activation=custom_tanh))
    model.add(Dense(units=8, activation=custom_tanh))
    model.add(Dense(units=1, activation='sigmoid'))
    '''
    # model.summary()
    filepath = 'weights.h5'
    chackpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0000001)
    callbacks = [chackpoint, reduce_lr]
    model.compile(optimizer=RMSprop(lr=0.001), loss=root_mean_squared_error)
    # model.compile(optimizer=RMSprop(lr=0.0001 ), loss=smape)
    his = model.fit(x=x_train, y=y_train, epochs=50, batch_size=4, validation_data=[x_test, y_test], callbacks=callbacks)
    # his = model.fit(x=x_train, y=y_train, epochs=10, batch_size=4)
    # for _ in range(10):
    #     num = np.random.randint(low=0, high=31)
    #     print(_)
    #     print(y_train[num])
    #     # his = model.fit(x=x_train[num].reshape(1, 7491, 4), y=y_train[num].reshape(1, 1), epochs=1, batch_size=4, validation_data=[x_test, y_test], callbacks=callbacks)
    #     his = model.fit(x=x_train[num].reshape(1, 7491, 4), y=y_train[num].reshape(1, 1), epochs=1, batch_size=1)
    #     print(model.predict(x=x_train[0].reshape(1, 7491, 4)))
    #     print(y_train[0])
    # model.evaluate(x_train, y_train)
    # model.predict(x=x_train[2].reshape(1, 7491, 4))
    # a
    # a_ = RMSE(y_train, a)
    #
    # model.save('./model_crnn_0.0734_mse_new_sigmoid.h5')
    # model_2 = load_model('./model_2.h5', custom_objects={'root_mean_squared_error':root_mean_squared_error})
    a = model.predict(x_test)
    ans = RMSE(ans_final_original[i], a) #1.6667090490038663
    rmse.append(ans)
    y.append(ans_final_original[i])
    yp.append(a)
    print('rmse:', ans)




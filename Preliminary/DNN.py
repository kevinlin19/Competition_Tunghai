from keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
import keras
from keras.models import Model
from keras.optimizers import Adam
from combine import *

def smape_error(y_true, y_pred):
    return K.mean(K.clip(K.abs(y_pred - y_true),  0.0, 1.0), axis=-1)

def RMSE(y_actual, y_predicted):
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms

scale = MinMaxScaler()
x_train = scale.fit_transform(df)
y_scale = MinMaxScaler()
y_train = y_scale.fit_transform(df_ans.ans.values.reshape(-1, 1))


def get_model(input_dim, output_dim):
    dropout = 0.5
    regularizer = 0.00004
    main_input = Input(shape=(input_dim,), dtype='float32', name='main_input')

    x = Dense(200, activation='sigmoid',
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(main_input)
    x = Dropout(dropout)(x)
    x = concatenate([main_input, x])
    x = Dense(200, activation='sigmoid',
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = BatchNormalization(beta_regularizer=regularizers.l2(regularizer),
                           gamma_regularizer=regularizers.l2(regularizer)
                           )(x)
    x = Dropout(dropout)(x)
    x = Dense(100, activation='sigmoid',
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = Dropout(dropout)(x)

    x = Dense(200, activation='sigmoid',
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = Dropout(dropout)(x)
    x = Dense(output_dim, activation='sigmoid',
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)

    model = Model(inputs=[main_input], outputs=[x])
    model.compile(loss='mse', optimizer='rmsprop')
    return model

model = get_model(14, 1)
model.summary()
# model.compile(loss=smape_error, optimizer=Adam(lr=0.00001))
# model.compile(loss='mse', optimizer=Adam(lr=0.00001))
model.fit(x_train, y_train, epochs=1000, batch_size=4, shuffle=True)
a = model.predict(x_train)
y_pred = y_scale.inverse_transform(a)

RMSE(df_ans.ans.values, y_pred)
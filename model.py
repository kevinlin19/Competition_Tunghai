from keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate
from keras import regularizers
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import keras.backend as K

def custom_sigmoid(x):
    return 2*K.sigmoid(x)
# get_custom_objects().update({'custom_sigmoid': Activation(custom_sigmoid)})


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
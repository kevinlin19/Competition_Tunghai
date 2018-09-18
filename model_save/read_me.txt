load_model('./model_2.h5', custom_objects={'root_mean_squared_error':root_mean_squared_error})
def custom_sigmoid(x):
    return 2*K.sigmoid(x)
get_custom_objects().update({'custom_sigmoid': Activation(custom_sigmoid)})

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
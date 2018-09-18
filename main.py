# from util import *
# from model import *
from data_preprocessing import *
import pickle
import pandas as pd
import os

# model = get_model(14, 1)
# model.summary()
# filepath = 'weights.h5'
# chackpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00000001)
# callbacks = [chackpoint, reduce_lr]
# model.compile(optimizer=RMSprop(lr=0.0001), loss='mse')
# model.fit(x=X_train, y=Y_train, epochs=1000, batch_size=4, validation_data=[x_test, y_test], callbacks=callbacks, shuffle=True)
# a = model.predict(x_train)
# y_pred = y_scale.inverse_transform(a)
# RMSE(df_ans.ans.values, y_pred)

# read file name
def read_file_name(path = './data/data_odiginal/test'):
    file_name = []
    for file in os.listdir(path):
        if file.endswith(".xls"):
            file_name.append(file)
    return file_name

# load model
def load_predict_xgb_model(model_path="./model_save/xgboost.pickle.dat", X_jacky_input=None):
    loaded_model = pickle.load(open(model_path, "rb"))
    # prediction
    test_ans = loaded_model.predict(X_jacky_input)
    # to csv
    df_test_answer = pd.DataFrame({'file_name':file_name, 'answer':test_ans})[['file_name', 'answer']]
    print(df_test_answer)
    df_test_answer.to_csv('./answer/final_answer.csv')

if __name__ == '__main__':
    input_data = xgb_data()
    file_name = read_file_name(path = './data/data_odiginal/test')
    load_predict_xgb_model(X_jacky_input=input_data)
import pyodbc as pbo
from DataBase.MSSQLServerConnection import server_connection as sc
from DataBase.OracleConnection import OracleConnection as Oc
from keras import callbacks
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import * 
import keras
from keras.models import Sequential
from keras.layers import *
import sqlalchemy
from sqlalchemy import create_engine as c_e


def fetch_data(table_name: str = "RECOMMENDERCALC.SEP_TERMINALS_PIVOT"):
    database_connection = Oc()
    database_connection.set_query('select * from {} sample(10)'.format(table_name))
    data = database_connection.fetch_data()
    data = pd.DataFrame(data)
    return data


def fetch_lbl_data():
    data = fetch_data()
    [_, n] = np.shape(data)
    data.rename(columns={0: 'TerminalKey'}, inplace=True)
    print('first column renamed as ... TerminalKey')
    data.rename(columns={n-1: 'Guild'}, inplace=True)
    print('last column renamed as ... Guild')
    data.fillna(0, inplace=True)
    print(data)
    print("Done Reading Data")
    return


def fetch_other_data():
    data = fetch_data(table_name="PFM_PIVOT")
    data.rename(columns={0: 'TerminalKey'}, inplace=True)
    print('first column renamed as ... TerminalKey')
    data.fillna(0, inplace=True)
    return data


total = list()
data_scaler = MinMaxScaler()


def create_model(number_of_dims=399, lr=0.1):
    """here we create a neural network model based on extracted features of data"""
    model_tf = Sequential()
    model_tf.add(Dense(round(0.70 * number_of_dims), input_dim=number_of_dims, activation='relu', name='layer_0'))
    model_tf.add(Dense(round(0.90 * number_of_dims), activation='relu', name='layer_1'))
    model_tf.add(Dense(round(1.20 * number_of_dims), activation='sigmoid', name='layer_2'))
    model_tf.add(Dense(round(0.85 * number_of_dims), activation='sigmoid', name='layer_4'))
    model_tf.add(Dense(round(0.60 * number_of_dims), activation='relu', name='layer_6'))
    model_tf.add(Dense(round(0.30 * number_of_dims), activation='relu', name='layer_7'))
    model_tf.add(Dense(1, activation='softmax', name='result'))
    opt = keras.optimizers.Adam(learning_rate=lr)
    model_tf.compile(loss='binary_crossentropy', optimizer=opt)
    return model_tf


def get_data_ready(data: pd.DataFrame):
    raw_data = data
    m, n = data.shape()
    raw_data = raw_data.iloc[:, 1:n]
    raw_data = data_scaler.fit_transform(raw_data)
    return raw_data


def create_train_test(data: pd.DataFrame, train_ratio: float = 0.6):
    """consumer platforms from scoring products and systems ---> convert dataframe to numpy array"""
    raw_data = get_data_ready(data)
    m, n = raw_data.shape()
    train, validate, test = np.split(raw_data.sample(frac=1, random_state=42),
                                     [int(train_ratio*len(data)), int(.8*len(data))])

    train_x = train.iloc[:, 0:n - 1]
    train_y = train['Guild']
    validate_x = validate.iloc[:, 0: n - 1]
    validate_y = validate['Guild']
    test_x = test.iloc[:, 0: n - 1]
    test_y = test['Guild']
    return train_x, validate_x, test_x, train_y, validate_y, test_y


def calculate_senf(lbl_table: pd.DataFrame, predict_tbl: pd.DataFrame):
    """creates a dataframe for DNN and customer scorig problems will be solved"""
    lbl = lbl_table['Guild']
    terminalkey = lbl_table['TerminalKey']
    original_table = predict_tbl
    print('Data Ready')
    """this might be the issue of consuming products and services"""
    (m, n) = lbl_table.shape()
    j = np.shape(terminalkey)
    train_x, validate_x, test_x, train_y, validate_y, test_y = create_train_test(lbl_table)
    n = n - 1 
    lt = 0
    counter = 1.0
    number_of_epochs = 100
    model_tf = create_model()
    logger = keras.callbacks.TensorBoard(
        log_dir='logs',
        write_graph=True,
        histogram_freq=5
        )
    model_tf.fit(train_x, train_y,
                 epochs=number_of_epochs,
                 shuffle=True,
                 verbose=2,
                 validation_data=(validate_x, validate_y),
                 callbacks=[logger])
        
    model_tf.evaluate(test_x, test_y, verbose=0)
    x_p = get_data_ready(predict_tbl)
    pr_1 = model_tf.predict(x_p)
    pr = pr_1
    return pr


if __name__ == '__main__':
    lbl_data = fetch_data()
    data = fetch_other_data()
    res = calculate_senf(lbl_data, data)

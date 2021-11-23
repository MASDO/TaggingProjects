import pyodbc as pbo
from DataBase.MSSQLServerConnection import server_connection as sc
from DataBase.OracleConnection import OracleConnection as Oc
from keras import callbacks
import pandas as pd
import numpy as np 
import sklearn as skl
from sklearn import tree as tr
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import * 
import keras
from tensorboard import *
from keras.models import Sequential
from keras.layers import *
from sklearn.neighbors import NearestNeighbors as NN
from Modeling.adjustLabledData import right_labeled_data as rld
import sqlalchemy 
from multiprocessing import Process, Value, Array, Pool, Manager as mngr
from sqlalchemy import create_engine as c_e


def fetch_data(table_name: str = "TOTAL_PFM_PIVOT"):
    database_connection = Oc()
    database_connection.setQuery('select * from RECOMMENDERCALC.PFM_PIVOT_total{} sample(1)'.format(table_name))
    data = database_connection.fetch_data()
    data = pd.DataFrame(data)
    [_, n] = np.shape(data)
    data.rename(columns={0: 'TerminalKey'}, inplace=True)
    print('first column renamed as ... TerminalKey')
    data.rename(columns={n-1: 'Bought'}, inplace=True)
    print('last column renamed')
    data.fillna(0, inplace=True)
    print(data)
    print("Done Reading Data")
    return data

def insert_data(query =''):
    cxnn = pbo.connect(Trusted_connection='yes',
           driver = '{ODBC Driver 13 for Sql Server}',
           server = 'RD10-129\MASOUD_0',
           database = 'RecommenderDB'
           )
    cursor = cxnn.cursor()
    cursor.execute(query)
    cxnn.commit()
    cxnn.close()


##create Scoring issues
total = list()
data_scaler = MinMaxScaler()
##create model on NN
def create_model(number_of_dims = 399 , LR = 0.1):
    """here we create a neural network model based on extracted features of data"""
    ##must be developed for automatic classifying problems creates Automatic clssification System  
    model_TF = Sequential()
    model_TF.add(Dense(round(0.70 * number_of_dims), input_dim=number_of_dims, activation='relu', name='layer_0'))
    model_TF.add(Dense(round(0.90 * number_of_dims), activation='relu', name='layer_1'))
    model_TF.add(Dense(round(1.20 * number_of_dims), activation='sigmoid', name='layer_2'))
    model_TF.add(Dense(round(0.85 * number_of_dims), activation='sigmoid', name='layer_4'))
    model_TF.add(Dense(round(0.60 * number_of_dims), activation='relu', name='layer_6'))
    model_TF.add(Dense(round(0.30 * number_of_dims), activation='relu', name='layer_7'))
    model_TF.add(Dense(1, activation='softmax', name='result'))
    opt = keras.optimizers.Adam(learning_rate=LR)
    model_TF.compile(loss='binary_crossentropy', optimizer=opt)
    return model_TF


def create_binary_classifier(x , y , X , X_T , Y_T):
    """customer scoring problems"""
    print('Starting Calculations')
    acc_score = 0
    while acc_score <= 0.8:
       clf = tr.DecisionTreeClassifier()
       clf = clf.fit(x, y)
       predict = clf.predict_proba(X)
       test_predict = clf.predict_proba(X_T)
       test_predict = pd.Series(test_predict[:, 1])
       pd.to_numeric(test_predict)
       test_predict = test_predict * 100
       test_predict.round()
       test_predict = test_predict.astype(float)
       test_predict.fillna(0)
       Y_T = pd.Series(Y_T) * 100
       Y_T = Y_T.astype(float)
       Y_T.fillna(0)
       acc_score = accuracy_score(Y_T, test_predict.round())
       result = predict
    return result
## Made the Decision Tree Classifier


def fetch_colnames():
    fetch_column_names = ("select name from sys.columns where " + 
                          "object_id = (select object_id from sys.tables where name = 'taglog' and schema_id = 9)")
    master_data_columns = fetch_data(query = fetch_column_names)
    master_data_columns = np.asarray(master_data_columns)
    master_data_columns = master_data_columns[:, 0]
    master_data_columns = list(master_data_columns)
    number_of_data_columns = len(master_data_columns) - 1
    return master_data_columns

##def create_data(id:int):
def create_total_tbl(chunck_size = -1 , nth_chunck = -1):
    if chunck_size == -1:
       chunck_size = 500000
    if nth_chunck == -1:
       nth_chunck = 0
    master_data_columns = fetch_colnames()
    table_sample  =  100
    offset = chunck_size * nth_chunck 
    fetch  = chunck_size
    Nolabel_customers_query = ("select * from [pivot].[taglog] /**tablesample({})**/ where label is null order by [TERMINALKEY] " + 
                                "offset {} rows fetch next {} rows only".format(offset , fetch))
    ##CustomersQuery
    OriginalTable = fetch_data(query= Nolabel_customers_query)
    OriginalTable = np.asarray(OriginalTable)
    OriginalTable = pd.DataFrame(OriginalTable , columns = master_data_columns)
    OriginalTable.set_index('terminalkey')
    OriginalTable.fillna(0,inplace = True)
    OriginalTable['label'] = 0
    return OriginalTable

def create_tbl(id:int):
    master_data_columns = fetch_colnames()
    labeled_customers_query = ("select * from [Pivot].taglog /**tablesample(30)**/ where label = " + 
                               "(select GuildCode from [GuildM].[Guild_count] where id = {})".format(id))

    other_labeled_customers_query = ("select * from [Pivot].taglog /**tablesample(15)**/ where label is not null and label != " + 
                                     "(select GuildCode from [GuildM].[Guild_count] where id = {})".format(id))
    ##CustomersQuery----> 
    print(labeled_customers_query)
    labeledTable = fetch_data(query = labeled_customers_query)
    labeledTable = np.asarray(labeledTable)
    labeledTable = pd.DataFrame(labeledTable , columns = master_data_columns)
    labeledTable['label'] = 1
    labeledTable.set_index('terminalkey')
    labeledTable.fillna(0,inplace = True)
    
    ##finds other labeled terminals without flags
    other_labeledTable = fetch_data(query = other_labeled_customers_query)
    other_labeledTable = np.asarray(other_labeledTable)
    other_labeledTable = pd.DataFrame(other_labeledTable , columns = master_data_columns)
    other_labeledTable['label'] = 0
    other_labeledTable.set_index('terminalkey')
    other_labeledTable.fillna(0,inplace = True)
    ##
    frames = [other_labeledTable, labeledTable]
    total_tbl = pd.concat(frames)
    return (labeledTable, total_tbl, other_labeledTable)

def fetch_create_samples(data , n = 0):
    """ this func creates sample for classifier creation """
    if n == 0:
       [_,n] = np.shape(data)
    X_test = data.iloc[:, 0:n-1]##n-2
    Y_test = data.iloc[:, n]##n-1
    return (X_test , Y_test)

##ThisCreatesASampleForModeling
def get_sample(data:pd.DataFrame , ratio = 0 , sampleSize = -1):
    if sampleSize == -1:
       sampleSize = 5000
    else :
       sampleSize = sampleSize

    with_label = data[data.iloc[:, -1] != 0]
    X_with = len(with_label)
    without_label = data[data.iloc[:,-1] == 0]
    X_without = len(without_label)
    if ratio == 0:
       ratio = X_with//X_without
    if(X_with <= 150):
       smpl_with_label = with_label.sample(n=X_with - 30,replace=False)
       smpl_without_label  = without_label.sample(n=500,replace = False)
    else:
       smpl_with_label = with_label.sample(n = round(sampleSize * (ratio + 0.05)) + 100 ,replace = False)
       smpl_without_label  = without_label.sample(n = round(sampleSize * (0.95 - ratio)) , replace = False)
    total_smpl  = pd.concat([smpl_with_label , smpl_without_label])
    total_smpl.drop_duplicates()
    total_smpl  = total_smpl.fillna(0)
    return total_smpl

def adjust_sample(total_smpl:pd.DataFrame):
    total_smpl_adjuster = rld(total_smpl ,100)##
    total_smpl_idx = total_smpl_adjuster.multipleSamples()##
    total_smpl_with = total_smpl[total_smpl_idx[0]]
    total_smpl_withOut = total_smpl[total_smpl_idx[1]]
    frames = [total_smpl_with , total_smpl_withOut]
    total_smpl = pd.concat(frames)
    return total_smpl
    
def calculate_senf(senf_Id , AllTable , sample_size):
    number_of_data_columns = len(fetch_colnames())
    model_TF=create_model()
    i = senf_Id ## this is the Guild Identifiers
    c = create_tbl(id=i)
    OriginalTable=c[1]
    without_label=AllTable
    terminalkey=without_label['terminalkey']
    Mdl_data=OriginalTable.iloc[:, 1:number_of_data_columns]
    [m , n]=np.shape(Mdl_data)
    Mdl_data=data_scaler.fit_transform(Mdl_data)
    Mdl_data=pd.DataFrame(Mdl_data)
    [l_1, l_2] = np.shape(Mdl_data)
    
    print('GOT DATA')
    print('Data Ready')
    
    Shape = np.shape(OriginalTable)
    x1 = OriginalTable.terminalkey.count()
    j = np.shape(terminalkey)
    pr = np.zeros([j[0],1])
    n = n - 1 
    if sample_size == -1:
        sample_size = 10000
    else :
        sample_size = sample_size
    decay_rate = 1
    lt = 0
    counter = 1.0
    while lt <= 0.3  and counter <= 1.0 :
        number_of_epochs = 10
        total_smpl = get_sample(Mdl_data,sampleSize=sample_size)
        total_smpl = adjust_sample(total_smpl)
    
        total_smpl_2 = get_sample(Mdl_data , sampleSize=sample_size)
        total_smpl_2 = adjust_sample(total_smpl_2)
    
        ##total_smpl = data_scaler.fit_transform(total_smpl)
        X_test = total_smpl_2.iloc[:, 0:n-1]##n-2
        Y_test = total_smpl_2.iloc[:, n]##n-1
        x = total_smpl.iloc[: , 0:n-1]##n-2
        y = total_smpl.iloc[: , n]##n-1
    
        X = Mdl_data.iloc[: , 0:n]##n-1
        X = X.fillna(0)
        X_T = without_label.iloc[:,1: n]##n-1
        X_T = data_scaler.fit_transform(X_T)
        ##X_T = X_T.fillna(0)
        logger = keras.callbacks.TensorBoard(
        log_dir='logs',
        write_graph=True,
        histogram_freq=5
        )
        model_TF.fit(x,
                     y,
                     epochs=number_of_epochs ,
                     shuffle=True,
                     verbose=2,
                     validation_data=(X_test , Y_test),
                     callbacks=[logger])
        
        model_TF.evaluate(x,y, verbose = 0)
        pr_1 = model_TF.predict(X_T)
        pr = pr_1
        lt = max(pr)
        counter += 1
        print(i)
        print(lt)
    if lt <= 0.5:
        score = np.zeros([len(terminalkey) , 1])
        score = pd.DataFrame(score)
        lm = 0
        while lm < 100:
            total_smpl= get_sample(Mdl_data , sampleSize=sample_size)
            total_smpl = adjust_sample(total_smpl)
            total_smpl_2 = get_sample(Mdl_data , sampleSize=sample_size)
            total_smpl_2 = adjust_sample(total_smpl_2)
            X_test = total_smpl_2.iloc[:, 0:n-1]##n-2
            Y_test = total_smpl_2.iloc[:, n]##n-1
            x = total_smpl.iloc[: , 0:n-1]##n-2
            y = total_smpl.iloc[: , n]##n-1
            X = Mdl_data.iloc[: , 0:n]##n-1
            X = X.fillna(0)
            X_T = without_label.iloc[:, 1:n]
            lnt = create_binary_classifier(x , y, X_T, X_test, Y_test)
            lnt = np.reshape(lnt[:,1] ,[len(lnt),1])
            print("-------------> done iteration number (--{}--) for GuildID (<<<{}>>>)".format(lm,senf_Id))
            score = score.add(pd.DataFrame(lnt)) 
            lm += 1
            
    sample_size = round(sample_size * decay_rate)
    result = pd.DataFrame()
    ##result['terminalkey'] = terminalkey 
    ##result['lable'] = pr
    result['terminalkey'] = terminalkey
    
    if lt > 0.9 :
        result['label'] = np.round(pr * 100)
    else:
        result['label'] = score
    
    result['id'] = i
    print(result)
    result = result[result['label'] >= 50]
    path = 'C:\\Users\\M_manteghipoor.SB24\\Desktop\\Monthly_Saman\\Tagging\\'
    csv_name = 'result_{}.csv'.format(i)
    t = sc()
    ##result.to_csv(path,index = False)
    ## customer scorign problems will solve the issues o consumer products and goods
    t.bulk_insert(path ,'results','Taggs',csv_name, result, 'PFMData')
    i = i + 1
    print("-------------> done classifying the taggs for Guild Number{}".format(i))
    ##df = pd.DataFrame(X_T)
    ##print(df)
    ##X_N = data_scaler.inverse_transform(df)
    ##print(X_N)
    ## run tensorBoard py -3.6 -m tensorboard.main --logdir=C:\Users\M_manteghipoor.SB24\source\repos\TensorFlowCTree\TensorFlowCTree\logs
    ##master_data_columns = fetch_data(query = insert_into_temp_table)
    print('done')




if __name__ == '__main__':
  bulk_size = 1000000
  for mn in range(0,6):
      number_of_pools = 5
      start = 1
      end = number_of_pools
      AllTable = create_total_tbl(chunck_size=bulk_size,nth_chunck=mn)
      mgr = mngr()
      itr_num = int(round(174/number_of_pools)) + 1
      for t in range(0,itr_num):
           pool = Pool(number_of_pools)
           if t > 0:
              start = end 
              end = start + number_of_pools
           multiple_arrays = [pool.apply_async(calculate_senf, (i, AllTable , 10000))
                              for i in range(start, end)]
           for res in multiple_arrays:
               try :
                   res.get()
               except Exception as e :
                   print(e)
                   print('Reached the end of file !!!')
           pool.terminate()
               ##pool.join()
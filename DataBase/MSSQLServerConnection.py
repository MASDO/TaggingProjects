from sqlalchemy import create_engine as c_e
import pyodbc as pbo
import pandas as pd
import numpy as np
from DataBase.server_connection import server_connection


class MsSqlConnection(server_connection):
    def __init__(self, query: str = '', database: str = '', cols_query=''):
        self.database = database
        self.query = query
        self.driver = '{ODBC Driver 17 for Sql Server}'
        self.server = '10.8.18.23'
        self.cols_query = cols_query
        self.UID = 'Manteghipoor'
        self.PWD = 'ZXCvbn123'
        self.db_name = 'Recommender'
        ##self.conn_str = (
        ##    'Driver={};'.format(self.driver),
        ##    'Server={};'.format(self.server),
        ##    'Database={};'.format(self.database),
        ##    'Trusted_Connection=no;'
        ##)
        self.engine = c_e(
            "mssql+pyodbc://{}:{}@{}:1433/{}?driver=ODBC+Driver+17+for+SQL+Server".
            format(self.UID, self.PWD, self.server, 'Recommender'),
            fast_executemany=True)

    def connection(self):
        self.cxnn = pbo.connect(Trusted_connection='no',
                                driver=self.driver,
                                server=self.server,
                                database=self.database,
                                UID=self.UID,
                                PWD=self.PWD)
        return self.cxnn


    def create_engine(self):
        self.engine.connect()
        cxnn = self.engine.raw_connection()
        return cxnn

    def setDriver(self, driver: str):
        self.driver = driver

    def setServer(self, server: str):
        self.server = server

    def setQuery(self, query):
        self.query = query

    def setDataBase(self, dbName):
        self.database = dbName
    ## Setters Defined
    ## define insert methods

    def insert_data(self):
        query = self.query
        cxnn = self.connection()
        cursor = cxnn.cursor()
        cursor.execute(query)
        cxnn.commit()
        cxnn.close()
    ## insert Method_Defined
    ## read method Defined
    def fetch_data(self):
        if self.query == '':
            query = 'select * from Recommendation_TBL'
        else:
            query = self.query
        cxnn_f = self.connection()
        cursor = cxnn_f.cursor()
        cursor.execute(query)
        self.data = cursor.fetchall()
        cxnn_f.commit()
        cxnn_f.close()
        return self.data
    ## read method defined
    ## bulk insert

    def bulk_insert(self, path, schema, tbl, csvName, dataframe, dbname='attrition_Db'):
        self.setDataBase(dbname)
        path_t = path + csvName
        dataframe.to_csv(path_or_buf=path, index=False)
        str_directory = "'" + path + "'"
        insert_into_temp_table = ("bulk insert " +
                                  dbname + ".[" + schema + "].[" + tbl + "] from " +
                                  str_directory +
                                  " with (firstrow = 2 , fieldterminator = ',', rowterminator = '0x0a')")
        self.query = insert_into_temp_table
        self.insert_data()

    def create_tbl(self, tbl_name: str, schema: str, dbname: str, is_temp: bool):
        if tbl_name == '':
            tbl_name = 'customer_attr'
        if schema == '':
            schema = 'delinq'
        if dbname == '':
            dbname = 'attrition_Db'
        if is_temp == 0:
            create_tbl_query = (
                        'CREATE TABLE [' + dbname + '].[' + schema + '].[' + tbl_name + ']([custno] [varchar](50) not NULL,' +
                        '[month_count] [int] NULL,' +
                        '[avg_conditioned] [float] NULL,' +
                        '[report_month] [varchar](50) NULL,' +
                        '[report_type] [int] NULL) ON [PRIMARY]')
        else:
            create_tbl_query = ('CREATE TABLE ##' + tbl_name + '([custno] [varchar](50) not NULL,' +
                                '[month_count] [int] NULL,' +
                                '[avg_conditioned] [float] NULL,' +
                                '[report_month] [varchar](50) NULL,' +
                                '[report_type] [int] NULL) ON [PRIMARY]')
        try:
            cxnn_c = self.connection()
            cursor_c = cxnn_c.cursor()
            cursor_c.execute(create_tbl_query)
            cxnn_c.commit()
            cxnn_c.close()
        except Exception as e:
            print('Table with same name exists')
            create_tbl_query = 'truncate table [' + dbname + '].[' + schema + '].[' + tbl_name + ']'
            cxnn_c = self.connection()
            cursor_c = cxnn_c.cursor()
            cursor_c.execute(create_tbl_query)
            cxnn_c.commit()
            cxnn_c.close()
    ## bulinsert method

    def column_names(self, tbl_name):
        """find table column names"""
        if self.cols_query == '':
            find_colname_query = ("select name from sys.columns where " +
                                  "object_id = (select object_id from sys.tables where name = '" + str(
                        tbl_name) + "')")
        ##setQuery = self.cols_query()
        else:
            find_colname_query = self.cols_query()
        self.setQuery(find_colname_query)
        colnames = self.fetch_data()
        colnames = np.asarray(colnames)
        colnames = colnames[:, 0]
        colnames = list(colnames)
        return colnames

    def delete_from_tbl(self, tbl_name, colname='', filtered_parameter=''):
        delete_query = 'delete from ' + tbl_name + 'where ' + colname + '=' + filtered_parameter
        print(delete_query)
        self.query = delete_query
        self.insert_data()
    @staticmethod
    def create_pivot(DTFRM: pd.DataFrame, col_list: list, values):
        X1 = pd.pivot(DTFRM['custno'], column_names=col_list, values=values)
        return X1

    def bulk_insertSQLAlchemy(self, dataframe: pd.DataFrame):
        dataframe['is_deleted'] = 0
        dataframe['CreatedBy'] = 'REC_APP'
        dataframe['ModifiedBy'] = 'Null'
        dataframe['IP'] = '127.0.0.1'
        bulk_insert_cxnn = self.create_engine()
        cur = bulk_insert_cxnn.cursor()
        cur.executemany("insert into dbo.RECOMMENDATION_TBL "
                        "(CustNo,ProductID, PROBABILITY, SEGMENT, REQ_COUNT "
                        ",REC_REF , IsDeleted , CreatedOn , ModifiedOn , CreatedBy , ModifiedBy , IP)"
                        "values (?, ?, ?, "
                        "?, ?, ?, ?,"
                        "getdate(), null , ?,?,?)",
                        dataframe.values.tolist())
        bulk_insert_cxnn.commit()

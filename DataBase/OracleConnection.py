import cx_Oracle as Cx
from sqlalchemy import create_engine as c_e
import pandas as pd
import numpy as np
from DataBase.server_connection import server_connection


class OracleConnection(server_connection):
    def __init__(self):
        """Configure DataBase for Oracle Mostly for reading Data"""
        super(server_connection).__init__()
        user = 'RECOMMENDERCALC'
        pwd = 'recommendercalc'
        self.dsn = Cx.makedsn(
            '10.0.12.166', 1521, service_name='recdr.int.sb24.com')
        self.engine = c_e(f'oracle+cx_oracle://{user}:{pwd}@{self.dsn}',
                          echo=True, max_identifier_length=128)
        self.engine.connect()
        self.cxnn = self.engine.raw_connection()
        self.insert_query = (
            "insert into APP.RECOMMENDATION_TBL "
            "(CUSTNO, PRODUCT, PROBABILITY, SEGMENT, REQ_COUNT ,REC_REF)"
            "values (:Custno , :productCode, :Score , :Segment , :REQ_COUNT , :REQ_REF)"
        )

    def connection(self):
        self.engine = c_e("oracle+cx_oracle://RECOMMENDERCALC:recommendercalc@RECDR",
                          max_identifier_length=128)
        return self.cxnn

    ##Define setters
    def set_driver(self, driver: str):
        self.driver = driver

    def set_server(self, server: str):
        self.server = server

    def set_query(self, query):
        self.query = query

    def set_database(self, dbName):
        self.database = dbName

    def get_connection_engine(self):
        return self.engine

    # Setters Defined
    # define insert methods
    def manipulate_data(self, dml_query='', p_code=''):
        if dml_query == '':
            dml_query = "delete from APP.RECOMMENDATION_TBL where PRODUCT ={} ".format(p_code)
        cxnn = self.connection()
        cursor = cxnn.cursor()
        cursor.execute(dml_query)
        cxnn.commit()

    def set_insert_query(self, insert_query):
        self.insert_query = insert_query

    def bulk_insert(self, dataframe: pd.DataFrame):
        dsn = self.dsn
        bulk_insert_cxnn = Cx.connect('APP', 'app', dsn)
        cur = bulk_insert_cxnn.cursor()
        cur.executemany(self.insert_query,
                        dataframe.to_dict('records'))
        bulk_insert_cxnn.commit()

    # TODO : insert function must create a dynamic query
    def create_tbl(self, tbl_name: str, schema: str, dbname: str, is_temp: bool):
        if tbl_name == '':
            tbl_name = 'customer_attr'
        if schema == '':
            schema = 'delinq'
        if dbname == '':
            dbname = 'attrition_Db'
        if is_temp:
            create_tbl_query = (
                    'CREATE TABLE '
                    '[' + dbname + '].[' + schema + '].[' + tbl_name + ']([custno] [varchar](50) not NULL,' +
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
            print('Tbl exists')
            create_tbl_query = 'truncate table [' + dbname + '].[' + schema + '].[' + tbl_name + ']'
            cxnn_c = self.connection()
            cursor_c = cxnn_c.cursor()
            cursor_c.execute(create_tbl_query)
            cxnn_c.commit()
            cxnn_c.close()

    def column_names(self, tbl_name, schema: str = ''):
        """find table column names"""
        if self.cols_query == '':
            find_colname_query = ("select column_name from sys.all_tab_columns " +
                                  "where owner ='{}' and table_name ='{}' "
                                  "order by column_ID".format(schema.upper(), tbl_name.upper()))
            print(find_colname_query)
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

    def fetch_data(self):
        if self.query == '':
            query = "select * from sys.tables"
        else:
            query = self.query
        cxnn_f = self.connection()
        cursor = cxnn_f.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        cxnn_f.commit()
        cxnn_f.close()
        return data

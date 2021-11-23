class server_connection():
    def __init__(self,host_IP:str ='10.0.12.166',
                      port_num:int = 1521,
                      db_name:str='APP',
                      paswd:str = 'app'):
        self.host_IP = host_IP
        self.db_name = db_name
        self.port_num = port_num
        self.paswd = paswd
        return

    def sethost_IP(self,new_server_addr:str):
        self.server_addr = new_server_addr
        pass
    def setdb_name(self, new_db_name):
        self.db_name = new_db_name
        pass
    def set_port_num(self, new_port_num):
        self.port_num = new_port_num
        pass
    def set_paswd(self, newpaswd):
        self.paswd = newpaswd
        pass



import pandas as pd
import cx_Oracle
import os


class OracleDB():
    """
        【 Required Library 】import pandas as pd, import cx_Oracle, import os
        
        ※ (Reference)
        self.sql_path = 'D:/작업방/업무 - 자동차 ★★★/쿼리 MES 3.0/'
        self.dataset_path = 'D:/작업방/업무 - 자동차 ★★★/Dataset/'
    """
    def __init__(self):
        try:
            cx_Oracle.init_oracle_client(lib_dir=r"D:\OracleSQL\instantclient_21_10")
        except:
            pass
        
        LOCATION = r"C:\Oracle\BIN"
        os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"]
        
        self.userName = 'PC576954P'
        password_list = [112, 111, 115, 99, 111, 49, 49, 35]
        self.password = ''.join([chr(i) for i in password_list])

        self.server_dict = {'PKNMAS' : ('172.28.109.21', 2121), 'PKMNS': ('172.28.72.65', 1570)}
        
        self.sql_path = 'D:/작업방/업무 - 자동차 ★★★/쿼리 MES 3.0/'
        self.dataset_path = 'D:/작업방/업무 - 자동차 ★★★/Dataset/'
        
    def connect(self, DB='PKNMAS'):
        # dsn = cx_Oracle.makedsn("172.28.109.21", 2121, service_name="PKNMAS")
        # dsn = cx_Oracle.makedsn("172.28.72.65", 1570, service_name="PKMAS")
        
        dsn = cx_Oracle.makedsn(self.server_dict[DB][0], self.server_dict[DB][1], service_name=DB)
        self.conn = cx_Oracle.connect(self.userName, self.password, dsn)
        
    def execute(self, file_path=None, script=None, add_script='', verbose=1, variables=[]):
        if file_path is not None:
            file = open(file_path, 'r',encoding='UTF8')
            sql = file.read()
        elif script is not None: 
            sql = script
        
        sql_all = sql + '\n' +add_script
        
        with self.conn.cursor() as cs:
            cs.execute(sql_all, variables)     # 쿼리 실행
            Column_Names = [cn[0] for cn in cs.description]

            df_fetch = cs.fetchall()       # 한번에 뽑기
            
            result = pd.DataFrame(df_fetch , columns=Column_Names)
            if verbose > 0:
                print(f"* Complete Load Query: {result.shape}")
            
            if file_path is not None:
                file.close()
        self.result = result

    def disconnect(self):
        self.conn.close()



# DB = OracleDB()
# DB.connect()
# DB.disconnect()

# add_script = "AND TB_M41_M2PCM_PLC_STD1_RST.MTL_NO = 'CQG0287'"
# DB.execute(file_path=DB.sql_path + 'MicroData/(Active) Micro_2PCM_1ST.sql', add_script=add_script)
# DB.result




################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
"""
import numpy as np
import pandas as pd

import cx_Oracle
import cx_Oracle as oci

import os
import platform

import datetime

dataset_path = 'D:/작업방/업무 - 자동차 ★★★/Dataset'

'''
# 64bit Oracle Clinet 미설치시 사용
LOCATION = "D:/ora64/instantclient-basic-windows.x64-19.8.0.0.0dbru/instantclient_19_8"
os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"]
#os.environ["NLS_LANG"]="AMERICAN_AMERICA.AL32UTF8"
'''
# cx_Oracle.init_oracle_client(lib_dir="C:/Users/POSCOUSER/Downloads/instantclient_19_9")
cx_Oracle.init_oracle_client(lib_dir="D:/Downloads/instantclient_21_6")

LOCATION = "C:/Oracle/BIN"
sql_path = 'D:/작업방/업무 - 자동차 ★★★/쿼리 MES 3.0/'

# print("ARCH:", platform.architecture())
# print("FILES AT LOCATION:")
# for name in os.listdir(LOCATION):
#     print(name)
os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"]

# conn = oci.connect('172.28.109.21:2121')

userName = 'PC576954P'
# password_list = [112, 111, 115, 99, 111, 49, 35]
password_list = [112, 111, 115, 99, 111, 49, 49, 35]
password = ''.join([chr(i) for i in password_list])

print()
dsn = cx_Oracle.makedsn("172.28.109.21", 2121, service_name="PKNMAS")
# dsn = cx_Oracle.makedsn("172.28.72.65", 1570, service_name="PKMAS")


#### sql File
conn = cx_Oracle.connect(userName, password, dsn)
# f = open(sql_path  + 'Sample_Query.sql', 'r',encoding='UTF8')
# f = open(sql_path  + '(active) AutoMobile_Slab,열연,CAL,CGL,정정LIne_v2.05.sql', 'r',encoding='UTF8')
# f = open(sql_path  + '(Active) 열연 조업결과 확인.sql', 'r',encoding='UTF8')
# f = open(sql_path  + 'Test_Query/Test_Query008.sql', 'r',encoding='UTF8')
# f = open(sql_path  + '품질설계정보.sql', 'r',encoding='UTF8')
# f = open(sql_path  + '(Active) MTL_진행관리(PGBISDS).sql', 'r',encoding='UTF8')
# f = open(sql_path  + '(active) AutoMobile_Slab,열연,CAL,CGL,정정LIne_v1.06.sql', 'r',encoding='UTF8')
# f = open(sql_path  + '(Active) MTL_Tracking.sql', 'r',encoding='UTF8')
# f = open(sql_path  + '모니터링쿼리/980DP 품질관제모니터링(냉연,도금).sql', 'r',encoding='UTF8')
# f = open(sql_path  + 'MicroData/(Active) Micro_CAL_2CAL.sql', 'r',encoding='UTF8')
# f = open(sql_path  + 'MicroData/(Active) Micro_CAL_4-2CAL.sql', 'r',encoding='UTF8')
# f = open(sql_path  + 'MicroData/(Active) Micro_CGL_5CGL.sql', 'r',encoding='UTF8')
# f = open(sql_path  + 'MicroData/(Active) Micro_CGL_7CGL.sql', 'r',encoding='UTF8')
# f = open(sql_path  + 'MicroData/(Active) Micro_CT.sql', 'r',encoding='UTF8')
f = open(sql_path  + 'MicroData/(Active) Micro_2PCM_1ST.sql', 'r',encoding='UTF8')


# exec(execute_script)
# print_DataFrame(df)
# df.to_clipboard(index=False)

sql_query_all = f.read()
sql_query_all
sql_query_all = sql_query_all + "\n AND TB_M41_M2PCM_PLC_STD1_RST.MTL_NO = 'CQF9162'"


# conditions = '''\n
# AND TB_M2KC01_E_MTRL_010.MTL_NO IN ('HRJ151850', 'HRD112520', 'HRD108500', 'HRD107780', 'HRD107770',
#        'HRD107750', 'HRD107740', 'HRD107720', 'HRD107710')          -- (IN) 열연코일 번호
# --AND TB_M2KC01_E_MTRL_010.MTL_NO LIKE ('%')          -- (LIKE) 열연코일 번호

# -- AND TB_M0AA11_S_DOQ0100.ORDER_HEAD_LINE_NO IN ('')                   -- Order번호
# -- AND SUBSTR(TB_M0AA11_S_DOQ0100.ORDER_HEAD_LINE_NO, 1, 10) IN ('')    -- OrderHead번호
# '''

with conn.cursor() as cs:
# cs = conn.cursor()
    # cs.execute(sql_query_all + conditions)     # 쿼리 실행
    cs.execute(sql_query_all)     # 쿼리 실행
    Column_Names = [cn[0] for cn in cs.description]

    df_fetch = cs.fetchall()       # 한번에 뽑기
    df_sql = pd.DataFrame(df_fetch , columns=Column_Names)
    # print(df_sql.shape)
    f.close()

# df_sql.to_clipboard()
# df_sql.to_clipboard(index=False)
np.array(df_fetch).shape
df_sql_frame = pd.DataFrame(df_fetch , columns=Column_Names)
df_sql_frame.to_clipboard()
conn.close()

# date_string = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
# df_sql.to_csv(f"{dataset_path}/{date_string}_sql.csv", index=False, encoding='utf-8-sig')



















# with 구문 ------------------------------------------------------------------------------------------
# with cx_Oracle.connect(userName, password, dsn) as connection:
#     cursor = connection.cursor()
conn = cx_Oracle.connect(userName, password, dsn)

# with conn.cursor() as cs:
# cs = conn.cursor()


# 한줄씩 읽기 --------------------
cs = conn.cursor()
cs.execute(abc)     # 쿼리 실행

Column_Names = [cn[0] for cn in cs.description]

# for line in cs:
#     print(line)
k = cs.fetchall()        # 한번에 뽑기
pd.DataFrame(k, columns=Column_Names)

cs.close()
conn.close()



# file로 읽기 --------------------
f = open(sql_path  + 'test.sql')
sql_query_all = f.read()
sql_querys = sql_query_all.split(';')

cs = conn.cursor()
cs.execute(sql_query_all)     # 쿼리 실행
Column_Names = [cn[0] for cn in cs.description]

k = cs.fetchall()        # 한번에 뽑기
pd.DataFrame(k, columns=Column_Names)

# for sql_query in sql_querys:
#     cs.execute(sql_query)
cs.close()






abc = '''
SELECT
'a' as "A"
,1 as "B"
FROM DUAL

UNION ALL

SELECT
'b' as "A"
,2 as "B"
FROM DUAL

UNION ALL

SELECT
'cs' as "A"
,3 as "B"
FROM DUAL
'''

execute_script = '''
sql_query_all = f.read()
with conn.cursor() as cs:
    # cs = conn.cursor()
    # cs.execute(sql_query_all + conditions)     # 쿼리 실행
    cs.execute(sql_query_all)     # 쿼리 실행
    Column_Names = [cn[0] for cn in cs.description]

    df_fetch = cs.fetchall()       # 한번에 뽑기
    df = pd.DataFrame(df_fetch , columns=Column_Names)
    print(df.shape)
    # df[['강종_소구분', 'ORDER_NO', '열연_압연일', '주문두께', '주문폭','열연코일번호', 'CT', 'CT후냉각', '설계_CT후냉각']]
    # df.to_clipboard(index=False)
    f.close()
conn.close()
'''

from pandasql import sqldf




"""
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 06:25:52 2019

@author: NB313128
"""
import os
os.chdir('C:\\Users\\nb313128\\repos\\voucher_models\\taxify')
import pyodbc
import pandas as pd
from math import ceil
#%% 
df = pd.read_excel('data/Taxify results.xlsx', sheet_name='Taxify users')
#%%
cnxn = pyodbc.connect(driver='{SQL Server}',
                      server='BISAN\BISAN', database='CustomerProd2018',trusted_connection='yes')

SQL = '''SET NOCOUNT ON;
            CREATE TABLE ##dedupe_temp (DedupeStatic BIGINT);
            SELECT 1'''
pd.read_sql_query(SQL, cnxn)

for j in range(ceil(len(df)/1000)):
    id_str = '),('.join([str(i) for i in 
                                 list(df.index)[j*1000:(j+1)*1000]])
    SQL = '''SET NOCOUNT ON;
            INSERT INTO ##dedupe_temp
            VALUES ({0});
            SELECT 1'''.format(id_str)
    pd.read_sql_query(SQL, cnxn)
    print(j+1,'of',ceil(len(df)/1000))
    
for mth in range(1,11):
    yr_mth = 201800 + mth
    print('busy extracting',yr_mth)
    SQL = '''SELECT 
            	Dedupegroup AS DedupeStatic
            	,Account_No AS AccountNumber
            	,Tran_No AS TransactionNumber
            	,Channel
            	,TRAN_DATE AS TransactionDate
            	,Tran_Amount AS TransactionAmount
            	,[TEXT] AS TransactionDescription
            	,Event
            	,[Type]
            	,'ChequeAccount' AS AccountType
            FROM CAProd{0} ca
            INNER JOIN ##dedupe_temp dt on dt.DedupeStatic = ca.Dedupegroup
            WHERE ([TEXT] LIKE ('%TAXIFY%') OR [TEXT] LIKE ('%UBER%') OR [TEXT] LIKE ('%GAUTRAIN%')) 
                AND channel='POS' AND Event='Card Purchase' AND [TEXT] NOT LIKE ('%UBER EATS%') AND
                [TEXT] NOT LIKE ('%BLOUBERG%')
            UNION
            SELECT
            	Dedupegroup AS DedupeStatic
            	,Account_No AS AccountNumber
            	,Tran_No AS TransactionNumber
            	,Channel
            	,TRAN_DATE AS TransactionDate
            	,Tran_Amount AS TransactionAmount
            	,[TEXT] AS TransactionDescription
            	,Event
            	,[Type]
            	,'SavingsAccount' AS AccountType
            FROM SAProd{0} sa
            INNER JOIN ##dedupe_temp dt on dt.DedupeStatic = sa.Dedupegroup
            WHERE ([TEXT] LIKE ('%TAXIFY%') OR [TEXT] LIKE ('%UBER%') OR [TEXT] LIKE ('%GAUTRAIN%')) 
                AND channel='POS' AND Event='Card Purchase' AND [TEXT] NOT LIKE ('%UBER EATS%') AND
                [TEXT] NOT LIKE ('%BLOUBERG%')
            UNION
            SELECT
            	Dedupegroup AS DedupeStatic
            	,Account_No AS AccountNumber
            	,-1 AS TransactionNumber
            	,Channel
            	,TRAN_DATE AS TransactionDate
            	,Tran_Amount AS TransactionAmount
            	,[TEXT] AS TransactionDescription
            	,Event
            	,[Type]
            	,'CreditCard' AS AccountType
            FROM CMProd{0} cm
            INNER JOIN ##dedupe_temp dt on dt.DedupeStatic = cm.Dedupegroup
            WHERE ([TEXT] LIKE ('%TAXIFY%') OR [TEXT] LIKE ('%UBER%') OR [TEXT] LIKE ('%GAUTRAIN%')) 
                AND channel='POS' AND Event='Card Purchase' AND [TEXT] NOT LIKE ('%UBER EATS%') AND
                [TEXT] NOT LIKE ('%BLOUBERG%')
            '''.format(yr_mth)

    if mth == 1:
        df_trans = pd.read_sql_query(SQL, cnxn)
    else:
        df_trans = pd.concat([df_trans,pd.read_sql_query(SQL, cnxn)])
        
SQL = '''
            SELECT	Dedupe_Static AS DedupeStatic
                    ,Cstmr_Birth_Dte AS BirthDate
                    ,Race 
                    ,[Language]
                    ,Gender
                    ,Marital_Status AS MaritalStatus
                    ,GeoProvince AS Province
                    ,NII
                    ,NIR
                    ,MainBank

            FROM ##dedupe_temp dt
            LEFT JOIN IIIDB.dbo.EEE_Data_201810 cl ON cl.Dedupe_Static = dt.DedupeStatic;'''
            
df_client = pd.read_sql_query(SQL, cnxn)

cnxn.close()
#%%
df.to_hdf('data/raw_data.h5', key='taxify_users_results')
df_trans.to_hdf('data/raw_data.h5', key='taxify_users_transactions')
df_client.to_hdf('data/raw_data.h5', key='taxify_users_client_info')
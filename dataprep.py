# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 06:25:52 2019

@author: NB313128
"""
import os
os.chdir('C:\\Users\\nb313128\\repos\\voucher_models\\taxify')
import pandas as pd
#%%
df = pd.read_hdf('data/raw_data.h5', key='taxify_users_results')
df_trans = pd.read_hdf('data/raw_data.h5', key='taxify_users_transactions')
df_client = pd.read_hdf('data/raw_data.h5', key='taxify_users_client_info')
df= df.set_index('CIS')
df_client= df_client.set_index('DedupeStatic')
#%%
# target / response
df_train = pd.DataFrame((df['Redeemed (Y/N)'] == 'Y').astype(int))
df_train.rename(columns = {'Redeemed (Y/N)':'target'},inplace=True)
# age calc
df_train['client_age'] = (2018 + 10/12) -\
                (pd.to_datetime(df_client.BirthDate).dt.year + pd.to_datetime(df_client.BirthDate).dt.month/12)
# other numerical client info
df_train = df_train.merge(df_client[['NII','NIR','MainBank']], left_index=True, right_index=True, how='left')

#%% label encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

# RACE
df_client.loc[df_client.Race.isin(['0','']),'Race'] = 'NA'
df_client.Race.fillna('NA',inplace=True)
le_race = LabelEncoder().fit(df_client.Race)
joblib.dump(le_race,'data/race_encoder.pkl')
df_train['race_enc'] = le_race.transform(df_client.Race)
# quick ref ['A', 'B', 'C', 'NA', 'W'] --> [0, 1, 2, 3, 4]

# LANGUAGE
# get this with: SELECT	distinct [Language] FROM IIIDB.dbo.EEE_Data_201810 
list_of_languages = ['NA', 'AFR', 'ENG', 'NDB', 'PED', 'SOT', 'SWA', 'TSO', 
                     'TSW', 'VEN', 'XHO', 'ZUL']
le_language = LabelEncoder().fit(list_of_languages)
joblib.dump(le_language,'data/language_encoder.pkl')
df_client.loc[df_client.Language.isin(['0','']),'Language'] = 'NA'
df_client.Language.fillna('NA',inplace=True)
df_train['language_enc'] = le_language.transform(df_client.Language)
# quick ref ['AFR', 'ENG', 'NA', 'NDB', 'PED', 'SOT', 'SWA', 'TSO', 'TSW',
#       'VEN', 'XHO', 'ZUL'] --> [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]

# GENDER
df_client.loc[df_client.Gender.isin(['0','']),'Gender'] = 'NA'
df_client.Gender.fillna('NA',inplace=True)
le_gender = LabelEncoder().fit(df_client.Gender)
joblib.dump(le_gender,'data/gender_encoder.pkl')
df_train['gender_enc'] = le_gender.transform(df_client.Gender)
# ['F', 'M', 'NA'] --> [0, 1, 2]

# MARITAL STATUS
list_marital = ['NA','D','K','M','S','U','W']
df_client.loc[df_client.MaritalStatus.isin(['0','']),'MaritalStatus'] = 'NA'
df_client.MaritalStatus.fillna('NA',inplace=True)
le_marital = LabelEncoder().fit(list_marital)
joblib.dump(le_marital,'data/marital_encoder.pkl')
df_train['marital_enc'] = le_marital.transform(df_client.MaritalStatus)
# ['D', 'K', 'M', 'NA', 'S', 'U', 'W'] --> [0, 1, 2, 3, 4, 5, 6]


# PROVINCE
list_province = ['NA','Eastern Cape', 'Free State', 'Gauteng', 'KwaZulu-Natal',
                 'Limpopo', 'Mpumalanga', 'North West', 'Northern Cape', 'Western Cape']
le_province = LabelEncoder().fit(list_province)
joblib.dump(le_province,'data/province_encoder.pkl')
df_client.loc[df_client.Province.isin(['0','']),'Province'] = 'NA'
df_client.Province.fillna('NA',inplace=True)
df_train['province_enc'] = le_province.transform(df_client.Province)
# ['Eastern Cape', 'Free State', 'Gauteng', 'KwaZulu-Natal',
#       'Limpopo', 'Mpumalanga', 'NA', 'North West', 'Northern Cape',
#       'Western Cape'] --> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#%% add monthly spends 
#add company classification
df_trans.loc[df_trans.TransactionDescription.str.contains('taxify',case=False),'company'] = 'taxify'
df_trans.loc[df_trans.TransactionDescription.str.contains('gautrain',case=False),'company'] = 'gautrain'
df_trans.loc[df_trans.TransactionDescription.str.contains('uber',case=False),'company'] = 'uber'
df_trans['period'] = pd.to_datetime(df_trans.TransactionDate).dt.year*100 + pd.to_datetime(df_trans.TransactionDate).dt.month
df_trans['day'] = pd.to_datetime(df_trans.TransactionDate).dt.year*10000 + pd.to_datetime(df_trans.TransactionDate).dt.month*100 +\
                    pd.to_datetime(df_trans.TransactionDate).dt.day
# create aggregates
for company in ['taxify', 'uber', 'gautrain']:
    df_company_amt = pd.DataFrame(df_trans.loc[df_trans.company==company].groupby(['DedupeStatic','period']).TransactionAmount.sum()).unstack(1)
    df_company_amt.columns = df_company_amt.columns.droplevel()
    df_company_cnt = pd.DataFrame(df_trans.loc[df_trans.company==company].groupby(['DedupeStatic','period']).TransactionAmount.count()).unstack(1)
    df_company_cnt.columns = df_company_cnt.columns.droplevel()
    
    df_train = df_train.merge(df_company_amt[[201805, 201806, 201807, 201808, 201809, 201810]].fillna(0),
                              left_index=True,right_index=True, how='left')
    df_train.rename(columns = {201805: company+'_amt_201805', 201806: company+'_amt_201806', 
                               201807: company+'_amt_201807', 201808: company+'_amt_201808', 
                               201809: company+'_amt_201809', 201810: company+'_amt_201810'},inplace=True)
    
    df_train = df_train.merge(df_company_cnt[[201805, 201806, 201807, 201808, 201809, 201810]].fillna(0),
                              left_index=True,right_index=True,how='left')
    df_train.rename(columns = {201805:company+'_cnt_201805', 201806:company+'_cnt_201806', 
                               201807:company+'_cnt_201807', 201808:company+'_cnt_201808', 
                               201809:company+'_cnt_201809', 201810:company+'_cnt_201810'},inplace=True)

#%%
df_train.to_hdf('data/training_data.h5', key='taxify_users_train_data')

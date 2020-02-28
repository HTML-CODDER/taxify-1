# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 04:06:43 2019

@author: NB313128
"""

import os
os.chdir('/home/lnr-ai/krisjan/taxify/')
import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
#%%
df_trans_full = pd.read_csv('data/taxify_users_full_transactions.csv')
df_trans = df_trans_full.loc[df_trans_full.day >= 20180801]
df_target = pd.read_hdf('data/training_data.h5', key='taxify_users_train_data')['target']

#%% SET UP TRANSACTION IMAGE GRID
#generate lists of all days and companies for joining later
unique_days = np.sort(df_trans.day.unique())
unique_clients = df_target.index.unique()
unique_companies = np.sort(df_trans.company.unique())
all_days = pd.DataFrame(list(itertools.product(unique_clients, unique_days)),columns=['DedupeStatic','day'])

# frames for each client
df_trans_img = pd.DataFrame(df_trans.groupby(['DedupeStatic','day','company']).TransactionAmount.sum())
df_trans_img = df_trans_img.unstack(level = 2)
df_trans_img.columns = df_trans_img.columns.droplevel(0)

# fill in missing months by joining to all_days
df_trans_img = all_days.merge(df_trans_img, left_on=['DedupeStatic','day'],right_index=True, how='left')
#sort
df_trans_img.sort_values(['DedupeStatic','day'], inplace = True)
# handle missing
df_trans_img.fillna(0, inplace=True)
# set index
df_trans_img = df_trans_img.set_index(['DedupeStatic','day'])
# standard scaling
scaler = StandardScaler()
df_trans_img = pd.DataFrame(scaler.fit_transform(df_trans_img.values), index = df_trans_img.index,
                            columns = df_trans_img.columns)
#%%
df_trans_img = pd.read_hdf('data/trans_img_for_dl_20190306.h5', key='df_img')
#%% CONSTRUCT ARRAY FOR CNN
# empty array of correct shape
X = np.full((len(unique_clients), len(unique_days), len(df_trans_img.columns)),np.nan)
# fill it using numpy advanced indexing
X[df_trans_img.stack().index.labels] = df_trans_img.stack().values.flat

y = df_target.sort_index().values

#%% TRAINING
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#%% get shape right for dl
def dl_shape(X):
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
#%%
class roc_callback(tf.keras.callbacks.Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        self.auc = []
        self.auc_val = []
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        self.auc.append(roc)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        self.auc_val.append(roc_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
#%%
def conv_model():
     vec_dim = len(df_trans_img.columns)

     model = tf.keras.Sequential()
     model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2) ,padding='same', activation='relu', input_shape=(X.shape[1],X.shape[2],1)))
     model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4) ,padding='same', activation='relu', input_shape=(X.shape[1],X.shape[2],1)))
     model.add(tf.keras.layers.MaxPool2D())
     model.add(tf.keras.layers.BatchNormalization())
     model.add(tf.keras.layers.Flatten())
#    model.add(tf.keras.layers.Dense(vec_dim*7, activation = 'relu',input_dim=vec_dim))
#    model.add(tf.keras.layers.Dropout(.3))
#     model.add(tf.keras.layers.Dense(vec_dim*4, activation = 'relu',input_dim=vec_dim))
#     model.add(tf.keras.layers.Dropout(.5))
     model.add(tf.keras.layers.Dense(vec_dim*2, activation = 'relu',input_dim=vec_dim))
     model.add(tf.keras.layers.Dropout(.25))
     model.add(tf.keras.layers.Dense(vec_dim, activation = 'relu',input_dim=vec_dim)) #, input_dim=vec_dim
     model.add(tf.keras.layers.Dropout(.1))
     model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
     model.compile(optimizer = tf.train.AdamOptimizer(0.01), loss = 'binary_crossentropy')
     return model
#%%
def conv_model_1():
    vec_dim = len(df_trans_img.columns)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(2,2) ,padding='same', activation='relu', input_shape=(X.shape[1],X.shape[2],1)))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4) ,padding='same', activation='relu', input_shape=(X.shape[1],X.shape[2],1))) 
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(vec_dim*12, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(.15))
    model.add(tf.keras.layers.Dense(vec_dim*2, activation = 'relu'))
#    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(vec_dim*1, activation = 'relu')) #, input_dim=vec_dim
#    model.add(tf.keras.layers.Dropout(.05))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = tf.train.AdamOptimizer(0.001), loss = 'binary_crossentropy')    
    return model 
#%%
print('---------------------------------------------------------')
for i in range(5):
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    #
    roc_history = roc_callback(training_data=(dl_shape(X_train),y_train),
                               validation_data=(dl_shape(X_test),y_test))
    #early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc_roc', patience=4, verbose=1, mode='max')
    model = conv_model_1()
    model.fit(dl_shape(X_train), y_train, batch_size=64, epochs = 5, validation_data=(dl_shape(X_test),y_test),
                  callbacks=[roc_history]) #, early_stop
    #
    cumsum_vec = np.cumsum(np.insert(roc_history.auc_val, 0, 0))
    ma_vec = (cumsum_vec[5:] - cumsum_vec[:-5]) / 5
    # summarize history for accuracy
    plt.figure(figsize=(10, 8))
    plt.plot(roc_history.auc)
    plt.plot(roc_history.auc_val)
    plt.plot(ma_vec)
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()
#%% ENCODE
from sklearn.model_selection import KFold
epoch_nr = 2
kf = KFold(n_splits = 5, shuffle=True)
unique_clients = np.sort(unique_clients)
#%%
i=0
for train_idx, test_idx in kf.split(unique_clients):
    model = conv_model_1()
    model.fit(dl_shape(X[train_idx]), y[train_idx], batch_size=96, epochs = epoch_nr)
    df_temp = pd.DataFrame(model.predict(dl_shape(X[test_idx])), index=unique_clients[test_idx])
    if i == 0:
        df_out = df_temp.copy()
    else:
        df_out = pd.concat([df_out, df_temp])
    i = i + 1
#%%
df_out.rename(columns = {0:'transactions_dl_encode'},inplace=True)
df_out.to_hdf('data/training_data.h5', key='taxify_transactions_dl_encode')
#%%
import matplotlib.pyplot as plt
import shap
#%%

e = shap.DeepExplainer(model, dl_shape(X))
shap_values = e.shap_values(dl_shape(X[27:28]))
shap.image_plot(shap_values,-dl_shape(X[27:28]),fig_size=(21,14)) #,figsize=(21,14)

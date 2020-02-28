#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 09:49:40 2018

@author: krisjan
"""

#%%
import os
os.chdir('/home/lnr-ai/krisjan/taxify/')
#%%
import pandas as pd
import xgboost as xgb
import numpy as np
from tqdm import tqdm
#%%
train_df = pd.read_hdf('data/training_data.h5', key='taxify_users_train_data')
df_out = pd.read_hdf('data/training_data.h5', key='taxify_transactions_dl_encode')
train_df = train_df.merge(df_out, left_index=True, right_index=True)
#%% get train_df from dataprep for final train
col_filter = list(train_df.drop(['target'],axis=1).columns)
col_filter = ['client_age', 'NII', 'NIR', 'MainBank', 'race_enc', 'language_enc',
              'gender_enc', 'marital_enc', 'province_enc', 'transactions_dl_encode']
X_train = train_df[col_filter]
y_train = train_df.target
#%%
myseed = 1
cv_folds = 5
ttu = 6
#random.seed(myseed)
max_trees = 4000
#%% objective function
date = str(pd.datetime.now().year) + str(pd.datetime.now().month)+str(pd.datetime.now().day)

def xgb_x_val_auc(param_list):

    md = int(param_list[0])
    mcw = int(param_list[1])
    gam = param_list[2]
    ss = param_list[3]
    csbt = param_list[4]
    spw = param_list[5]
    lr = param_list[6]
    ra = param_list[7]
    rl = param_list[8]
    
    xgb_param = {'base_score': 0.5,
                         'booster': 'gbtree',
                         'colsample_bylevel': 1,
                         'colsample_bytree': csbt,
                         'gamma': gam,
                         'learning_rate': lr,
                         'max_delta_step': 0,
                         'max_depth': md,
                         'min_child_weight': mcw,
                         'missing': None,
    #                     'n_estimators': max_trees,
                         'n_jobs': ttu,
                         'objective': 'binary:logistic',
                         'reg_alpha': ra,
                         'reg_lambda': rl,
                         'scale_pos_weight': spw,
                         'random_state': myseed,
                         'subsample': ss,
                         'silent':1}

    xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    cvresult = xgb.cv(xgb_param, xgtrain, nfold=cv_folds, num_boost_round= max_trees,
            metrics='auc', early_stopping_rounds=50, seed=myseed)
    cur_res = cvresult.tail(1).values[0][2]
    with open('train_log/'+date+'.txt', 'a') as f:
        print('-----------', file=f)
        print(list(param_list), file=f)
        print('ntrees: ',cvresult.shape[0], file=f)
        print('auc:',cur_res, file=f)
    return cur_res

#%%
import genetic_algorithm as gen
#%% seed starting individual
md, mcw, gam, ss, csbt, spw, lr, ra, rl = [5, 1, 1, .8, .8, 1, .1, 0, 1]
pop_seed = pd.DataFrame([[md, mcw, gam, ss, csbt, spw, lr, ra, rl],
                         [4, 2, 0.8971370753635228, 0.30915579792707026, 
                          0.8329803017845185, 6.669532207844701, 
                          0.05149689274641363, 37.43241165195754, 6.291063362234251],
                          [18, 1, 1.8688743787970807, 0.6318719906421438, 
                           0.6170957251991486, 4.023007940299918, 0.3479060026801067, 
                           31.32787336478784, 2.2597710460534177]])
#%% test run
xgb_x_val_auc([md, mcw, gam, ss, csbt, spw, lr, ra, rl])

    #%%
list_of_types = ['int', 'int', 'float', 'float','float', 'float', 'float', 'float', 'float']

lower_bounds = [1,1,0,.4,.4,1,.001,0,1]
upper_bounds = [20,20,10,1,1,10,.5,100,10]
pop_size = 1000
generations = 15

#%% Genetic Search for Parameters

best_gen_params, best_gen_scores = gen.evolve(list_of_types, lower_bounds, upper_bounds, 
                                          pop_size, pop_seed, generations, xgb_x_val_auc,
                                          mutation_prob=.05, mutation_str=.2, 
                                          perc_strangers=.05, perc_elites=.1)
#                                          ,old_scores=best_gen_scores, save_gens=False)

#%% overall
# Best individual is [4.5600000000000005, 2.5360000000000005, 0.8971370753635228, 0.30915579792707026, 
# 0.8329803017845185, 6.669532207844701, 0.05149689274641363, 37.43241165195754, 6.291063362234251]
# best score: 0.7282206
# ntrees: 216
#---------------
#best individual [18.0, 1.0, 1.8688743787970807, 0.6318719906421438, 0.6170957251991486, 
#                 4.023007940299918, 0.3479060026801067, 31.32787336478784, 2.2597710460534177]
#best score: 0.727912 
# ntrees:  56
#------------ switched to map for optimization
#best individual [1.0, 8.3, 3.7118060251648277, 0.633218485865329, 0.7620830210050096, 
#3.605327445070424, 0.21669983048276087, 26.831468464108443, 0.7179914062225624]
#best score: 0.1034704
md, mcw, gam, ss, csbt, spw, lr, ra, rl = [18, 1, 1.8688743787970807, 0.6318719906421438, 
                                           0.6170957251991486, 4.023007940299918, 0.3479060026801067, 
                                           31.32787336478784, 2.2597710460534177]
ntrees = 16

#%% BOOTSTRAP THRESHOLD OPTIMIZATION
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

#%% no iterations
iters = 1000

#%%
auc_list = []
for i in tqdm(range(iters)):

    X_train, X_test, y_train, y_test = train_test_split(train_df[col_filter], train_df.target,test_size=0.2)
                         
    xgb_param = {'base_score': 0.5,
                         'booster': 'gbtree',
                         'colsample_bylevel': 1,
                         'colsample_bytree': csbt,
                         'gamma': gam,
                         'learning_rate': lr,
                         'max_delta_step': 0,
                         'max_depth': md,
                         'min_child_weight': mcw,
                         'missing': None,
    #                     'n_estimators': max_trees,
                         'n_jobs': ttu,
                         'objective': 'binary:logistic',
                         'reg_alpha': ra,
                         'reg_lambda': rl,
                         'scale_pos_weight': spw,
                         'random_state': myseed,
                         'subsample': ss,
                         'silent':1}

    xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)                         
    estimator = xgb.train(xgb_param, xgtrain, num_boost_round=ntrees)

    predicted = estimator.predict(xgb.DMatrix(X_test.values))
    
    auc_list.append(roc_auc_score(y_test,predicted))
    
    thres_acc_list = []
    df_confusion = pd.DataFrame(index=range(101), columns=['TN','FP','FN','TP', 'burn_rate', 'send_reduction'])
    for threshold_100 in range(0,101):
        threshold = threshold_100/100
        predicted_class = (predicted > threshold).astype(int)
        df_confusion.loc[threshold_100,'FP'] = ((predicted_class == 1) & (y_test == 0)).sum()
        df_confusion.loc[threshold_100,'TP'] = ((predicted_class == 1) & (y_test == 1)).sum()
        df_confusion.loc[threshold_100,'TN'] = ((predicted_class == 0) & (y_test == 0)).sum()
        df_confusion.loc[threshold_100,'FN'] = ((predicted_class == 0) & (y_test == 1)).sum()
        df_confusion.loc[threshold_100,'send_fraction'] = sum(predicted_class==1)/len(predicted_class)
        if (df_confusion.loc[threshold_100,'FP'] + df_confusion.loc[threshold_100,'TP'])!=0:
            df_confusion.loc[threshold_100,'burn_rate'] = df_confusion.loc[threshold_100,'TP'] / \
                                            (df_confusion.loc[threshold_100,'FP'] + df_confusion.loc[threshold_100,'TP'])
        else:
            df_confusion.loc[threshold_100,'burn_rate'] = 0 
        thres_acc_list.append(accuracy_score(y_test, predicted_class))
        
    if i == 0:
        df_thres = np.array(thres_acc_list)
        df_send_fraction = pd.DataFrame(df_confusion.send_fraction)
        df_burn_rate = pd.DataFrame(df_confusion.burn_rate)
        
    else:
        df_thres = np.vstack((df_thres, thres_acc_list))
        df_send_fraction = df_send_fraction.merge(pd.DataFrame(df_confusion.send_fraction),left_index=True,
                                                    right_index=True, suffixes = ['','_'+str(i)])
        df_burn_rate = df_burn_rate.merge(pd.DataFrame(df_confusion.burn_rate),left_index=True,
                                                    right_index=True, suffixes = ['','_'+str(i)])
        
df_thres = pd.DataFrame(df_thres)
#%%
# means
mean_acc_per_thres = df_thres.mean()
mean_burn_rate_per_thres = df_burn_rate.mean(axis=1)
mean_send_fraction_per_thres = df_send_fraction.mean(axis=1)

#standard deviations
std_burn_rate_per_thres = df_burn_rate.std(axis=1)

# what is the best value of each metric?
np.mean(auc_list)
#02-27: 0.7255787722923981
#03-01: 0.718223432156334
#03-05: 0.663328330817892 used map for optimization

mean_acc_per_thres.max()
mean_burn_rate_per_thres.max()
#02-27: 0.16357887618580846
#03-01: 0.16457876109142874
#03-05: 0.09922024064391663 failed experiment

# what is the best threshold setting of each metric?
mean_acc_per_thres.idxmax()
mean_burn_rate_per_thres.idxmax()
#02-27: 62
#03-01: 53
#03-06: 36

# send reduction at best threshold
mean_send_fraction_per_thres.loc[mean_burn_rate_per_thres.idxmax()]*100
#02-27: 0.0056509999999999755
#03-01: 0.31089999999999995
#03-05: 2.28 send percentage much better! maybe not failed experiment?

#%%
y_train = train_df.target
X_train = train_df[col_filter]
                     
xgb_param = {'base_score': 0.5,
                     'booster': 'gbtree',
                     'colsample_bylevel': 1,
                     'colsample_bytree': csbt,
                     'gamma': gam,
                     'learning_rate': lr,
                     'max_delta_step': 0,
                     'max_depth': md,
                     'min_child_weight': mcw,
                     'missing': None,
#                     'n_estimators': max_trees,
                     'n_jobs': ttu,
                     'objective': 'binary:logistic',
                     'reg_alpha': ra,
                     'reg_lambda': rl,
                     'scale_pos_weight': spw,
                     'random_state': myseed,
                     'subsample': ss,
                     'silent':1}

xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)                         
estimator = xgb.train( xgb_param, xgtrain, num_boost_round=ntrees)

estimator.save_model('data/20190305_xgb.model')
#%% SHAP Values
import shap
import matplotlib.pylab as pl
 
#%%  
shap_values = shap.TreeExplainer(estimator).shap_values(X_train.values)
 
#%% Most Imp NEW- SHAP vlaues
 
global_shap_vals = np.abs(shap_values).mean(0)[:-1] 
inds = np.argsort(global_shap_vals) 
most_imp = pd.DataFrame(global_shap_vals[inds], index = X_train.columns[inds],columns=['mean SHAP value magnitude (change in log odds)'])
#%%
shap.summary_plot(shap_values, X_train,max_display = 20, color='FF0000')
#%%
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
le_marital = joblib.load('data/marital_encoder.pkl') 
le_gender = joblib.load('data/gender_encoder.pkl')
X_display = X_train.copy()
X_display['marital_enc'] = le_marital.inverse_transform(X_display['marital_enc'])
X_display['gender_enc'] = le_gender.inverse_transform(X_display['gender_enc'])
#%%
shap.dependence_plot('uber_amt_201809', shap_values, X_train, interaction_index=24, display_features=X_display, color='#000000')

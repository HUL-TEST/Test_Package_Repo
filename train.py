import pandas as pd
import numpy as np
import os

#from StackReg.DataClean import clean 
#from StackReg.train_pred import StackingCLF

from DataClean import clean
from train_pred import StackingCLF
from amlrun import get_amlrun

from azureml.core import Workspace,Datastore
from azureml.core import Experiment,Run
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.datasets import load_boston,load_diabetes
from sklearn.model_selection import train_test_split
import pickle

X1,y1 = load_boston().data,load_boston().target
X2,y2 = load_diabetes().data,load_diabetes().target

X1 = clean(X1,y = y1).return_clean_data()
X2 = clean(X2,y = y2).return_clean_data()

xtrain_bsnt,xtest_bstn,ytrain_bstn,ytest_bstn = train_test_split(X1,y1)
xtrain_diab,xtest_diab,ytrain_diab,ytest_diab = train_test_split(X2,y2)

#pd.concat([pd.DataFrame(xtest_bstn),pd.Series(ytest_bstn)],axis = 1).to_csv('Boston_Test.csv',index = False)
#pd.concat([pd.DataFrame(xtest_diab),pd.Series(ytest_diab)],axis = 1).to_csv('Diabetes_Test.csv',index = False)

learners = [GradientBoostingRegressor(random_state = 1000),Lasso(random_state=1000),
                                Ridge(random_state=1000)]

#ws = Workspace.from_config()
#exp = Experiment(workspace=ws,name = 'test_exp_1')
#run = get_amlrun()

###### Defining meta-learner params :

alpha = np.arange(start = 0.1,stop = 1.0,step = 10)
l1_ratio  =  np.arange(start = 0.1,stop = 1.0,step = 10)

run = Run.get_context()

i = 0
min_rmse = np.inf
min_rmse_ind = 0

for a,b in zip(alpha,l1_ratio):

    meta_learner = ElasticNet(random_state=1000,alpha=alpha,l1_ratio=l1_ratio)    
    clf = StackingCLF(learners=learners,meta_learner=meta_learner)
    clf_fit = clf.fit(xtrain_bsnt,ytrain_bstn)
    clf_preds = clf_fit.predict(xtrain_bsnt)

    rmse_score = np.sqrt(mean_squared_error(ytrain_bstn,clf_preds))

    if rmse_score<min_rmse:
        min_rmse = rmse_score
        min_rmse_ind = i
        model_pkl_name = 'best_model_stack1'+'.pkl'
        output_loc = os.getcwd()+'/outputs/'+model_pkl_name
        pickle.dump(clf_fit,open(output_loc,'wb'))


if run!= None:
    run.log('Alpha',alpha[min_rmse_ind])
    run.log('L1_Ratio',l1_ratio[min_rmse_ind])
    run.log('rmse_score',min_rmse)
    run.log('Output Path',output_loc)
    run.log('Best Model',model_pkl_name)

run.complete()



import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import ClassifierMixin,RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone

class StackingCLF(RegressorMixin,TransformerMixin):
    
    def __init__(self, learners = None,meta_learner = None):
        
        assert isinstance(learners,list), print('Provide estimators as a list object.')
        
        for x in learners:
            try:
                check_estimator(x)
            except Exception:
                print('Provide and sklearn estimator object.%s object does not have get_params function.' % (x))

        self.learners = learners
        self.meta_learner = meta_learner

        if (len(learners)>1) and (meta_learner == None):
            print('When multiple learners are provided, need meta-learner.')
            return

    def fit(self,X,y):
        preds_df = pd.DataFrame(index = range(X.shape[0]))
        self.learner_fit = {}
        for clf in self.learners:
            self.learner_name = str(type(clf)).split('.')[-1].split('\'')[0]
            self.learner_fit[self.learner_name] = clf.fit(X,y)
            preds_df[self.learner_name+'_preds'] = self.learner_fit[self.learner_name].predict(X)
            
        if self.meta_learner != None:
            meta_lr_name = str(type(self.meta_learner)).split('.')[-1].split('\'')[0]
            self.meta_fit = self.meta_learner.fit(preds_df,y)
            preds_df[meta_lr_name+'_meta_preds'] = self.meta_fit.predict(preds_df)
            
        return self

    def predict(self,X):
        if self.meta_learner == None:
            return self.learner_fit[self.learner_name].predict(X)
        else:
            preds_df1 = pd.DataFrame(index = range(X.shape[0]))
            for clf in self.learner_fit.keys():
                preds_df1[clf] = self.learner_fit[clf].predict(X)
            
            preds_df1['meta_preds'] = self.meta_fit.predict(preds_df1)
        
        return preds_df1['meta_preds']


        





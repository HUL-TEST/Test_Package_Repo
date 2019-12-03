
import pandas as pd
import numpy as np
from sklearn.utils import check_X_y
import os

class clean:
    
    def __init__(self,X,y = [],target_col = None ,check_x_y = True,check_obj_cols = True,missing_type = np.nan,ohe= True):
        
        self.check_x_y = check_x_y
        self.check_obj_cols = check_obj_cols
        self.missing_type = missing_type
        self.ohe = ohe

        if target_col == None:
            if len(y) == 0:
                print('It y is missing then input target columns name!!')
                return
            else:
                self.X,self.y = X,y
        else:
           print(X.shape)
           self.X,self.y = X.drop([target_col],axis = 1) ,X[target_col]
        
        assert isinstance(self.y,pd.core.api.Series) | isinstance(self.y,np.core.ndarray) , print('Target should be Series or an array ')
        assert isinstance(self.X,pd.core.api.DataFrame) | isinstance(self.X,np.core.ndarray) , print('X must be a dataframe or numpy-nd array')
         
        self.data_clean()
        
    
    def treat_nans(self):
            
        if not isinstance(self.X,pd.core.api.DataFrame):
            self.X = pd.DataFrame(self.X)
            
        if self.missing_type == np.nan:
           for c in self.X.columns:
                if self.X[c].dtype == object:
                    self.X[c] = self.X.fillna(self.X[c].value_counts().idxmax())
                else:
                    self.X[c] = self.X[c].fillna(self.X[c].mean())
        else:
            for c in self.missing_cols:
                try:
                    val = self.X[self.X[c]!=self.missing_type][c].astype(float).mean()    
                except ValueError:
                    val = self.X[c].value_counts().idxmax()
            
                self.X[c] = self.X[c].replace(to_replace = self.missing_type,value = val)
        


                    
            
    def check_x_y_arr(self):
            ###check for consistency of X and y.

            self.ismissing = False
            self.missing_cols = []
            if self.missing_type == np.nan:
                try:
                    self.X,self.y = check_X_y(self.X,self.y,y_numeric = True)
                except ValueError:
                    self.missing_cols = self.X.columns[self.X.isna().any()].tolist()
                    self.ismissing = True
            else:
                self.missing_cols = [ c for c in self.X.columns if self.missing_type in self.X[c].values.tolist() ]
                self.ismissing = True if self.missing_cols != [] else False
            
            return
        
    
    def get_obj_cols(self):
        ### This is to get list of columns that have object datatype:
            self.obj_cols = []
            self.non_obj_cols = []
            if not isinstance(self.X,pd.core.api.DataFrame):
                self.X = pd.DataFrame(self.X)
        
            self.obj_cols = self.X.select_dtypes(include = ['object']).columns.tolist()
            self.non_obj_cols = self.X.select_dtypes(include = [int,float]).columns.tolist()
    
    def get_ohe(self):
        self.no_dummy_cols = []
        for s in self.obj_cols:
            if (self.X[s].nunique() <= 2) and (self.X[s].dtype != 'object'):
                self.no_dummy_cols.append(s)
                continue
        
        self.X = pd.get_dummies(data = self.X,columns = set(self.obj_cols).difference(set(self.no_dummy_cols)))
        
            
    def data_clean(self):
        
        if self.check_obj_cols:
            self.get_obj_cols()
        
        if self.check_x_y :
            self.check_x_y_arr()
            if self.ismissing == True:
                self.treat_nans()
        
        if self.ohe:
            if self.obj_cols != None: 
                self.get_ohe()

        print('Completed cleaning')
        
    
    def return_clean_data(self):
        return self.X



################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# --- Ensemble of Ensemble
from sklearn.ensemble import VotingRegressor
from itertools import combinations

import sys
sys.path.append(r'D:/작업방/업무 - 자동차 ★★★/Workspace_Python/DS_Module')
from DS_DataFrame import *
from DS_DataFrame import DataHandler


import copy
import functools
import numpy as np
import pandas as pd
import torch
import tensorflow as tf






# ('matrix_info' : ('frame_info' : ['data', 'index', 'columns']), kind, dtypes, nuniques)
# Vector 형태 (1차원 vector 또는 (-1,1) Shpaed Matrix)의 숫자형 데이터에 Scaler를 적용하는 Class
class ScalerVector:
    """
    【required (Library)】 numpy, pandas, sklearn.preprocessing.*, copy.deepcopy
    【required (Function)】DataHandler, class_object_execution

    < Input >
     . scaler : Scaler Object or String
                * required: 'str type Name_of_Class', 'str type Name_of_Instance', 'Class object', 'instance object'
     . x : '1dim vector' or '(-1, 1)shaped matrix'

    < Method >
     . fit
     . transform
     . fit_transform
     . inverse_transform
    """
    def __init__(self, scaler='StandardScaler', **kwargs):
        self.name = 'Undefined'
        self.scaler = class_object_execution(scaler, **kwargs)
        self.DataHandler = DataHandler()

    def fit(self, x):
        fitted_info = self.DataHandler.data_info(x, save_data=False)
        self.fitted_ndim = fitted_info.ndim
        self.fitted_object = self.DataHandler.vector_info_split(x)
        self.scaler.fit(self.fitted_object.data['data'].reshape(-1,1))
        self.name = self.fitted_object.data['name']
        self.transformed_names = [self.name]

    def transform(self, x, fitted_format=False, apply_name=False, ndim=None, kind=None):
        transformed_info = self.DataHandler.data_info(x, save_data=False)
        transformed_object = self.DataHandler.vector_info_split(x)
        transformed_data = self.scaler.transform(transformed_object.data['data'].reshape(-1,1))

        if fitted_format:
            apply_name = transformed_object.data['name'] if apply_name is False else apply_name
            return self.DataHandler.transform(transformed_data, apply_instance=self.fitted_object, apply_columns=apply_name,
                apply_ndim=self.fitted_ndim, apply_index=transformed_object.data['index'], apply_dtypes='float')
        else:
            apply_kind = True if kind is None else kind
            apply_ndim = transformed_info.ndim if ndim is None else ndim
            return self.DataHandler.transform(transformed_data, apply_instance=transformed_object, apply_columns=True,
                apply_kind=apply_kind, apply_ndim=apply_ndim, apply_dtypes='float')

    def fit_transform(self, x, ndim=None, kind=None):
        self.fit(x)
        fitted_format = True if ndim is None and kind is None else False
        return self.transform(x, fitted_format=fitted_format, apply_name=True, ndim=ndim, kind=kind)

    def inverse_transform(self, x, fitted_format=True, apply_name=False, ndim=None, kind=None, dtypes=None):
        inversed_info = self.DataHandler.data_info(x, save_data=False)
        inversed_object = self.DataHandler.vector_info_split(x)
        inversed_data = self.scaler.inverse_transform(inversed_object.data['data'].reshape(-1,1))

        if fitted_format:
            apply_name = inversed_object.data['name'] if apply_name is False else apply_name
            # apply_kind = self.fitted_object.kind if kind is None else kind
            apply_ndim = self.fitted_ndim
            apply_dtypes = dtypes if dtypes is not None else True
            return self.DataHandler.transform(inversed_data, apply_instance=self.fitted_object, apply_columns=apply_name,
                apply_ndim=apply_ndim,
                apply_dtypes=apply_dtypes, apply_index=inversed_object.data['index'])
        else:
            apply_name = inversed_object.data['name'] if apply_name is False else apply_name
            apply_kind = True if kind is None else kind
            apply_ndim = inversed_info.ndim if ndim is None else ndim
            apply_dtypes = dtypes if dtypes is not None else self.fitted_object.dtypes
            return self.DataHandler.transform(inversed_data, apply_instance=inversed_object, apply_columns=apply_name,
                apply_kind=apply_kind, apply_ndim=apply_ndim, 
                apply_dtypes=apply_dtypes, apply_index=True)

    def __repr__(self):
        return f"(ScalerInstance) {self.name}: {self.scaler}"


# Vector 형태 (1차원 vector 또는 (-1,1) Shpaed Matrix)의 문자형 데이터에 Encoder를 적용하는 Class
class EncoderVector:
    """
    【required (Library)】 numpy, pandas, sklearn.preprocessing.*, copy.deepcopy
    【required (Function)】DataHandler, class_object_execution

    < Input >
     . encoder : Scaler Object or String
                * required: 'str type Name_of_Class', 'str type Name_of_Instance', 'Class object', 'instance object'
     . x : '1dim vector' or '(-1, 1)shaped matrix'

    < Method >
     . fit
     . transform
     . fit_transform
     . inverse_transform
     . get_params
     . get_feature_names_out
    """
    def __init__(self, encoder='OneHotEncoder', **kwargs):
        self.name='undefined'

        if 'OneHotEncoder' in str(encoder):
            if 'drop' not in kwargs.keys():
                kwargs.update({'drop':'first'})
            if 'sparse' not in kwargs.keys():
                kwargs.update({'sparse':False})
        # self.kwargs = kwargs
        self.DataHandler = DataHandler()
        self.encoder = class_object_execution(encoder, **kwargs)

    def fit(self, x):
        encoder_str = str(self.encoder)
        fitted_info = self.DataHandler.data_info(x, save_data=False)
        self.fitted_ndim = fitted_info.ndim
        self.fitted_object = self.DataHandler.vector_info_split(x)
        fitted_series = pd.Series(**self.fitted_object.data)
        # fitted_series = pd.Series(**self.fitted_object.data).apply(lambda x: str(x))
        
        if 'OneHotEncoder' in encoder_str:
            self.encoder.fit(fitted_series.to_frame())
            try:
                self.transformed_names = list(map(lambda x: str(self.fitted_object.data['name']) + str(x)[2:], self.encoder.get_feature_names_out()))
            except:
                self.transformed_names = list(map(lambda x: str(self.fitted_object.data['name']) + str(x)[2:], self.encoder.get_feature_names()))
        elif 'LabelEncoder' in encoder_str:
            self.encoder.fit(fitted_series)
            self.transformed_names = [self.fitted_object.data['name']]
        elif 'OrdinalEncoder' in encoder_str:
            self.encoder.fit(fitted_series.to_frame())
            self.transformed_names = [self.fitted_object.data['name']]

        self.name = self.fitted_object.data['name']

    def transform(self, x, fitted_format=False, apply_name=False, ndim=None, kind=None):
        encoder_str = str(self.encoder)
        transformed_info = self.DataHandler.data_info(x, save_data=False)
        transformed_object = self.DataHandler.vector_info_split(x)
        transformed_series = pd.Series(**transformed_object.data)
        # transformed_series = pd.Series(**transformed_object.data).apply(lambda x: str(x))
        if 'OneHotEncoder' in encoder_str or 'OrdinalEncoder' in encoder_str:
            transformed_data = self.encoder.transform(transformed_series.to_frame())
        elif 'LabelEncoder' in encoder_str:
            transformed_data = self.encoder.transform(transformed_series)
        
        # name
        if (fitted_format is True) or (apply_name is True):
            apply_name = self.transformed_names
        elif apply_name is False:
            if 'OneHotEncoder' in encoder_str:
                try:
                    apply_name = list(map(lambda x: str(transformed_object.data['name']) + str(x)[2:], self.encoder.get_feature_names_out()))
                except:
                    apply_name = list(map(lambda x: str(transformed_object.data['name']) + str(x)[2:], self.encoder.get_feature_names()))
            else:
                apply_name = [transformed_object.data['name']]
        else:
            if 'OneHotEncoder' in encoder_str:
                try:
                    apply_name = list(map(lambda x: str(apply_name) + str(x)[2:], self.encoder.get_feature_names_out()))
                except:
                    apply_name = list(map(lambda x: str(apply_name) + str(x)[2:], self.encoder.get_feature_names()))
            else:
                apply_name = apply_name

        # transform
        if 'OneHotEncoder' not in encoder_str:
            apply_name = apply_name[0]
            if fitted_format:
                return self.DataHandler.transform(transformed_data, apply_instance=self.fitted_object, apply_columns=apply_name,
                            apply_ndim=self.fitted_ndim, apply_index=transformed_object.data['index'], apply_dtypes='int')
            else:
                apply_kind = True if kind is None else kind
                apply_ndim = transformed_info.ndim if ndim is None else ndim
                return self.DataHandler.transform(transformed_data, apply_instance=transformed_object, apply_columns=True,
                    apply_kind=apply_kind, apply_ndim=apply_ndim, apply_dtypes='int')
        else:       # OneHotEncoder
            parmas = self.encoder.get_params()
            if parmas['sparse']:
                transformed_data = transformed_data.toarray()
            
            if fitted_format:
                # apply_kind = self.fitted_object.kind if kind is None else kind
                apply_ndim = self.fitted_ndim if transformed_data.shape[1] == 1 else 2
                return self.DataHandler.transform(transformed_data, apply_kind=self.fitted_object.kind, apply_columns=apply_name,
                    apply_ndim=apply_ndim, apply_index=transformed_object.data['index'], apply_dtypes='int')
            else:
                apply_kind = transformed_object.kind if kind is None else kind
                apply_ndim = (transformed_info.ndim if ndim is None else ndim) if transformed_data.shape[1] == 1 else 2

                return self.DataHandler.transform(transformed_data, apply_kind=apply_kind, apply_columns=apply_name,
                    apply_ndim=apply_ndim, apply_index=transformed_object.data['index'], apply_dtypes='int')

    def fit_transform(self, x, ndim=None, kind=None):
        self.fit(x)
        fitted_format = True if ndim is None and kind is None else False
        return self.transform(x, fitted_format=fitted_format, apply_name=True, ndim=ndim, kind=kind)

    def inverse_transform(self, x, fitted_format=True, apply_name=False, ndim=None, kind=None, dtypes=None):
        encoder_str = str(self.encoder)
        inversed_info = self.DataHandler.data_info(x, save_data=False)
        inversed_object = self.DataHandler.data_info_split(x, ndim=2)
        inversed_data = self.encoder.inverse_transform(inversed_object.data['data'])

        # name
        if (fitted_format is True) or (apply_name is True):
            apply_name = self.fitted_object.data['name']
        elif apply_name is False:
            if 'OneHotEncoder' in encoder_str:
                apply_name = list(map(lambda x: x[:x.rfind('_')], inversed_object.data['columns']))[0]
            else:
                apply_name = inversed_object.data['columns']
        else:
            apply_name = apply_name if type(apply_name) == list else [apply_name]

        if fitted_format:
            apply_dtypes = dtypes if dtypes is not None else True
            return self.DataHandler.transform(inversed_data, apply_instance=self.fitted_object, apply_columns=apply_name,
                apply_ndim=self.fitted_ndim, apply_index=inversed_object.data['index'], apply_dtypes=apply_dtypes)
        else:
            apply_kind = True if kind is None else kind
            apply_ndim = inversed_info.ndim if ndim is None else ndim
            apply_dtypes = dtypes if dtypes is not None else False
            
            return self.DataHandler.transform(inversed_data, apply_instance=inversed_object, apply_columns=apply_name,
                apply_kind=apply_kind, apply_ndim=apply_ndim, 
                apply_dtypes=apply_dtypes, apply_index=True)

    def get_params(self):
        return self.encoder.get_params()

    def get_feature_names_out(self):
        return np.array(self.transformed_names)

    def __repr__(self):
        return f"(EncoderInstance) {self.name}: {self.encoder}"



### ★★★ ###
# Matrix/Frame/DataFrame 데이터에 Scaler 또는 Encoder를 적용하는 Class
class ScalerEncoder:
    """
    【required (Library)】 numpy, pandas, sklearn.preprocessing.*, copy.deepcopy, functools.reduce
    【required (Function)】DataHandler, class_object_execution, ScalerVector, EncoderVector, dtypes_split

    < Input >
     . encoder : dictionay type {'columns' : Scaler/Encoder Object or String, ...}
                      (default) {'#numeric' : 'StandardScaler', '#object':'OneHotEncoder', '#time', 'StandardScaler'}
                * required Scaler/Encoder: 'str type Name_of_Class', 'str type Name_of_Instance', 'Class object', 'instance object'
     . X : 1dim, 2dim vector or matrix

    < Method >
     . fit
     . transform
     . fit_transform
     . inverse_transform
    """
    def __init__(self, encoder=None, **kwargs):
        self.apply_encoder = {'#numeric':'StandardScaler', '#object':'OneHotEncoder', '#time': 'StandardScaler'}
        if encoder is not None:
            self.apply_encoder.update(encoder)

        self.DataHandler = DataHandler()
        self.kwargs = kwargs
        self.encoder = {}
        self.match_columns = {}

    def fit(self, X):
        self.encoder = {}

        fitted_info = self.DataHandler.data_info(X, save_data=False)
        self.fitted_ndim = fitted_info.ndim
        self.fitted_object = self.DataHandler.data_info_split(X, ndim=2)

        fitted_DataFrame = pd.DataFrame(**self.fitted_object.data).astype(self.fitted_object.dtypes)
        
        self.columns_dtypes = pd.DataFrame(dtypes_split(fitted_DataFrame, return_type='columns_all')).T

        for c in fitted_DataFrame:
            if c in self.apply_encoder.keys():
                if 'scaler' in str(self.apply_encoder[c]).lower():
                    se = ScalerVector(scaler=self.apply_encoder[c])
                elif 'encoder' in str(self.apply_encoder[c]).lower():
                    se = EncoderVector(encoder=self.apply_encoder[c])
            else:
                apply_se = self.apply_encoder['#' + self.columns_dtypes.loc[c, 'dtype_group']]
                if 'scaler' in str(apply_se).lower():
                    se = ScalerVector(scaler=apply_se)
                elif 'encoder' in str(apply_se).lower():
                    se = EncoderVector(encoder=apply_se)
            se.fit(fitted_DataFrame[c])
            self.encoder[c] = copy.deepcopy(se)
            self.match_columns[c] = se.transformed_names

    def transform(self, X, fitted_format=False, columns=None, ndim=None, kind=None):
        transformed_info = self.DataHandler.data_info(X, save_data=False)
        transformed_object = self.DataHandler.data_info_split(X, ndim=2)
        
        X_DataFrame = pd.DataFrame(**transformed_object.data).astype(transformed_object.dtypes)
        if transformed_object.kind != 'pandas':
            X_DataFrame.columns = self.encoder.keys() if columns is None else columns
        
        # transform ***
        transformed_DataFrame = pd.DataFrame()
        for c in X_DataFrame:
            transformed_columnvector = pd.DataFrame(self.encoder[c].transform(X_DataFrame[c], fitted_format=True))
            transformed_DataFrame = pd.concat([transformed_DataFrame, transformed_columnvector], axis=1)
        
        # return ***
        if fitted_format:
            apply_ndim = self.fitted_ndim if transformed_DataFrame.shape[1] == 1 else 2
            return self.DataHandler.transform(transformed_DataFrame, apply_kind=self.fitted_object.kind, apply_ndim=apply_ndim)
        else:
            apply_kind = transformed_object.kind if kind is None else kind
            apply_ndim = (transformed_info.ndim if ndim is None else ndim) if transformed_DataFrame.shape[1] == 1 else 2
            return self.DataHandler.transform(transformed_DataFrame, apply_kind=apply_kind, apply_ndim=apply_ndim)
        
    def fit_transform(self, X, ndim=None, kind=None):
        self.fit(X)
        fitted_format = True if ndim is None and kind is None else False
        return self.transform(X, fitted_format=fitted_format, ndim=ndim, kind=kind)

    def inverse_transform(self, X, fitted_format=False, columns=None, ndim=None, kind=None, dtypes=None):
        inversed_info = self.DataHandler.data_info(X, save_data=False)
        inversed_object = self.DataHandler.data_info_split(X, ndim=2)
        
        X_DataFrame = pd.DataFrame(**inversed_object.data).astype(inversed_object.dtypes)
        if inversed_object.kind != 'pandas':
            X_DataFrame.columns = reduce(lambda x,y : x + y, self.match_columns.values()) if columns is None else columns

        Xcolumns = copy.deepcopy(X_DataFrame.columns)
        match_columns = copy.deepcopy(self.match_columns)

        # inverse_transform ***
        inversed_DataFrame = pd.DataFrame()
        while bool(len(Xcolumns)):
            inversed_target = pd.DataFrame()

            c = Xcolumns[0]
            Xcolumns = Xcolumns.drop(c)
            # print(c, Xcolumns)
            if type(dtypes) == dict:
                if c in dtypes:
                    apply_dtypes = dtypes[c]
                else:
                    apply_dtypes = None
            else:       # bool, str, dtype
                apply_dtypes = dtypes

            for fc, tc in match_columns.items():
                if c in tc:
                    inversed_target = X_DataFrame[tc]

                    del match_columns[fc]
                    tc.remove(c)
                    Xcolumns = Xcolumns.drop(tc)
                    
                    inversed_data = self.encoder[fc].inverse_transform(inversed_target, fitted_format=fitted_format, ndim=2, dtypes=apply_dtypes)
                    inversed_data.columns = [self.encoder[fc].name]

                    inversed_DataFrame = pd.concat([inversed_DataFrame, inversed_data], axis=1)
                    break
            
            if c == X_DataFrame.columns[-1]:
                break
            
        # return ***
        if fitted_format:
            apply_ndim = self.fitted_ndim if inversed_DataFrame.shape[1] == 1 else 2
            # apply_dtypes = dict(filter(lambda x: x[0] in X_DataFrame.columns, self.fitted_object.dtypes.items()))
            apply_dtypes = dict(filter(lambda x: x[0] in inversed_DataFrame.columns, self.fitted_object.dtypes.items()))
            # print(inversed_DataFrame, self.fitted_object.kind, apply_ndim, apply_dtypes)
            return self.DataHandler.transform(inversed_DataFrame, apply_kind=self.fitted_object.kind, 
                    apply_ndim=apply_ndim, apply_dtypes=apply_dtypes)
        else:
            apply_kind = inversed_object.kind if kind is None else kind
            apply_ndim = (inversed_info.ndim if ndim is None else ndim) if inversed_DataFrame.shape[1] == 1 else 2
            return self.DataHandler.transform(inversed_DataFrame, apply_kind=apply_kind, apply_ndim=apply_ndim, apply_dtypes=dtypes)

    def __repr__(self):
        return f"(ScalerEncoder) {self.encoder}"



### ★★★ ###
# Matrix/Frame/DataFrame 데이터에 Scaler 또는 Encoder를 적용하는 Class
class ScalerEncoder:
    """
    【required (Library)】 numpy, pandas, sklearn.preprocessing.*, copy.deepcopy, functools.reduce
    【required (Function)】DataHandler, class_object_execution, ScalerVector, EncoderVector, dtypes_split

    < Input >
     . encoder : dictionay type {'columns' : Scaler/Encoder Object or String, ...}
                      (default) {'#numeric' : 'StandardScaler', '#object':'OneHotEncoder', '#time', 'StandardScaler'}
                * required Scaler/Encoder: 'str type Name_of_Class', 'str type Name_of_Instance', 'Class object', 'instance object'
     . X : 1dim, 2dim vector or matrix

    < Method >
     . fit
     . transform
     . fit_transform
     . inverse_transform
    """
    def __init__(self, encoder=None, **kwargs):
        self.apply_encoder = {'#numeric':'StandardScaler', '#object':'OneHotEncoder', '#time': 'StandardScaler'}
        if encoder is not None:
            self.apply_encoder.update(encoder)

        self.DataHandler = DataHandler()
        self.kwargs = kwargs
        self.encoder = {}
        self.match_columns = {}

    def fit(self, X):
        self.encoder = {}

        fitted_info = self.DataHandler.data_info(X, save_data=False)
        self.fitted_ndim = fitted_info.ndim
        self.fitted_object = self.DataHandler.data_info_split(X, ndim=2)

        fitted_DataFrame = pd.DataFrame(**self.fitted_object.data).astype(self.fitted_object.dtypes)
        
        self.columns_dtypes = pd.DataFrame(dtypes_split(fitted_DataFrame, return_type='columns_all')).T

        for c in fitted_DataFrame:
            if c in self.apply_encoder.keys():
                if 'scaler' in str(self.apply_encoder[c]).lower():
                    se = ScalerVector(scaler=self.apply_encoder[c])
                elif 'encoder' in str(self.apply_encoder[c]).lower():
                    se = EncoderVector(encoder=self.apply_encoder[c])
            else:
                apply_se = self.apply_encoder['#' + self.columns_dtypes.loc[c, 'dtype_group']]
                if 'scaler' in str(apply_se).lower():
                    se = ScalerVector(scaler=apply_se)
                elif 'encoder' in str(apply_se).lower():
                    se = EncoderVector(encoder=apply_se)
            se.fit(fitted_DataFrame[c])
            self.encoder[c] = copy.deepcopy(se)
            self.match_columns[c] = se.transformed_names

    def transform(self, X, fitted_format=False, columns=None, ndim=None, kind=None):
        transformed_info = self.DataHandler.data_info(X, save_data=False)
        transformed_object = self.DataHandler.data_info_split(X, ndim=2)
        
        X_DataFrame = pd.DataFrame(**transformed_object.data).astype(transformed_object.dtypes)
        if transformed_object.kind != 'pandas':
            X_DataFrame.columns = self.encoder.keys() if columns is None else columns
        
        # transform ***
        transformed_DataFrame = pd.DataFrame()
        for c in X_DataFrame:
            transformed_columnvector = pd.DataFrame(self.encoder[c].transform(X_DataFrame[c], fitted_format=True))
            transformed_DataFrame = pd.concat([transformed_DataFrame, transformed_columnvector], axis=1)
        
        # return ***
        if fitted_format:
            apply_ndim = self.fitted_ndim if transformed_DataFrame.shape[1] == 1 else 2
            return self.DataHandler.transform(transformed_DataFrame, apply_kind=self.fitted_object.kind, apply_ndim=apply_ndim)
        else:
            apply_kind = transformed_object.kind if kind is None else kind
            apply_ndim = (transformed_info.ndim if ndim is None else ndim) if transformed_DataFrame.shape[1] == 1 else 2
            return self.DataHandler.transform(transformed_DataFrame, apply_kind=apply_kind, apply_ndim=apply_ndim)
        
    def fit_transform(self, X, ndim=None, kind=None):
        self.fit(X)
        fitted_format = True if ndim is None and kind is None else False
        return self.transform(X, fitted_format=fitted_format, ndim=ndim, kind=kind)

    def inverse_transform(self, X, fitted_format=False, columns=None, ndim=None, kind=None, dtypes=None):
        inversed_info = self.DataHandler.data_info(X, save_data=False)
        inversed_object = self.DataHandler.data_info_split(X, ndim=2)
        
        X_DataFrame = pd.DataFrame(**inversed_object.data).astype(inversed_object.dtypes)
        if inversed_object.kind != 'pandas':
            X_DataFrame.columns = reduce(lambda x,y : x + y, self.match_columns.values()) if columns is None else columns

        Xcolumns = copy.deepcopy(X_DataFrame.columns)
        match_columns = copy.deepcopy(self.match_columns)

        # inverse_transform ***
        inversed_DataFrame = pd.DataFrame()
        while bool(len(Xcolumns)):
            inversed_target = pd.DataFrame()

            c = Xcolumns[0]
            Xcolumns = Xcolumns.drop(c)
            # print(c, Xcolumns)
            if type(dtypes) == dict:
                if c in dtypes:
                    apply_dtypes = dtypes[c]
                else:
                    apply_dtypes = None
            else:       # bool, str, dtype
                apply_dtypes = dtypes

            for fc, tc in match_columns.items():
                if c in tc:
                    inversed_target = X_DataFrame[tc]

                    del match_columns[fc]
                    tc.remove(c)
                    Xcolumns = Xcolumns.drop(tc)
                    
                    inversed_data = self.encoder[fc].inverse_transform(inversed_target, fitted_format=fitted_format, ndim=2, dtypes=apply_dtypes)
                    inversed_data.columns = [self.encoder[fc].name]

                    inversed_DataFrame = pd.concat([inversed_DataFrame, inversed_data], axis=1)
                    break
            
            if c == X_DataFrame.columns[-1]:
                break
            
        # return ***
        if fitted_format:
            apply_ndim = self.fitted_ndim if inversed_DataFrame.shape[1] == 1 else 2
            # apply_dtypes = dict(filter(lambda x: x[0] in X_DataFrame.columns, self.fitted_object.dtypes.items()))
            apply_dtypes = dict(filter(lambda x: x[0] in inversed_DataFrame.columns, self.fitted_object.dtypes.items()))
            # print(inversed_DataFrame, self.fitted_object.kind, apply_ndim, apply_dtypes)
            return self.DataHandler.transform(inversed_DataFrame, apply_kind=self.fitted_object.kind, 
                    apply_ndim=apply_ndim, apply_dtypes=apply_dtypes)
        else:
            apply_kind = inversed_object.kind if kind is None else kind
            apply_ndim = (inversed_info.ndim if ndim is None else ndim) if inversed_DataFrame.shape[1] == 1 else 2
            return self.DataHandler.transform(inversed_DataFrame, apply_kind=apply_kind, apply_ndim=apply_ndim, apply_dtypes=dtypes)

    def __repr__(self):
        return f"(ScalerEncoder) {self.encoder}"











class DataSet():
    """
    【required (Library)】copy, functools, numpy(np), pandas(pd), torch, tensorflow(tf)
    
    < Attribute >
      self.inputdata_info
      self.dataloader_info
      self.dataloader
      
    < Method >
      . self.Dataset
      . self.Split
      . self.Encoding (self.Decoding)
      . self.Batch
      . self.Reset_dataloader
      
    < Funtion >
      . self.make_data_info
      . self.dataloader_to_info
      . self.info_to_dataloader
      . self.data_transform_from_numpy
      . self.split_size
      . self.data_slicing
      . self.data_split
      . self.data_transform
      . self.make_batch
      
    
    """
    def __init__(self, X=None, y=None,  X_columns=None, y_columns=None, kwargs_columns={},
                type = ['pandas', 'numpy', 'tensorflow', 'torch'], set_type={},
                X_encoder=None, y_encoder=None, encoder={},
                shuffle=False, random_state=None, **kwargs):
        
        # all input value save
        local_values = locals().copy()
        
        dataloader_info = {}
        # dataloader = {}
        self.data_columns_couple = {None: None, 'X':X_columns, 'y': y_columns}
        
        for arg_name, arg_value in local_values.items():
            if arg_name not in ['self', 'random_state', 'kwargs']:
                exec(f"self.{arg_name} = arg_value")
                
                if arg_name in ['X', 'y']:
                    if arg_value is not None and arg_value.shape[0] > 0:
                        dataloader_info[arg_name] = {}
                        dataloader_info[arg_name]['data'] = self.make_data_info(arg_value, prefix=arg_name, columns=self.data_columns_couple[arg_name], return_data=True) if arg_value is not None else None
                    
        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_value is not None and kwarg_value.shape[0] > 0:
                kwargs_column = kwargs_columns[kwarg_name] if kwarg_name in kwargs_columns.keys() else None
                dataloader_info[kwarg_name] = {}
                dataloader_info[kwarg_name]['data'] = self.make_data_info(kwarg_value, prefix=kwarg_name, columns=kwargs_column, return_data=True) if kwarg_value is not None else None
        
        # self.local_values = local_values
        self.set_type = set_type
        # self.inputdata_info = dataloader_info
        
        initialize_object = self.info_to_dataloader(dataloader_info, set_type=set_type)
        self.dataloader = initialize_object['dataloader']
        self.dataloader_info = initialize_object['data_info']
        self.inputdata_info = copy.deepcopy(self.dataloader_info)
                
        self.random_state = random_state
        self.define_random_generate(random_state)
        self.indices = {}
        self.length = {}
        
        self.dataloader_process = ''

    # [funcation] ---------------------------
    def define_random_generate(self, random_state=None):
        self.random_generate = np.random.RandomState(self.random_state) if random_state is None else np.random.RandomState(random_state)
        # self.random_generate = np.random.default_rng(random_state)

    def make_data_info(self, data, prefix=None, columns=None, iloc_index=None, loc_index=None, dtype=None, return_data=False):
        prefix = 'v' if prefix is None else prefix
        type_of_data = str(type(data))
        
        array_data = np.array(data)
        shape = array_data.shape
        n_dim = len(shape)
        
        # type
        result_of_type = None
        if 'pandas' in type_of_data:
            result_of_type = 'pandas'
        elif 'numpy' in type_of_data:
            result_of_type = 'numpy'
        elif 'tensorflow' in type_of_data:
            result_of_type = 'tensorflow'
            # result_of_type =  'tensorflow - variable' if 'variable' in type_of_data.lower() else 'tensorflow - constant'      
        elif 'torch' in type_of_data:
            result_of_type = 'torch'
        else:
            result_of_type = 'else'
        
        # dtype
        if dtype is None:
            if  result_of_type == 'pandas':
                dtype = data.dtype if n_dim == 1 else data.dtypes.to_dict()
            else:
                dtype = data.dtype
            
            
        # columns
        if columns is not None:
            set_columns = columns
        elif result_of_type == 'pandas':
            set_columns = str(data.name) if n_dim == 1 else np.array(data.columns)
        else:
            if n_dim == 1:
                set_columns = prefix
            elif n_dim > 1:
                col_base = np.array(range(shape[-1])).astype('str')
                set_columns = np.tile(col_base, np.array(shape[:-2]).prod()).reshape(*list(shape[:-2]), shape[-1]).astype('str')
                set_columns = np.char.add(prefix, set_columns)
                
        # index
        if iloc_index is None and loc_index is None:
            if result_of_type == 'pandas':
                loc_index = np.array(data.index)
                iloc_index = np.array(range(shape[0]))
            else:
                loc_index = np.array(range(shape[0]))
                iloc_index = np.array(range(shape[0]))
        else:
            if result_of_type == 'pandas':
                loc_index = np.array(data.index) if loc_index is None else np.array(loc_index)
                iloc_index = (np.array(range(shape[0])) if loc_index is None else loc_index) if iloc_index is None else np.array(iloc_index)
            else:
                loc_index = (np.array(range(shape[0])) if iloc_index is None else iloc_index) if loc_index is None else np.array(loc_index)
                iloc_index = (np.array(range(shape[0])) if loc_index is None else loc_index) if iloc_index is None else np.array(iloc_index)
        
        if return_data:
            return {'type': result_of_type, 'ndim': n_dim, 'dtype': dtype, 'columns': set_columns, 'loc_index' :loc_index, 'iloc_index':iloc_index, 'data': array_data}
        else:
            return {'type': result_of_type, 'ndim': n_dim, 'dtype': dtype, 'columns': set_columns, 'loc_index' :loc_index, 'iloc_index':iloc_index}

    def dataloader_to_info(self, dataloader, data_info=None, update_loader=False):
        return_data_info = {}
        for name, dataset in dataloader.items():
            return_data_info[name] = {}
            for dataset_name, data in dataset.items():
                if data_info is None:
                    return_data_info[name][dataset_name] = self.make_data_info(data=data)
                else:
                    info = data_info[name][dataset_name]
                    return_data_info[name][dataset_name] = self.make_data_info(data=data, columns=info['columns'], iloc_index=info['iloc_index'], loc_index=info['loc_index'], return_data=True)
        if update_loader is True:
            return_dataloader = self.info_to_dataloader(return_data_info, update_info=False)
            return {'dataloader': return_dataloader['dataloader'], 'data_info':return_data_info}
        else:
            return {'dataloader': dataloader, 'data_info':return_data_info}

    def info_to_dataloader(self, data_info, set_type={}, update_info=False):
        return_dataloader = {}
        self.set_type.update(set_type)
        
        for name, dataset in data_info.items():
            return_dataloader[name] = {}
            for dataset_name, info in dataset.items():
                if self.set_type is None:
                    apply_type = info['type']
                elif type(self.set_type) == str:
                    apply_type = self.set_type
                elif type(self.set_type) == dict:
                    apply_type = self.set_type[name] if name in self.set_type.keys() else info['type']
                return_dataloader[name][dataset_name] = self.data_transform_from_numpy(numpy_data=info['data'], set_type=apply_type,
                                                        index=info['loc_index'], columns=info['columns'], dtype=info['dtype'])
        if len(self.set_type) > 0 or update_info is True:
            return_data_info = self.dataloader_to_info(return_dataloader, data_info=data_info, update_loader=False)
            return {'dataloader': return_dataloader, 'data_info':return_data_info['data_info']}
        else:
            return {'dataloader':return_dataloader, 'data_info':data_info}

    def data_transform_from_numpy(self, numpy_data, set_type, index=None, columns=None, dtype=None):
        shape = numpy_data.shape
        ndim = len(shape)
        
        # dtype
        if dtype is not None:
            if type(dtype) == dict:     # pandas
                if set_type == 'pandas':
                    apply_dtype = dtype
                else:
                    apply_dtype = np.dtype('float32')
            elif 'torch' in str(dtype):     # torch
                apply_dtype = np.dtype(str(dtype).split('.')[-1])
            elif '<dtype:' in str(dtype):   # tensorflow
                apply_dtype = np.dtype(dtype.as_numpy_dtype)
            else:
                apply_dtype = dtype
        else:
            apply_dtype = dtype
                
        # set_type transform
        if set_type == 'numpy':
            return numpy_data
        elif set_type == 'pandas':
            if ndim == 1:
                return pd.Series(numpy_data, index=index, name=columns, dtype=apply_dtype)
            elif ndim == 2:
                if apply_dtype is None:
                    return pd.DataFrame(numpy_data, index=index, columns=columns)
                else:
                    if type(apply_dtype) != dict:
                        if columns is None:
                            apply_dtype = {c: apply_dtype for c in range(numpy_data.shape[1])}
                        else:
                            apply_dtype = {c: apply_dtype for c in columns}
                    return pd.DataFrame(numpy_data, index=index, columns=columns).astype(apply_dtype)
        elif set_type == 'tensorflow':
            return tf.constant(numpy_data, dtype=apply_dtype)
        elif set_type == 'torch':
            return torch.FloatTensor(numpy_data)
            # return torch.tensor(numpy_data, dtype=eval(f'torch.{str(apply_dtype)}'))

    def split_size(self, valid_size, test_size):
        self.valid_size = valid_size
        self.test_size = test_size
        
        self.train_size = 1 - test_size
        self.train_train_size = self.train_size - valid_size
        self.train_valid_size = valid_size

    def generate_index(self, data, data_info=None, valid_size=None, test_size=None, shuffle=True,
                      random_state=None):
        try:
            data_length = len(data)
        except:
            data_length = data.shape[0]
        if data_info is None:
            data_info = self.make_data_info(data=data)
            
        index = data_info['iloc_index']
        indices = {}

        # shuffle
        if shuffle is True:
            random_generate = self.random_generate if random_state is None else np.random.RandomState(random_state)
            apply_index = random_generate.permutation(index)
        else:
            apply_index = index
        
        # split_size
        self.split_size(valid_size = 0.0 if valid_size is None else valid_size,
                        test_size = 0.3 if test_size is None else test_size)
        
        # train_valid_test split
        train_len = int(data_length * (1-self.test_size))
        train_train_len = int(train_len * (1-self.train_valid_size))
        train_valid_len = train_len - train_train_len
        test_len = data_length - train_len
        
        for k, v in zip(['data','train', 'train_train', 'train_valid', 'test'], [train_len, train_train_len, train_valid_len, test_len]):
            if v > 0:
                self.length[k] = v
        
        # save
        indices['all_index'] = index
        indices['apply_index'] = apply_index
        indices['train_index'] = apply_index[:train_len]
        indices['train_train_index'] = indices['train_index'][:train_train_len]
        if train_valid_len > 0:
            indices['train_valid_index'] = indices['train_index'][train_train_len:]
        indices['test_index'] = apply_index[train_len:]
        
        return indices

    def data_slicing(self, data=None, apply_index=None, data_info=None, index_type='iloc'):
        # data_info
        if data_info is None:
            data_info = self.make_data_info(data=data, return_data=True)

        numpy_data = data_info['data']
        
        if index_type == 'iloc':
            index_series = pd.Series(data_info['loc_index'], index=range(len(numpy_data)) )
            iloc_index = np.array(index_series[apply_index].index) 
            loc_index = np.array(index_series[apply_index].values)
        elif index_type == 'loc':
            index_series = pd.Series(range(len(numpy_data)), index=data_info['loc_index'])
            iloc_index = np.array(index_series[apply_index].values) 
            loc_index = np.array(index_series[apply_index].index)
        
        # slicing
        sliced_data = np.take(numpy_data, iloc_index, axis=0)
        # sliced_index = np.take(data_index, apply_index, axis=0)
        return self.data_transform_from_numpy(numpy_data=sliced_data, index=loc_index, 
                                            set_type=data_info['type'], columns=data_info['columns'], dtype=data_info['dtype'])       

    def data_split(self, data, data_info=None, indices=None, valid_size=0, test_size=0, shuffle=True,
                   index_type='iloc', random_state=None, verbose=0):
        # data_info
        if data_info is None:
            data_info = self.make_data_info(data=data, return_data=True)
            
        if indices is None:
            if len(self.indices) == 0:
                indices = self.generate_index(data=data, valid_size=valid_size, test_size=test_size, 
                                              shuffle=shuffle, random_state=random_state)
            else:
                indices = self.indices
        
        split_data = {}
        for key, index in indices.items():
            split_data[key] = self.data_slicing(data=data, apply_index=index, data_info=data_info, index_type=index_type)
            if verbose > 0:
                print(f"{key}: {split_data[key].shape}")
            
        return {'split_data': split_data, 'indices':indices}

    def data_transform(self, data, data_info=None, encoder=None, type='encoding'):
        if encoder is None:
            return data
        else:
            try:
                if 'enco' in type:
                    return encoder.transform(data)
                elif 'deco' in type:
                    return encoder.inverse_transform(data)
            except:
                # data_info
                if data_info is None:
                    data_info = self.make_data_info(data=data, return_data=True)
                if 'enco' in type:
                    transformed_np_data = encoder.transform(data_info['data'])
                elif 'deco' in type:
                    transformed_np_data = encoder.inverse_transform(data_info['data'])
                transformed_data = self.data_transform_from_numpy(numpy_data=transformed_np_data, 
                                               set_type=data_info['type'], columns=data_info['columns'], dtype=None)
                return transformed_data

    def make_batch(self, data, batch_size=None):
        try:
            data_length = len(data)
        except:
            data_length = data.shape[0]
        
        batch_size = data_length if batch_size is None or batch_size <= 0 or batch_size > data_length else batch_size
        batch_index = 0
        batch = []
        while True:
            if batch_index + batch_size >= data_length:
                batch.append(data[batch_index:])
                batch_index = data_length
                # batch = np.array(batch)
                break
            else:
                batch.append(data[batch_index:batch_index + batch_size])
                batch_index = batch_index + batch_size
        return batch

    # [method] ---------------------------
    def Reset_dataloader(self):
        self.dataloader_info = copy.deepcopy(self.inputdata_info)
        self.dataloader = self.info_to_dataloader(self.dataloader_info)['dataloader']
        self.dataloader_process = ''

    def Dataset(self, kwargs_columns={}, set_type={}, **kwargs):
        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_value is not None and kwarg_value.shape[0] > 0:
                kwargs_column = kwargs_columns[kwarg_name] if kwarg_name in kwargs_columns.keys() else None
                self.dataloader_info[kwarg_name] = {}
                self.dataloader_info[kwarg_name]['data'] = self.make_data_info(kwarg_value, prefix=kwarg_name, columns=kwargs_column, return_data=True) if kwarg_value is not None else None
                self.inputdata_info[kwarg_name] = {}
                self.inputdata_info[kwarg_name] = self.dataloader_info[kwarg_name]['data']
        # set dataset / data_info
        update_object = self.info_to_dataloader(dataloader_info, set_type=set_type)
        self.dataloader = self.info_to_dataloader(self.dataloader_info, set_type=set_type)['dataloader']
        self.dataloader_info = update_object['data_info']
        self.inputdata_info = copy.deepcopy(self.dataloader_info)
        return self

    def Encoding(self, X_encoder=None, y_encoder=None, encoder={}):
        # encoder dictionary
        self.encoder.update(encoder)
        X_encoder = self.X_encoder if X_encoder is None else X_encoder
        y_encoder = self.y_encoder if y_encoder is None else y_encoder
        if X_encoder is not None:
            encoder['X'] = X_encoder
        if y_encoder is not None:
            encoder['y'] = y_encoder
        
        # encoding
        encoder_keys = encoder.keys()
        for name, data_dict in self.dataloader.items():
            for dataset_name, data in data_dict.items():
                if name in encoder_keys and encoder[name] is not None:
                    data_info = self.dataloader_info[name][dataset_name]
                    encodered_data = self.data_transform(data=data, data_info=data_info, encoder=encoder[name], type='encoding')
                    self.dataloader[name][dataset_name] = encodered_data
                    self.dataloader_info[name][dataset_name] = self.make_data_info(data=encodered_data, columns=data_info['columns'], 
                                                                                    iloc_index= data_info['iloc_index'],
                                                                                    return_data=True)
        # __repr__
        self.dataloader_process = self.dataloader_process + ('\n . ' if self.dataloader_process == '' else ' > ') + 'Encoding'

    def Decoding(self, X_encoder=None, y_encoder=None, encoder={}):
        # encoder dictionary
        self.encoder.update(encoder)
        X_encoder = self.X_encoder if X_encoder is None else X_encoder
        y_encoder = self.y_encoder if y_encoder is None else y_encoder
        if X_encoder is not None:
            encoder['X'] = X_encoder
        if y_encoder is not None:
            encoder['y'] = y_encoder
        
        # decoding
        encoder_keys = encoder.keys()
        for name, data_dict in self.dataloader.items():
            for dataset_name, data in data_dict.items():
                if name in encoder_keys and encoder[name] is not None:
                    data_info = self.dataloader_info[name][dataset_name]
                    encodered_data = self.data_transform(data=data, data_info=data_info, encoder=encoder[name], type='decoding')
                    self.dataloader[name][dataset_name] = encodered_data
                    self.dataloader_info[name][dataset_name] = self.make_data_info(data=encodered_data, columns=data_info['columns'], 
                                                                                    iloc_index= data_info['iloc_index'],
                                                                                    dtype=self.inputdata_info[name]['data']['dtype'],
                                                                                    return_data=True)
        # __repr__
        self.dataloader_process = self.dataloader_process + ('\n . ' if self.dataloader_process == '' else ' > ') + 'Decoding'

    def Split(self, valid_size=0, test_size=0.3, shuffle=True, random_state=None):
        random_state = self.random_state if random_state is None else random_state
        
        # train_valid_test_split
        for name, data_dict in self.dataloader.items():
            data_info = self.dataloader_info[name]['data']
            splited_dict = self.data_split(data=data_dict['data'], data_info=data_info,
                            valid_size=valid_size, test_size=test_size, shuffle=shuffle, 
                            index_type='iloc', random_state=self.random_state)
            del splited_dict['split_data']['all_index']
            del splited_dict['split_data']['apply_index']
            del splited_dict['indices']['all_index']
            del splited_dict['indices']['apply_index']
            
            for dataset_name, index in splited_dict['indices'].items():
                data = splited_dict['split_data'][dataset_name]
                set_name = dataset_name.replace('index','set')
                self.dataloader[name][set_name] = data
                self.dataloader_info[name][set_name] = self.make_data_info(data=data,
                                                                   columns=data_info['columns'],
                                                                   iloc_index=index,
                                                                   loc_index=np.take(data_info['loc_index'], index),
                                                                   dtype=data_info['dtype'],
                                                                   return_data=True)
        # __repr__
        self.dataloader_process = self.dataloader_process + ('\n . ' if self.dataloader_process == '' else ' > ') + 'Split'

    def Batch(self, batch_size=None, shuffle=True, random_state=None):
        if shuffle is True:
            random_generate = self.random_generate if random_state is None else np.random.RandomState(random_state)
        
        # transform to batch
        for name, data_dict in self.dataloader.items():
            for dataset_name, data in data_dict.items():
                batch_data = []
                data_info = self.dataloader_info[name][dataset_name]
                dataset_index = data_info['loc_index']
                if shuffle is True:
                    dataset_index = random_generate.permutation(dataset_index)
                batch_indices = self.make_batch(dataset_index, batch_size=batch_size)
                
                for batch_index in batch_indices:
                    batch_sliced_data = self.data_slicing(data, data_info=data_info, apply_index=batch_index, index_type='loc')
                    batch_data.append( batch_sliced_data )
                
                # update dataloader
                self.dataloader[name][dataset_name] = batch_data
                
                # update dataloader_info
                self.dataloader_info[name][dataset_name]['batch_iloc_index'] = batch_indices
        # __repr__
        self.dataloader_process = self.dataloader_process + ('\n . ' if self.dataloader_process == '' else ' > ') + f'Batch({batch_size})'

    def __repr__(self):
        data_set_names = ', '.join(list(self.dataloader.keys()))
        return f'<class: DataLoader Object ({data_set_names})>{self.dataloader_process}'



# a0 = np.arange(60).reshape(-1,4)
# a1 = np.arange(120).reshape(-1,4)
# a1 = np.arange(60).reshape(-1,4)
# a2 = np.array(range(20)).ravel()
# a3 = a1.reshape(6,-1,5)

# a5 = pd.DataFrame(a1, columns=['A','CC','D','K']).sample(len(a1))
# a7 = pd.DataFrame(a0, columns=['A','CC','D','K'])
# a6 = pd.Series(a2).sample(len(a2))

# a10 = torch.FloatTensor(a1)
# a11 = tf.constant(a1, dtype='float')
# a12 = tf.Variable(a1, dtype='float')
# a13 = tf.Variable(a2, dtype='float')



# ds = DataSet(X=a1, y=a2, z=a7, w=a7.sample(12), 
#              kk=a7.sample(15), kwa=a7.sample(15), kwb=a7.sample(15), 
#              set_type={'kk':'torch', 'kwa':'numpy', 'kwb':'tensorflow'})
# ds.inputdata_info
# ds.dataloader_info['X']['train_valid_set']
# ds.dataloader

# ds.Split(valid_size=0.1)

# se = ScalerEncoder()
# se.fit(a1)
# se2 = ScalerEncoder()
# se2.fit(a7)

# ds.Encoding(encoder={'X':se, 'z':se2})
# ds.Decoding(encoder={'X':se, 'z':se2})
# ds.Batch(batch_size=7)

# ds.Reset_dataloader()








#### EarlyStopping ###########################################
import time
from IPython.display import clear_output

class EarlyStopping():
    """
    【 Required Library 】numpy, pandas, matplotlib.pyplot, time, from IPython.display import clear_output
     < Initialize(patience=4, optimize='minimize') >
      . patience: 1,2,3,4 ...
      . optimize: minimize / maximize 
     
     < early_stop(score, save=None, label=None, reference_score=None, reference_save=None, reference_label=None, verbose=0, sleep=0.05, save_all=False) >
      (input)
       . score: metrics_score
       . save: anything that would like to save at optimal point
       . label: plot label
       
       . reference_score: reference metrics score
       . reference_save: reference_save value
       . reference_label: plot reference_label
       
       . verbose: 0, 1, 'plot', 'all'
       . sleep: when plotting, sleeping time(seconds).
       . save_all:
     
    """
    def __init__(self, patience=4, optimize='miminize'):
        self.patience = np.inf if patience is None else patience
        self.optimize = optimize
        
        self.metrics = []       # (epoch, event, score, save, r_score, r_save)
        self.metrics_frame = pd.DataFrame()
        self.patience_scores = []
        self.optimum = (0, np.inf if 'min' in optimize else -np.inf, '', None, None)    # (epoch, score, save, r_score, r_save)
    
    def reset_patience_scores(self):
        self.patience_scores = []
    
    def early_stop(self, score, save=None, label=None,
                   reference_score=None, reference_save=None, reference_label=None,
                   verbose=0, sleep=0.02, save_all=False):
        
        result = 'none'
        epoch = len(self.metrics)+1
        label_score = 'score' if label is None else label
        label_r_score = 'r_score' if reference_label is None else reference_label
        
        if 'min' in self.optimize:
            if score < self.optimum[1]:     # optimum
                self.patience_scores = []
                result = 'optimum'
            else:
                self.patience_scores.append(score)
                if len(self.patience_scores) > self.patience:
                    result = 'break'
                else:
                    result = 'patience'
        elif 'max' in self.optimize:
            if score > self.optimum[1]:     # optimum
                self.patience_scores = []
                result = 'optimum'
            else:
                self.patience_scores.append(score)
                if len(self.patience_scores) > self.patience:
                    result = 'break'
                else:
                    result = 'patience'
        
        # state save
        state = (epoch, result, score, save, reference_score, reference_save) if (save_all is True or result == 'optimum') else (epoch, result, score, '', reference_score, '')
        self.metrics.append(state)

        # update state metrics
        if result == 'optimum':
            if  self.optimum[0] > 0:
                prev_optim_index = self.metrics.index( list(filter(lambda x: x[0]==self.optimum[0], self.metrics))[0] )
                if save_all is True:
                    self.metrics[prev_optim_index] = tuple( ('none' if ei==1 else element) for ei, element in enumerate(self.metrics[prev_optim_index]) )
                else:
                    self.metrics[prev_optim_index] = tuple( ('none' if ei==1 else ('' if ei in [3,5] else element) ) for ei, element in enumerate(self.metrics[prev_optim_index]) )
            self.optimum = (epoch, score, save, reference_score, reference_save)
        
        # metrics_frame = pd.concat([self.metrics_frame, pd.Series(state, index=['epoch', 'event', label_score, 'save', 'r_score', 'r_save'], name=len(self.metrics_frame)).to_frame().T], axis=0)
        metrics_frame = pd.DataFrame(self.metrics, columns=['epoch', 'event', label_score, 'save', label_r_score, 'r_save'])
        metrics_frame['event'] = pd.Categorical(metrics_frame['event'], categories=['none', 'patience', 'break', 'optimum'], ordered=True)
        metrics_frame[label_score] = metrics_frame[label_score].astype('float')
        metrics_frame[label_r_score] = metrics_frame[label_r_score].astype('float')
        
        # plot        
        if verbose == 'plot' or verbose=='all':
            clear_output(wait=True)
        self.plot = plt.figure()
        
        # reference_score
        if reference_score is not None:
            plt.plot(metrics_frame['epoch'], metrics_frame[label_r_score], 'o-', alpha=0.5, color='orange', label='reference' if reference_label is None else reference_label)
            
        plt.plot(metrics_frame['epoch'], metrics_frame[label_score], alpha=0.5, color='steelblue', label='estimate' if label is None else label)
        plt.legend(loc='upper right')
        
        metrics_colors = ['steelblue', 'gold', 'red', 'green']
        for me, (mgi, mgv) in enumerate(metrics_frame.groupby('event')):
            plt.scatter(mgv['epoch'], mgv[label_score], color=metrics_colors[me])            
        for mi, mg in metrics_frame[metrics_frame['event'] != ''].iterrows():
            event_name = 'p' if mg['event'] == 'patience' else ('★' if mg['event']=='optimum' else ('break' if mg['event'] == 'break' else ''))
            plt.text(mg['epoch'], mg[label_score], event_name)
        plt.xlabel('epoch')
        plt.ylabel('score')
        plt.yscale('symlog')
        if verbose == 'plot' or verbose=='all':
            plt.show()
            time.sleep(sleep)
        else:
            plt.close()
        
        # print state
        if (type(verbose)==int and verbose > 1) or verbose=='all':
            if (verbose in ['plot', 'all']) and result != 'optimum':
                print(f"(Optimum) epoch: {self.optimum[0]}, {label_score}: {str(self.optimum[1])[:6]}, {label_r_score}: {str(self.optimum[3])[:6]}")
            
            if reference_score is not None:
                print(f"epoch: {len(self.metrics)}, {label_score}: {str(score)[:6]}, {label_r_score}: {str(reference_score)[:6]} {f'**{result}' if result != 'none' else ''}")
            else:
                print(f"epoch: {len(self.metrics)}, {label_score}: {str(score)[:6]} {f'**{result}' if result != 'none' else ''}")
        elif verbose == 1:
            if result != 'break':
                print(epoch, end=' ')
            else:
                print(epoch, end=' *break\n')
                print(f"(Optimum) epoch: {self.optimum[0]}, {label_score}: {str(self.optimum[1])[:6]}, {label_r_score}: {str(self.optimum[3])[:6]}") 
        
        self.metrics_frame = metrics_frame.copy()
        return result




# Ensemble_Models
class EnsembleModels():
    def __init__(self, models=None, weights=None):
        self.models = models
        self.weights = [1/len(models)] * len(models) if weights is None else [w/sum(weights) for w in weights]

    def predict(self, x):
        pred_y = np.zeros(len(x))
        pred_y_list = []
        for mdl, w in zip(self.models, self.weights):
            mdl_pred_y = mdl.predict(x)
            pred_y_list.append(mdl_pred_y)
            pred_y += (mdl_pred_y * w)
        
        self.pred_y = pred_y    
        self.pred_y_list = pred_y_list
        return pred_y




################################################################################################################

# HyperParameter Tunning
from bayes_opt import BayesianOptimization, UtilityFunction

# (git) bayes_opt : https://github.com/fmfn/BayesianOptimization 
# (git_advance) bayes_opt : https://github.com/fmfn/BayesianOptimization/blob/master/examples/advanced-tour.ipynb
# (install) conda install -c conda-forge bayesian-optimization


# from bayes_opt import BayesianOptimization
# from bayes_opt import UtilityFunction
class BayesOpt:
    """
     【required (Library)】 bayes_opt.BayesianOptimization, bayes_opt.UtilityFunction
     【required (Custom Module)】 EarlyStopping
     
      . __init__(self, f, pbounds, random_state=None, verbose=2)
         f : function
         pbounds : {'x':(-150, 150), 'y':(-50, 100), 'z':(1000, 1200)}
         random_state : 1, 2, 3...
         verbose : 1, 2, 3... 
    """
    def __init__(self, f, pbounds, random_state=None, verbose=2):
        self.verbose = verbose
        self.f = f
        self.pbounds = pbounds
        self.random_state = random_state
        self.random_generate = np.random.RandomState(self.random_state)
        
        self.bayes_opt = BayesianOptimization(f=f, pbounds=pbounds, random_state=random_state, verbose=verbose)
        self._space = self.bayes_opt._space
        
        self.res = []
        self.max = {'target':-np.inf, 'params':{}}
        self.repr_max = {}
        
        self.last_state = ''
    
    def decimal(self, x, rev=0):
        return 2 if x == 0 else int(-1*(np.floor(np.log10(abs(x)))-3-rev))
    
    def auto_decimal(self, x, rev=0):
        if np.isnan(x):
            return np.nan
        else:
            decimals = self.decimal(x, rev=rev)
            if decimals < 0:
                return x
            else:
                return round(x, decimals)

    def print_result(self):
        epoch = len(self.bayes_opt._space.target)
        last_target = self.auto_decimal(self.bayes_opt._space.target[-1])
        last_params = {k: self.auto_decimal(v) for k, v in zip(self.bayes_opt._space.keys, self.bayes_opt._space.params[-1])}
        last_state = '**Maximum' if epoch == np.argmax(self.bayes_opt._space.target) + 1 else self.last_state
        
        if self.verbose > 0:
            if self.verbose > 1 or last_state == '**Maximum':
                print(f"{epoch} epoch) target: {last_target}, params: {str(last_params)[:255]} {last_state}")
        self.last_state = ''
    
    def maximize(self, init_points=5, n_iter=25, acq='ucb', kappa=2.576, xi=0.0, patience=None, **gp_params):
        if patience is not None:
            bayes_utils = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
            n = 1
            
            # init_points bayesian
            for i in range(init_points):
                self.bayes_opt.probe(self.bayes_opt._space.random_sample(), lazy=False)
                self.print_result()
                n += 1
            
            # EarlyStop
            early_stop_instance = EarlyStopping(patience=patience, optimize='maximize')
            early_stop_instance.early_stop(score=self.bayes_opt.max['target'], save=self.bayes_opt.max['params'])
            
            last_state = 'break' if patience == 0 else None
            while last_state != 'break' or n < n_iter:
                # Bayesian Step
                next_points = self.bayes_opt.suggest(bayes_utils)
                next_target = self.f(**next_points)
                self.bayes_opt.register(params=next_points, target=next_target)
            
                if n >= n_iter:
                    last_state = early_stop_instance.early_stop(score=next_target, save=next_points)
                    self.last_state = '' if last_state == 'None' else last_state

                self.print_result()
                n += 1
            
        else:
            self.bayes_opt.maximize(init_points=init_points, n_iter=n_iter, acq=acq, kappa=kappa, xi=xi, **gp_params)
        
        # result            
        target_auto_format = self.auto_decimal(self.bayes_opt.max['target'])
        parmas_auto_format = {k: self.auto_decimal(v) for k, v in self.bayes_opt.max['params'].items()}
        self.repr_max = {'target':target_auto_format, 'params': parmas_auto_format}

        self.res = self.bayes_opt.res
        self.max = self.bayes_opt.max

    def __repr__(self):
        if len(self.repr_max) > 0:
            return f"(bayes_opt) BayesianOptimization: {self.repr_max}"
        else:
            return f"(bayes_opt) BayesianOptimization: undefined"
        



################################################################################################################
# print(get_python_lib())
class EstimatorSearch:
    """
     【required (Library)】 bayes_opt.BayesianOptimization, bayes_opt.UtilityFunction
     【required (Class)】 BayesOpt
     【required (Function)】auto_formating

    """
    def __init__(self, estimator, train_X=None, train_y=None, valid_X=None, valid_y=None, 
        params={}, params_dtypes={},
        optim='bayes', optimizer_params={}, optimize_params={},
        scoring=None, scoring_type='metrics', scoring_params={}, negative_scoring=False,
        verbose=0
        ):
        self.estimator = estimator

        self.train_X = train_X
        self.train_y = train_y

        self.valid_X = train_X if valid_X is None else valid_X
        self.valid_y = train_y if valid_y is None else valid_y

        self.params = params
        self.params_dtypes = params_dtypes
        if verbose>0 and len(params) > 0:
            print(f"fixed parmas: {params}")

        self.scoring = scoring
        self.scoring_type = scoring_type
        self.scoring_params = scoring_params
        self.negative_scoring = negative_scoring

        self.optim_method = optim
        self.optim = None
        self.optimizer_params = optimizer_params
        self.optimize_params = optimize_params

        self.verbose = verbose

        self.best_estimator = None

    def transform_dtype(self, dtype, x):
        if 'class' in str(dtype).lower():
            if 'int' in str(dtype).lower():
                return int(round(x,0))
            else:
                return dtype(x)
        elif type(dtype) == str:
            if 'int' in dtype.lower():
                return int(round(x,0))
            else:
                return eval(f"{dtype.lower()}({x})") 

    def gap_between_pred_true(self, true_y, pred_y):
        return np.sum((true_y - pred_y)**2) / len(true_y)

    def __call__(self, **params):
        if len(self.params_dtypes) > 0:
            apply_params = {k: (self.transform_dtype(self.params_dtypes[k], v) if k in self.params_dtypes.keys() else v) for k, v in params.items()}
        else:
            apply_params = params
        apply_params.update(self.params)
        
        # print(apply_params)
        model = self.estimator(**apply_params)
        model.fit(self.train_X, self.train_y)
        pred_y = model.predict(self.valid_X)
        score_result = self.score(y_true=self.valid_y, y_pred=pred_y, X=self.valid_X, y=self.valid_y, estimator=model, **self.scoring_params)
        
        if self.negative_scoring:
            return -score_result
        else:
            return score_result
    
    def optimizer(self, optim='bayes', verbose=None, **optimizer_params):
        """
         . bayesian_optimization : pbounds, random_state=None, verbose=2
        """
        verbose = self.verbose if verbose is None else verbose
        if len(optimizer_params) == 0:
            optimizer_params.update(self.optimizer_params)
        
        def optim_params_setting(params_dict, name, init):
            params_dict[name] = init if name not in optimizer_params.keys() else params_dict[name]
            return params_dict

        optim_method = self.optim_method if optim is None else optim
        if 'bayes' in optim_method:
            self.optim = BayesOpt(f=self.__call__, verbose=verbose, **optimizer_params)

        return self

    def optimize(self, **optimize_params):
        """
         . bayesian_optimization : init_points=5, n_iter=25, acq='ucb', kappa=2.576, xi=0.0, **gp_params
        """
        if len(optimize_params) == 0:
            optimize_params.update(self.optimize_params)

        if 'bayes' in self.optim_method:
            self.optim.maximize(**optimize_params)
            self.res = self.optim.res
            self.opt = self.optim.max

            if len(self.params_dtypes) > 0:
                self.res = [{'target': e['target'], 'params': {k: (self.transform_dtype(self.params_dtypes[k], v) if k in self.params_dtypes.keys() else v) for k, v in e['params'].items()}} for e in self.res]
                self.opt = {'target': self.opt['target'], 'params': {k: (self.transform_dtype(self.params_dtypes[k], v) if k in self.params_dtypes.keys() else v) for k, v in self.opt['params'].items()}}

        if self.best_estimator is not None:
            self.best_params = self.params.copy()
            self.best_params.update(self.opt['params'])
            self.best_estimator = self.estimator(**self.best_params)

            train_X_overall = pd.concat([self.train_X, self.valid_X], axis=0)
            train_y_overall = pd.concat([self.train_y, self.valid_y], axis=0)
            self.best_estimator.fit(train_X_overall, train_y_overall)
            print(f"(best_estimator is updated) result: {self.opt['target']}, best_params: {self.best_params}")

    def fit(self, train_X=None, train_y=None, valid_X=None, valid_y=None, scoring=None, scoring_type=None, negative_scoring=None, optim=None,
        verbose=1, optimizer_params={}, optimize_params={}, return_result=True):
        self.train_X = self.train_X if train_X is None else train_X
        self.train_y = self.train_y if train_y is None else train_y
        self.valid_X = (self.train_X if self.valid_X is None else self.valid_X) if valid_X is None else valid_X
        self.valid_y = (self.train_y if self.valid_y is None else self.valid_y) if valid_y is None else valid_y
        self.scoring = self.scoring if scoring is None else scoring
        self.scoring_type = self.scoring_type if scoring_type is None else scoring_type
        self.negative_scoring = self.negative_scoring if negative_scoring is None else negative_scoring
        optimizer_params = self.optimizer_params if len(optimizer_params) == 0 else optimizer_params
        optimize_params = self.optimize_params if len(optimizer_params) == 0 else optimize_params


        if optim is None:
            if self.optim is None:
                optim_method = 'bayes' if self.optim_method is None else self.optim_method
                self.optimizer(optim=optim_method, **optimizer_params).optimize(**optimize_params)
        else:
            optim_method = optim
            self.optimizer(optim=optim_method, **optimizer_params).optimize(**optimize_params)
        
        self.best_params = self.params.copy()
        self.best_params.update(self.opt['params'])

        self.best_estimator = self.estimator(**self.best_params)

        train_X_overall = pd.concat([self.train_X, self.valid_X], axis=0)
        train_y_overall = pd.concat([self.train_y, self.valid_y], axis=0)
        self.best_estimator.fit(train_X_overall, train_y_overall)

        if verbose:
            print(f"(Opimize) result: {self.opt['target']}, best_params: {self.best_params}")
        if return_result:
            return self.best_estimator

    def score(self, y_true=None, y_pred=None, X=None, y=None, estimator=None, **scoring_params):
        if 'metric' in self.scoring_type.lower() :
            if self.scoring is None:
                self.scoring = self.gap_between_pred_true
            result = self.scoring(y_true, y_pred)

        elif 'cross_val' in self.scoring_type.lower():
            result = np.mean(cross_val_score(estimator=estimator, X=X, y=y, scoring=self.scoring, **scoring_params))
        return result




################################################################################################################
# ['Mo', 'Ba', 'Cr', 'Sr', 'Pb', 'B', 'Mg', 'Ca', 'K']
class FeatureInfluence():
    """
    【required (Library)】 numpy, pandas
    【required (Class)】DataHandler, Mode
    【required (Function)】class_object_execution, auto_formating, dtypes_split

    < Input >

    < Output >
    
    """
    def __init__(self, train_X=None, estimator=None, n_points=5, encoder=None, encoderX=None, encoderY=None, conditions={}, y_name='pred_y', confidential_interval=None):
        self.estimator=estimator
        self.train_X = train_X
        self.n_points = n_points
        self.conditions = conditions

        self.encoder = encoder
        self.encoderX = encoderX
        self.encoderY = encoderY
        self.y_name = y_name

        self.DataHandler = DataHandler()
        self.grid_X = None
        
        self.confidential_interval = confidential_interval
        
    # define train, grid data handler instance
    def define_train_grid_data(self, train_X=None, grid_X=None, conditions={}, n_points=None):
        # train_data
        if train_X is not False:
            if train_X is None:
                if self.train_X is None:
                    raise Exception('train_X must be required to predict')
                else:
                    train_X = self.train_X

            self.train_X_info = self.DataHandler.data_info(train_X, save_data=False)
            if self.train_X_info.kind == 'pandas':
                self.train_X_info_split = self.DataHandler.data_info_split(train_X)
            else:
                self.train_X_info_split = self.DataHandler.data_info_split(train_X, columns=list(range(self.train_X.shape[1])))
        
        # grid_data
        if grid_X is not False:
            if grid_X is None:
                if self.grid_X is None:
                    grid_X = self.make_grid(train_X=train_X, conditions=conditions, n_points=n_points, return_result=True, save_result=False)
                else:
                    grid_X = self.grid_X

            self.grid_X_info = self.DataHandler.data_info(grid_X, save_data=False)
            if self.grid_X_info.kind == 'pandas':
                self.grid_X_info_split = self.DataHandler.data_info_split(grid_X)
            else:
                self.grid_X_info_split = self.DataHandler.data_info_split(grid_X, columns=list(range(self.grid_X.shape[1])))

    # checking unpredictable
    def predictable_check(self, target_data=None, criteria_data=None):
        target_frame = self.DataHandler.transform(target_data, apply_kind='pandas')
        criteria_frame = self.DataHandler.transform(criteria_data, apply_kind='pandas')
        
        numeric_columns = dtypes_split(target_frame, return_type='columns_list')['numeric']
        target_numeric = target_frame[numeric_columns]
        criteria_numeric = criteria_frame[numeric_columns]

        criteria_min = criteria_numeric.min()
        criteria_max = criteria_numeric.max()

        if target_numeric.shape == (1,0):
            return pd.DataFrame(columns=['unpred', 'unpred_columns', 'lower_columns', 'upper_columns'])
        else:
            unpredictable = pd.Series((target_numeric < criteria_min).any(axis=1) | (target_numeric > criteria_max).any(axis=1))
            unpredictable_cols_lower = pd.DataFrame(target_numeric < criteria_min).apply(lambda x: [i for i, xc in zip(x.index, x) if xc==True] ,axis=1)
            unpredictable_cols_upper = pd.DataFrame(target_numeric > criteria_max).apply(lambda x: [i for i, xc in zip(x.index, x) if xc==True] ,axis=1)
            unpredictable_cols = unpredictable_cols_lower + unpredictable_cols_upper

            unpredictable_result = pd.concat([unpredictable, unpredictable_cols, unpredictable_cols_lower, unpredictable_cols_upper], axis=1)
            unpredictable_result.columns = ['unpred', 'unpred_columns', 'lower_columns', 'upper_columns']
            return unpredictable_result

    # generate grid_table *
    def make_grid(self, train_X=None, conditions={}, n_points=None, save_result=True, return_result=False):
        re_num = re.compile('\d')

        # define train instance
        self.define_train_grid_data(train_X=train_X, grid_X=False)
        
        if n_points is None:
            n_points = self.n_points

        X_analysis = pd.DataFrame(**self.train_X_info_split.data).astype(self.train_X_info_split.dtypes)

        X_mean = X_analysis.apply(lambda x: x.mean() if 'int' in str(x.dtype) or 'float' in str(x.dtype) else x.value_counts().index[0] ,axis=0)
        X_std = X_analysis.apply(lambda x: x.std() if 'int' in str(x.dtype) or 'float' in str(x.dtype) else np.nan, axis=0)
        X_dict = X_mean.to_dict()
        dtypes = self.train_X_info_split.dtypes

        for xc in conditions:
            el = conditions[xc]
            if 'int' in str(dtypes[xc]) or 'float' in str(dtypes[xc]):
                if type(el) == str:
                    if 'monte' in el.lower():
                        if '(' in el:
                            monte_str, X_mean_el = el.split('(')
                            X_mean_el = int(X_mean_el.replace(')',''))
                        else:
                            monte_str = el
                            X_mean_el = X_mean[xc]
                        monte_n_list = re_num.findall(monte_str)
                        monte_n_points = int(''.join(monte_n_list)) if monte_n_list else n_points
                        el_list = np.random.randn(monte_n_points) * X_std[xc] + X_mean_el
                    elif '~' in el:
                        x_min, x_max = map(str, X_analysis[xc].agg(['min','max']))
                        el_split = el.split('~')
                        if len(el_split) == 2:
                            split_list = list(map(lambda x: str(x).strip().replace('min', x_min).replace('max',x_max), el_split))
                            if split_list[0] == '':
                                split_list[0] = x_min
                            if split_list[1] == '':
                                split_list[1] = x_max
                            el_list = np.linspace(*map(float, split_list), n_points)
                        elif len(el_split) == 1:
                            el_list = float(el_split[0].strip().replace('min', x_min).replace('max',x_max))
                elif 'int' in str(type(el)) or 'float' in str(type(el)):
                    el_list = conditions[xc]
                elif 'numpy' in str(type(el)):
                    el_list = conditions[xc]
            else:       # object dtype
                el_unique_vc = X_analysis[xc].value_counts()/len(X_analysis[xc])
                
                if type(el) == str:
                    if el.strip() in ['all', '~', 'min~max', 'min ~ max']:
                        el_list = list(el_unique_vc.index)
                    elif 'monte' in el.lower(): 
                        monte_str = el
                        monte_n_list = re_num.findall(monte_str)
                        monte_n_points = int(''.join(monte_n_list)) if monte_n_list else n_points
                        if monte_n_points > len(el_unique_vc):
                            monte_n_points = len(el_unique_vc)
                        el_list = list(np.random.choice(el_unique_vc.index, size=monte_n_points, replace=False, p=el_unique_vc))
                    else:
                        el_list = el
                elif type(el) == list:
                    el_list = el.copy()
                else:
                    el_list = el
            X_dict[xc] = el_list
            # break
        # X_dict 
        
        X_dict_array =  {k: v for k, v in X_dict.items() if type(v) == np.ndarray or type(v) == list}
        if len(X_dict_array) > 0:
            grid_X_frame_temp = pd.DataFrame(np.array(np.meshgrid(*X_dict_array.values())).reshape(len(X_dict_array),-1).T, columns=X_dict_array.keys())
                       
            for k, v in X_dict.items():
                if k not in X_dict_array.keys():
                    grid_X_frame_temp[k] = v
            grid_X_frame = grid_X_frame_temp[list(X_dict.keys())]
        else:
            grid_X_frame = pd.Series(X_dict).to_frame().T
        grid_X = self.DataHandler.transform(grid_X_frame, apply_kind=self.train_X_info_split.kind)
        
        for xc in grid_X:
            apply_dtype = 'float' if 'int' in str(dtypes[xc]) or 'float' in str(dtypes[xc]) else 'object'
            if apply_dtype == 'object':
                obj_type = type(X_analysis[xc].iloc[0])
                if 'str' in str(obj_type):
                    grid_X[xc] = grid_X[xc].apply(lambda x: str(x))
                if 'int' in str(obj_type):
                    grid_X[xc] = grid_X[xc].apply(lambda x: int(x))
                if 'float' in str(obj_type):
                    grid_X[xc] = grid_X[xc].apply(lambda x: float(x))
                if 'bool' in str(obj_type):
                    grid_X[xc] = grid_X[xc].apply(lambda x: bool(x))
                grid_X[xc] = grid_X[xc].astype(self.train_X_info_split.dtypes[xc])
            else:
                grid_X[xc] = grid_X[xc].astype(apply_dtype)

        if save_result:
            self.n_points = n_points
            self.conditions = conditions
            self.grid_X = grid_X

        if return_result:
            return grid_X
        else:
            return self

    # apply model to grid_table
    def grid_apply_model(self, grid_X=None, x=None, y=None, model='LinearRegression', train_X=None, conditions={}, n_points=None,
        inplace=True, return_result=False):
        
        # define train, grid instance
        self.define_train_grid_data(train_X=train_X, grid_X=grid_X, conditions=conditions, n_points=n_points)

        # grid, train frame
        grid_X_frame = pd.DataFrame(**self.grid_X_info_split.data).astype(self.grid_X_info_split.dtypes)
        train_X_frame = pd.DataFrame(**self.train_X_info_split.data).astype(self.train_X_info_split.dtypes)

        # x, y → list
        if x is not None:
            if type(x) == list:
                x_list = x.copy()
            else:
                x_list = [x]
        
        if y is not None:
            if type(y) == list:
                y_list = y.copy()
            else:
                y_list = [y]

        # grid_apply by model
        for yc in y_list:
            model_instance = class_object_execution(model)
            model_instance.fit(train_X_frame[x_list], train_X_frame[yc])
            grid_X_frame[yc] = model_instance.predict(grid_X_frame[x_list])

        result = grid_X_frame[y_list + x_list]

        # inplace
        if inplace is True:
            self.grid_X = self.DataHandler.transform(grid_X_frame, apply_kind=self.grid_X_info.kind)
        
        # return
        if return_result is True:
            return self.DataHandler.transform(result, apply_kind=self.grid_X_info.kind)
        else:
            return self

    # predict_from_grid(train_X)
    def predict_from_train_X(self, train_X=None, grid_X=None, conditions={}, n_points=None,
                    estimator=None, encoder=None, encoderX=None, encoderY=None):
        if grid_X is None:
            if self.grid_X is None:
                grid_X = self.make_grid(train_X=train_X, conditions=conditions, n_points=n_points, return_result=True, save_result=False)
            else:
                grid_X = self.grid_X
        
        # if estimator is None:
        #     estimator = self.estimator
        # if encoder is None:
        #     encoder = self.encoder
        # if encoderX is None:
        #     encoderX = self.encoderX
        # if encoderY is None:
        #     encoderY = self.encoderY
        grid_instance = self.DataHandler.data_info_split(grid_X)
        grid_frame = pd.DataFrame(**grid_instance.data).astype(grid_instance.dtypes)
        
        if encoder is not None:
            grid_frame_apply = encoder.transform(grid_frame)
        elif encoderX is not None:
            grid_frame_apply = encoderX.transform(grid_frame)
        else:
            grid_frame_apply = grid_frame.copy()

        pred_y_temp = estimator.predict(grid_frame_apply)
        pred_y_temp = self.DataHandler.transform(pred_y_temp, apply_kind='pandas', apply_index=grid_frame.index, apply_columns=[self.y_name])

        if encoderY is None and encoder is not None:
            encoderY = copy.deepcopy(encoder)
            for xc in self.train_X_info_split.data['columns']:
                del encoderY.encoder[xc]
            encoderY = list(encoderY.encoder.values())[0]

        if encoderY is not None:
            pred_y = self.DataHandler.transform(encoderY.inverse_transform(pred_y_temp), apply_kind=grid_instance.kind)
        else:
            pred_y = self.DataHandler.transform(pred_y_temp, apply_kind=grid_instance.kind)
        return {'grid_frame': grid_frame, 'pred_y':pred_y}

    # predict ***
    def predict(self, grid_X=None, train_X=None, conditions={}, n_points=None,
        estimator=None, encoder=None, encoderX=None, encoderY=None, y_name=None, 
        unpredictable=True,
        return_all=False):

        if train_X is None:
            train_X = self.train_X
        
        # grid_X auto_filling        
        if grid_X.shape[1] < train_X.shape[1] :
            train_mean = self.make_grid(train_X, save_result=False, return_result=True)
            grid_X_temp = pd.DataFrame()
            for i in train_mean:
                if i in grid_X.columns:
                    grid_X_temp[i] = grid_X[i]
                else:
                    grid_X_temp[i] = [train_mean[i].values[0]] * grid_X.shape[0]
            grid_X = grid_X_temp.copy()
        
        # define train instance
        self.define_train_grid_data(train_X=train_X, grid_X=grid_X, conditions=conditions, n_points=n_points)

        if estimator is None:
            if  self.estimator is None:
                raise Exception("estimator must be required.")
            else:
                estimator = self.estimator
        if encoder is None:
            encoder = self.encoder
        if encoderX is None:
            encoderX = self.encoderX
        if encoderY is None:
            encoderY = self.encoderY
        if y_name is None:
            if self.y_name is None:
                y_name = 'pred_y' 
            else:
                y_name = self.y_name

        result = self.predict_from_train_X(train_X=train_X, grid_X=grid_X, conditions=conditions, n_points=n_points,
            estimator=estimator, encoder=encoder, encoderX=encoderX, encoderY=encoderY)
        grid_frame = result['grid_frame']
        # return result
        pred_y = pd.Series(result['pred_y'], name=y_name, index=result['pred_y'].index)
        # pred_y = pd.Series(result[y_name], name=y_name, index=result[y_name].index)
    
        predictable_frame = self.predictable_check(target_data=grid_frame, criteria_data=train_X)
        unpredict = pd.Series()
        if len(predictable_frame) > 0:
            unpredict = pred_y[predictable_frame['unpred']]

        if unpredictable is True and len(unpredict) > 0:
            print(f"{list(unpredict.index)} index data is unpredictable date (from estimator).")
        if return_all is False:
            return self.DataHandler.transform(pred_y, apply_ndim=1, apply_kind=self.train_X_info_split.kind)
        elif return_all is True:
            if predictable_frame:
                return pd.concat([grid_frame, pred_y, predictable_frame], axis=1)
            else:
                return pd.concat([grid_frame, pred_y], axis=1)

    # plot one element
    def plot_element(self, x=None, x2=None, train_X=None, grid_X=None, conditions={}, estimator=None, 
        encoder=None, encoderX=None, encoderY=None, y_name=None,
        n_points=None, xlim=None, x2lim=None, ylim=None,
        figsize=None, title=None, contour=True, display_unpredictable=True, text_points = 7,
        decimals = None,
        return_plot=True):

        if train_X is None:
            train_X = self.train_X

        # define train instance
        self.define_train_grid_data(train_X=train_X, grid_X=False)


        # estimator, encoderX, encoderY, y_name
        if estimator is None:
            if  self.estimator is None:
                raise Exception("estimator must be required.")
        else:
            self.estimator = estimator
        if encoder is not None:
            self.encoder = encoder
        if encoderX is not None:
            self.encoderX = encoderX
        if encoderY is not None:
            self.encoderY = encoderY
        if y_name is None:
            if self.y_name is None:
                y_name = 'pred_y' 
            else:
                y_name = self.y_name
        
        if len(conditions) == 0:
            conditions = self.conditions
        
        if x2 is None:
            apply_condition = {x:'min~max'}
            apply_condition.update(dict(filter(lambda el: el[0]==x, conditions.items())))
        else:
            apply_condition = {x:'min~max', x2:'min~max'}
            apply_condition.update(dict(filter(lambda el: el[0]==x, conditions.items())))
            apply_condition.update(dict(filter(lambda el: el[0]==x2, conditions.items())))

        # data_set
        pred_result = self.predict_from_train_X(train_X=train_X, conditions=apply_condition, n_points=n_points,
                    estimator=self.estimator, encoder=self.encoder, encoderX=self.encoderX, encoderY=self.encoderY)

        grid_frame = pred_result['grid_frame']
        pred_y = pd.Series(pred_result['pred_y'], name=y_name)
        predictable_frame = self.predictable_check(target_data=grid_frame, criteria_data=train_X)

        plot_table_1D = pd.concat([grid_frame, pred_y, predictable_frame], axis=1)
        plot_table_lower_1D = plot_table_1D[plot_table_1D['lower_columns'].apply(lambda k: False if len(k) == 0 else True)]
        plot_table_upper_1D = plot_table_1D[plot_table_1D['upper_columns'].apply(lambda k: False if len(k) == 0 else True)]

        outlier_colors = {True:'orange', False:'steelblue'}

        if return_plot:
            f = plt.figure()
        
        # plot
        if x2 is None:       # 1D
            if title is not None:
                plt.title(title)
            else:
                plt.title(f"{y_name} by {x} (y_range: {auto_formating(pred_y.max() - pred_y.min())})")
            
            # plot_scatter
            if display_unpredictable:
                for gi, gv in plot_table_1D.groupby('unpred'):
                    plt.scatter(gv[x], gv[y_name], edgecolor='white', alpha=0.3, color=outlier_colors[gi])
            else:
                plt.scatter(plot_table_1D[x], plot_table_1D[y_name], edgecolor='white', alpha=0.3)
            
            # plot_line
            self.plot_data = plot_table_1D[[x, y_name]]
            plt.plot(plot_table_1D.groupby(x)[y_name].mean(), alpha=0.7)
            
            text_points_list = [int(i) for i in np.linspace(0, len(plot_table_1D)-1, text_points)]
            for pei, (pti, ptd) in enumerate(plot_table_1D.iterrows()):
                if pei in text_points_list:
                    plt.text(ptd[x], ptd[y_name], auto_formating(ptd[y_name]))
            
            if self.confidential_interval is not None:
                plt.fill_between(plot_table_1D[x], plot_table_1D[y_name]+self.confidential_interval, plot_table_1D[y_name]-self.confidential_interval,
                                 alpha=0.1, facecolor='green')
            
            if ylim is not None:
                plt.ylim(ylim[0], ylim[1])
            if display_unpredictable:
                if len(plot_table_lower_1D) > 0:
                    plt.plot(plot_table_lower_1D.groupby(x)[y_name].mean(), color='orange')
                if len(plot_table_upper_1D) > 0:
                    plt.plot(plot_table_upper_1D.groupby(x)[y_name].mean(), color='orange')
            plt.xlabel(x)
            plt.ylabel(y_name)
            
        else:
            plot_table_2D = plot_table_1D.groupby([x2,x])[y_name].mean().unstack(x).sort_index(ascending=False)
            self.trainX = train_X
            self.apply_condition = apply_condition
            self.plot_data = plot_table_2D
            
            if title is not None:
                plt.title(title)
            else:
                plt.title(f"{y_name} by {x}~{x2} (y_range: {auto_formating(plot_table_2D.max().max() - plot_table_2D.min().min())})")
            
            vmin = None if ylim is None else ylim[0]
            vmax = None if ylim is None else ylim[1]
            CTMap = plt.contourf(plot_table_2D.columns, plot_table_2D.index, plot_table_2D, cmap='jet', vmin=vmin, vmax=vmax)
            # plt.contour(plot_table_2D.columns, plot_table_2D.index, plot_table_2D, cmap='jet', vmin=vmin, vmax=vmax)
            
            if decimals is None:
                decimals = 0 if -np.log10(plot_table_2D.mean().mean())+1 < 0 else int(-np.log10(plot_table_2D.mean().mean())+1)
            plt.clabel(CTMap, inline=True, colors ='grey', fmt=f'%.{decimals}f', fontsize=15)
            CTbar = plt.colorbar(extend = 'both')
            CTbar.set_label(y_name)
            plt.xlabel(x)
            plt.ylabel(x2)
            
            if x2lim is not None:
                plt.ylim(x2lim[0], x2lim[1])
        
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
            
        if return_plot:
            plt.close()
            self.plot = f
            return self.plot

        
        # return pd.concat([plot_table_1D, plot_table_lower_1D, plot_table_upper_1D])
        
        # pass

    # influence_summary ***
    def influence_summary(self, train_X=None, conditions={}, n_points=None, grid_X=None,
        estimator=None, encoder=None, encoderX=None, encoderY=None, y_name=None, 
        feature_importances=True, summary_plot=True, summary_table=True, sort=False):
        '''
         . *train_X : training Dataset
         . *estimator : model
         . (if need) encoder : apply both X and y variables.
         . (if need) encoderX : apply only X variables.
         . (if need) encoderY : apply only y variable.
        '''
        if train_X is None:
            if self.train_X is not None:
                train_X = self.train_X
        # define train instance
        self.define_train_grid_data(train_X=train_X, grid_X=False)

        # define train, grid instance
        conditions_dict = {c:'min~max' for c in self.train_X_info_split.data['columns']}
        if conditions:
            conditions_dict.update(conditions)

        # estimator, encoderX, encoderY, y_name
        if estimator is None:
            if  self.estimator is None:
                raise Exception("estimator must be required.")
            else:
                estimator = self.estimator
        if encoder is None:
            encoder = self.encoder
        if encoderX is None:
            encoderX = self.encoderX
        if encoderY is None:
            encoderY = self.encoderY
        if y_name is None:
            if self.y_name is None:
                y_name = 'pred_y' 
            else:
                y_name = self.y_name

        if n_points is None:
            n_points = self.n_points

        # influence analysis
        self.influence_dict_all = {}
        self.influence_dict = {}
        feature_influence = {}

        dtypes = self.train_X_info_split.dtypes

        y_min_total = np.inf
        y_max_total = -np.inf
        for ic in conditions_dict:
            grid_X = self.make_grid(train_X=train_X, conditions={ic: conditions_dict[ic]}, n_points=n_points, return_result=True, save_result=False)
            pred_result = self.predict_from_train_X(train_X=train_X, grid_X=grid_X, conditions={ic: conditions_dict[ic]}, n_points=n_points,
                    estimator=estimator, encoder=encoder, encoderX=encoderX, encoderY=encoderY)
            
            grid_temp_frame = pred_result['grid_frame']
            grid_temp_frame[y_name] = pred_result['pred_y']
            self.influence_dict_all[ic] = self.DataHandler.transform(grid_temp_frame, apply_kind=self.train_X_info.kind)
            self.influence_dict[ic] = self.DataHandler.transform(grid_temp_frame[[ic]+[y_name]], apply_kind=self.train_X_info.kind)

            # summary_plot
            y_argmin_idx = grid_temp_frame[y_name].argmin()
            y_argmax_idx = grid_temp_frame[y_name].argmax()
            x_argmin = grid_temp_frame[ic].iloc[y_argmin_idx]
            x_argmax = grid_temp_frame[ic].iloc[y_argmax_idx]
            x_min = grid_temp_frame[ic].min()
            x_max = grid_temp_frame[ic].max()
            y_min = grid_temp_frame[y_name].min()
            y_max = grid_temp_frame[y_name].max()
            
            if 'int' in str(train_X[ic].dtype).lower() or 'float' in str(train_X[ic].dtype).lower():
                feature_influence[ic] = {'delta_y': y_max - y_min, 'delta_x_ymax': x_argmax - x_argmin,
                        'delta_x': x_max - x_min, 'max_slope': (y_max - y_min) / (x_argmax - x_argmin)}
            else:
                feature_influence[ic] = {'delta_y': y_max - y_min, 'delta_x_ymax': f"{x_argmin} ~ {x_argmax}",
                        'delta_x': np.nan, 'max_slope': np.nan}
            feature_influence[ic] = {k: auto_formating(v, return_type='str') for k, v in feature_influence[ic].items()}
            feature_influence[ic]['y_range'] = auto_formating(y_min, return_type='str') + ' ~ ' + auto_formating(y_max, return_type='str')
            feature_influence[ic]['x_ymax_range'] = auto_formating(x_argmin, return_type='str') + ' ~ ' + auto_formating(x_argmax, return_type='str')
            feature_influence[ic]['x_range'] = auto_formating(x_min, return_type='str') + ' ~ ' + auto_formating(x_max, return_type='str')

            y_min_total = min(y_min_total, y_min)
            y_max_total = max(y_max_total, y_max)
             
        # feature_plot
        self.feature_plot = {}
        for ic in conditions_dict:
            f = plt.figure()
            self.plot_element(x=ic, train_X=train_X, grid_X=self.influence_dict_all[ic], conditions={ic:'min~max'},
                estimator=estimator, encoder=encoder, encoderX=encoderX, encoderY=encoderY, y_name=y_name, return_plot=False)
            plt.ylim(y_min_total*0.95, y_max_total*1.05)
            plt.close()
            self.feature_plot[ic] = f
        
        # feature_influence (table)
        self.summary_table = pd.DataFrame(feature_influence).T
        self.summary_table['plot'] = pd.Series(self.feature_plot)
        
        if feature_importances:
            print(f'==== < Feature Importances Plot > ====')
            print(f' → self.feature_importances_plot')

        self.feature_importances_plot = plt.figure(figsize=(5, self.train_X.shape[1]*0.13+2) )
        plt.barh(self.summary_table.index[::-1], self.summary_table['delta_y'].apply(lambda x: x.replace(',','')).astype('float')[::-1])
        
        if feature_importances:
            plt.show()
        else:
            plt.close()
        

        # summary_plot
        if summary_plot:
            print(f'==== < Feature Influence Summary Plot > ====')
            print(f' → self.summary_plot')
        
        ncols = len(conditions_dict)
        fig_ncols = 4 if ncols > 4 else ncols
        fig_nrows = ((ncols // 4)+1) if ncols > 4 else 0.4

        fig = plt.figure(figsize=(fig_ncols * 4, fig_nrows * 4))
        fig.subplots_adjust(hspace=0.5)   # 위아래, 상하좌우 간격

        for idx, ic in enumerate(conditions_dict, 1):
            plt.subplot(int(fig_nrows)+1, fig_ncols, idx)
            self.plot_element(x=ic, train_X=train_X, grid_X=self.influence_dict_all[ic], conditions={ic:'min~max'},
                estimator=estimator, encoder=encoder, encoderX=encoderX, encoderY=encoderY, y_name=y_name, return_plot=False)
            plt.ylim(y_min_total*0.95, y_max_total*1.05)
        if summary_plot:
            plt.show()
        else:
            plt.close()
        self.summary_plot = fig

        if sort:
            self.summary_table = self.summary_table.sort_values('delta_y', ascending=False)

        print(f'==== < Feature Influence Summary Table > ====')
        print(f' → self.summary_table')
        if summary_table:
            print_DataFrame(self.summary_table)



# fi = FeatureInfluence(estimator=XGR, n_points=50, encoder=se_encoder, train_X=train_set[X_cols], y_name='Fare')
# # fi.make_grid(conditions={'Pclass':'~','Embarked':'monte2'}, return_result=True)
# fi.predict(test_set[X_cols], return_all=True)

# fi.plot_element(x='Survived')





################################################################################################################
class BestEstimatorSearch:
    """
     【required (Library)】numpy, pandas, bayes_opt.BayesianOptimization, bayes_opt.UtilityFunction, 
                           copy.deepcopy, itertools.combinations
     【required (Class)】 BayesOpt, EstimatorSearch, ModelEvaluate
     【required (Function)】 auto_formating, print_DataFrame


     < Result Guidance by method>
      (init) estimators, metrics, scoring_option, train_X, train_y, valid_X, valid_y, test_X, test_y

      (fit) train_X, train_y, valid_X, valid_y, test_X, test_y, verbose
        → self.estimators

      (emsemble) weights, sorting, verbose
        → self.ensemble_result
        → self.best_ensemble
        → self.best_ensemble_estimator

      (ensemble_summary) best_estimator, n_points, encoder, encoderX, encoderY, verbose
        → self.summary_table
        → self.summary_plot
    """
    
    def __init__(self, estimators={'LS':('linear', Lasso, {'random_state':0}, {'optimizer_params':{'pbounds': {'alpha':(0.0001,100)}, 'random_state':0}} )}, 
        metrics={'r2_adj': 'r2_adj', 'rmse': 'rmse'}, 
        scoring_option={'scoring':'neg_mean_squared_error', 'scoring_type':'cross_val_score'},
        train_X=None, train_y=None, valid_X=None, valid_y=None, test_X=None, test_y=None):
        """
        estimators = {'estimator_name':('linear', 'estimator', 'parmas', 'optimizer_params') ...}
        metrics = ['metric1', 'metric2']
        """
        self.estimators_params = estimators
        self.metrics = metrics
        self.scoring_option = scoring_option

        self.train_X = train_X
        self.train_y = train_y
        self.valid_X = valid_X
        self.valid_y = valid_y
        self.test_X = test_X
        self.test_y = test_y

        self.estimators = None
        self.ensemble_result = None
        self.feature_influence = None

    def fit(self, train_X=None, train_y=None, valid_X=None, valid_y=None, test_X=None, test_y=None, verbose=1):
        self.train_X = self.train_X if train_X is None else train_X
        self.train_y = self.train_y if train_y is None else train_y
        self.valid_X = (self.train_X if self.valid_X is None else self.valid_X) if valid_X is None else valid_X
        self.valid_y = (self.train_y if self.valid_y is None else self.valid_y) if valid_y is None else valid_y
        self.test_X = (self.train_X if self.test_X is None else self.test_X) if test_X is None else test_X
        self.test_y = (self.train_y if self.test_y is None else self.test_y) if test_y is None else test_y

        models = {}
        for en, ev in self.estimators_params.items():
            models[en] = {}
            if verbose > 0:
                print(f"【 {en} model fitting 】", end=' ')

            if 'verbose' not in ev[3].keys():
                ev[3].update({'verbose':0})
            
            # optimize model
            ms_otim = EstimatorSearch(estimator=ev[1], params=ev[2], **ev[3], **self.scoring_option)
            ms_otim.fit(train_X=self.train_X, train_y=self.train_y, valid_X=self.valid_X, valid_y=self.valid_y, verbose=0, return_result=False)
            
            pred_y = ms_otim.best_estimator.predict(self.test_X)

            # metric model
            ms_me = ModelEvaluate(self.test_X, self.test_y, model=ms_otim.best_estimator, verbose=0)
            ms_metric = {}
            for mk, mv in self.metrics.items():
                try:
                    ms_metric[mk] = eval(f'ms_me.{mv}')
                except:
                    ms_metric[mk] = mv(self.test_y, pred_y)
            ms_metric_str = ', '.join([f"{k}: {auto_formating(v)}" for k,v in ms_metric.items()])

            # plotting model
            ms_plot = plt.figure(figsize=(5, self.train_X.shape[1]*0.13+2))
            plt.title(f"{en}\n{ms_metric_str}")
            if 'linear' in ev[0]:
                pd.Series(ms_otim.best_estimator.coef_, index=self.train_X.columns).sort_values().plot.barh()
            elif ('ensemble' in ev[0]) and ('tree' in ev[0]):
                pd.Series(ms_otim.best_estimator.feature_importances_, index=self.train_X.columns).sort_values().plot.barh()
            plt.close()

            models[en]['estimator'] = copy.deepcopy(ms_otim.best_estimator)
            models[en]['evaluate'] = dict(ms_me.metrics._asdict())
            models[en]['metric'] = ms_metric
            models[en]['plot'] = ms_plot
            if verbose > 0:
                print(f"   (estimator) {ms_otim.best_estimator}\n   (metric) {ms_metric_str} ***")
        
        self.estimators = models
        if verbose > 0:
            print('='*100)
            print('done. → (result) self.estimators')
        
        return self

    def ensemble(self, train_X=None, train_y=None, valid_X=None, valid_y=None, test_X=None, test_y=None,
        estimators=None, weights=None, sorting='auto', verbose=1):

        self.train_X = self.train_X if train_X is None else train_X
        self.train_y = self.train_y if train_y is None else train_y
        self.valid_X = (self.train_X if self.valid_X is None else self.valid_X) if valid_X is None else valid_X
        self.valid_y = (self.train_y if self.valid_y is None else self.valid_y) if valid_y is None else valid_y
        self.test_X = (self.train_X if self.test_X is None else self.test_X) if test_X is None else test_X
        self.test_y = (self.train_y if self.test_y is None else self.test_y) if test_y is None else test_y

        if self.estimators is None:
            self.fit(verbose=0)

        models = {en: ev['estimator'] for en, ev in self.estimators.items()} if estimators is None else estimators
        weights = {en: 1/ev['evaluate']['rmse'] for en, ev in self.estimators.items()} if weights is None else weights

        # models combinations
        estimators = {}
        estimators_weights = {}       

        count = len(models) + 1
        for n in range(2, len(models)+1):
            comb_mdl_name = list(combinations(models.keys(), n))
            comb_mdl_models = list(combinations(models.values(), n))
            comb_mdl_weights = list(combinations(weights.values(), n))

            for name, model, weight in zip(comb_mdl_name, comb_mdl_models, comb_mdl_weights):
                estimators[f'M{count:03}'] = [(mn, mm) for mn, mm in zip(name, model)]
                estimators_weights[f'M{count:03}'] = list(weight)
                count += 1
        estimators_comb_series = pd.Series({k: [e[0] for e in v] for k, v in estimators.items()}, name='estimators_comb')

        # basic estimators
        basic_estimator_idxs = [f"M{n:03}" for n in np.arange(1, len(models)+1)]
        basic_summary_dict = {}
        for e, (en, mdl) in zip(basic_estimator_idxs, models.items()):
            if verbose > 0:
                print(f"< {e} : {en} >")

            basic_me = ModelEvaluate(self.test_X, self.test_y, model=mdl, verbose=verbose)
            basic_metric = dict(basic_me.metrics._asdict())

            basic_summary_dict[e] = {'estimators_comb':en, **basic_metric, 'ensemble_estimators':mdl}
        basic_score_frame = pd.DataFrame(basic_summary_dict).T
        
        # votting
        votting_models = {}
        votting_socres = {}
        for i, (en, e) in enumerate(zip(estimators_comb_series, estimators)):
            if verbose > 0:
                print(f"< {e} : {', '.join(en)} >")
            
            VR = VotingRegressor(estimators=estimators[e], weights=estimators_weights[e])
            VR.fit(self.train_X, self.train_y)
            pred_y = VR.predict(self.train_X)
            
            VR_me = ModelEvaluate(self.test_X, self.test_y, model=VR, verbose=verbose)

            # votting_metric = {}
            # for mk, mv in self.metrics.items():
            #     try:
            #         votting_metric[mk] = eval(f'VR_me.{mv}')
            #     except:
            #         votting_metric[mk] = mv(self.test_y, pred_y)
            votting_metric = dict(VR_me.metrics._asdict())

            votting_models[e] = copy.deepcopy(VR)
            votting_socres[e] = votting_metric

        estimators_series = pd.Series(votting_models, name='ensemble_estimators')

        votting_scores_frame = pd.DataFrame(votting_socres)
        votting_scores_frame = pd.concat([estimators_comb_series.to_frame().T, votting_scores_frame, estimators_series.to_frame().T], axis=0)
        votting_scores_frame = votting_scores_frame.T

        # concat result (basics ↔ combinations)
        votting_scores_frame = pd.concat([basic_score_frame, votting_scores_frame], axis=0)

        # sorting
        if sorting == 'auto' or sorting is None:
            ensemble_sort = votting_scores_frame.copy()
            ensemble_sort.insert(ensemble_sort.shape[1]-1, 'ensemble_score', ensemble_sort['rmse'] * (1-ensemble_sort['r2_score']) * (1-ensemble_sort['mape']), True)
            # ensemble_sort['ensemble_score'] = ensemble_sort['rmse'] * (1-ensemble_sort['r2_score'])
            ensemble_sort.sort_values('ensemble_score', axis=0, inplace=True)
            
            if sorting == 'auto':
                votting_scores_frame = ensemble_sort.copy()
        else:
            ensemble_sort = votting_scores_frame.copy()
            ensemble_sort.sort_values(sorting, axis=0, inplace=True)
            votting_scores_frame = ensemble_sort.copy()

        self.ensemble_result = votting_scores_frame
        self.best_ensemble = ensemble_sort.iloc[0, :]
        self.best_ensemble_estimator = self.best_ensemble['ensemble_estimators']

        if verbose > 0:
            print()
            print(f"done.")
            print(f"→ (ensemble_result) self.ensemble_result")
            print(f"  (best_ensemble) self.best_ensemble")
            print(f"  (best_ensemble_estimator) self.best_ensemble_estimator")
            print_DataFrame(ensemble_sort)
        
        return self

    def ensemble_summary(self, train_X=None, train_y=None, valid_X=None, valid_y=None, test_X=None, test_y=None,
            best_estimator=None, n_points=50,
            encoder=None, encoderX=None, encoderY=None, verbose=1):
        
        self.train_X = self.train_X if train_X is None else train_X
        self.train_y = self.train_y if train_y is None else train_y
        self.valid_X = (self.train_X if self.valid_X is None else self.valid_X) if valid_X is None else valid_X
        self.valid_y = (self.train_y if self.valid_y is None else self.valid_y) if valid_y is None else valid_y
        self.test_X = (self.train_X if self.test_X is None else self.test_X) if test_X is None else test_X
        self.test_y = (self.train_y if self.test_y is None else self.test_y) if test_y is None else test_y

        if self.estimators is None:
            self.fit(verbose=0)
        if self.ensemble_result is None:
            self.ensemble(verbose=0)

        if encoderX is not None:
            train_X = encoderX.inverse_transform(self.train_X)
        elif encoder is not None:
            train_X = encoderX.inverse_transform(self.train_X)
        else:
            train_X = self.train_X
        
        if best_estimator is None:
            best_estimator = self.best_ensemble_estimator

        feature_influence = FeatureInfluence(train_X=train_X, estimator=best_estimator, 
                    encoder=encoder, encoderX=encoderX, encoderY=encoderY,
                    n_points=n_points)

        summary_table = True if verbose > 0 else False
        summary_plot = True if verbose > 0 else False
        feature_influence.influence_summary(summary_table=summary_table, summary_plot=summary_plot)

        self.feature_influence = feature_influence
        self.summary_table = feature_influence.summary_table
        self.summary_plot = feature_influence.summary_plot

        if verbose > 0:
            print()
            print(f"done.")
            print(f"  (summary_table) self.summary_table")
            print(f"  (summary_plots) self.summary_plot")


################################################################################################################




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
import seaborn as sns
# import cvxpy
import scipy.stats as stats


import copy
from collections import namedtuple

# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import *

# import sys
# sys.path.append(r'D:\작업방\업무 - 자동차 ★★★\Worksapce_Python\DS_Module')
# sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

# absolute_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 4) Aritificial_Intelligent/교재_실습_자료/'
# absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'




# ModelEvaluate
class ModelEvaluate():
    '''
    < input >
    X : DataFrame or 2-Dim matrix
    y : Series
    model : sklearnModel
    '''
    def __init__(self, X, y, model, model_type=None, verbose=1):
        model_name = str(type(model)).lower()
        X_frame = pd.DataFrame(X)
        y_true = np.array(y).ravel()
        y_pred = np.array(model.predict(X)).ravel()
        n_data, dof = X.shape

        if model_type is None:
            if ('regress' in model_name) or ('lasso' in model_name) or ('ridge' in model_name) or ('elasticnet' in model_name):
                model_type = 'regressor'
            if 'classi' in model_name:
                model_type = 'classifier'

        if model_type == 'regressor':
            #### sum_square ****
            sum_square_instance = namedtuple('sum_square', ['sst', 'ssr', 'sse'])
            self.sst = sum((y_true-y_true.mean())**2)
            self.ssr = sum((y_true.mean() - y_pred)**2)
            self.sse = sum((y_true - y_pred)**2)            

            sum_square_list = [self.sst, self.ssr, self.sse]
            self.sum_square = sum_square_instance(*[self.auto_decimal(m) for m in sum_square_list]) 

            #### metrics ****
            metrics_instance = namedtuple('metrics', ['r2_score', 'r2_adj', 'mse', 'rmse', 'mae', 'mape'])

            self.r2_score = 1 - self.sse/self.sst
            # self.r2_adj = 1 - ((n_data-1) * self.ssr/self.sst) / (n_data - dof - 1)
            self.r2_adj = 1 - ((n_data-1) * self.sse/self.sst) / (n_data - dof - 1)
            

            self.mse = self.sse/(n_data-2)
            self.rmse = np.sqrt(self.mse)
            self.mae = sum(np.abs(y_true - y_pred)) / n_data
            mape_series = pd.Series(1 - np.abs( (y_true - y_pred) / y_true ))
            mape_series = mape_series[~((mape_series == np.inf) | (mape_series == -np.inf))]
            self.mape = mape_series.mean()
            
            metrics_list = [self.r2_score, self.r2_adj, self.mse, self.rmse, self.mae, self.mape]
            self.metrics = metrics_instance(*[self.auto_decimal(m) for m in metrics_list]) 

            if verbose > 0:
                print(' .', self.sum_square) 
                print(' .', self.metrics) 

            #### hypothesis ****
            try:
                if sum([i.lower() in model_name for i in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']]):
                    hypothesis_instance = namedtuple('hypothesis', ['tvalues', 'pvalues'])

                    params = np.array([model.intercept_] + list(model.coef_))
                    newX = pd.DataFrame({"Constant": np.ones(n_data)}, index=X_frame.index).join(X_frame)
                    std_b = np.sqrt(self.mse*(np.linalg.inv(np.dot(newX.T,newX)).diagonal()))
                    t = params/ std_b               
                    p = 2 * (1 - stats.t.cdf(np.abs(t), n_data - dof))

                    try:
                        X_names = ['const'] + list(X.columns)
                    except:
                        X_names = ['const'] + ['x' + str(i+1) for i in np.arange(dof)]
                    try:
                        y_name = y.name
                    except:
                        y_name = 'Y'


                    self.tvalues = {k: self.auto_decimal(v) for k, v in zip(X_names, t)}
                    self.pvalues = {k: self.auto_decimal(v) for k, v in zip(X_names, p)}

                    self.hypothesis = hypothesis_instance(self.tvalues, self.pvalues)

                    #### linear ****
                    linear_instance = namedtuple('linear', ['coef', 'formula'])
                    self.coef = {k: self.auto_decimal(v) for k, v in zip(X_names, params)}
                    self.formula = y_name + ' = ' + ''.join([ f'{str(v)}·{k}' if i == 0 else (' + ' if v > 0 else ' - ') + (str(abs(v)) if k=='const' else f'{abs(v)}·{str(k)}') for i, (k, v) in enumerate(self.coef.items())])

                    self.linear = linear_instance(self.coef, self.formula)
                    if verbose > 0:
                        print(' .', self.hypothesis) 
                        print(' .', self.linear)
            except:
                pass

        elif model_type == 'classifier':
            pass

    def decimal(self, x, rev=0):
        return 2 if x == 0 else int(-1*(np.floor(np.log10(abs(x)))-3-rev))

    def auto_decimal(self, x, rev=0):
        if pd.isna(x):
            return np.nan
        else:
            return round(x, self.decimal(x, rev=rev))



# # class FeatureInfluence
# class FeatureInfluence():
#     def __init__(self, estimator, scaler, n_points=50, X_range=None, X_mean=None, y_name=None):
#         self.estimator = estimator
#         self.scaler = scaler
#         self.n_points = n_points
#         self.X_range = X_range
#         self.X_mean = X_mean
#         self.y_name = y_name
    
#     def fit(self, X, y, on=None, plot=False):
#         estimator = self.estimator
#         scaler = self.scaler
#         n_points = self.n_points
#         X_range = self.X_range
#         X_mean = self.X_mean
#         y_name = self.y_name

#         if on is None:
#             features = X.columns.to_list()
#         else:
#             features = on.copy()

#         result = {}
#         plot_temper = {}
#         for x in features:
#             # make mean_X DataFrame
#             if X_mean is None:
#                 data_mean = pd.concat([X.mean().to_frame().T]*n_points, ignore_index=True)
#             else:
#                 data_mean = pd.concat([X_mean.to_frame().T]*n_points, ignore_index=True)
#                 data_mean = pd.DataFrame(scaler.transform(data_mean), columns=data_mean.columns)

#             if X_range is None:
#                 x_min, x_max = X[x].agg(['max','min'])
#                 x_min_max = np.linspace(x_max, x_min, num=n_points)
#             else:
#                 x_min_max = np.linspace(X_range[1], X_range[0], num=n_points)
#             data_mean[x] = x_min_max

#             x_column_idx = np.argwhere(data_mean.columns == x)[0][0]

#             data_mean.T.to_clipboard()
#             # evaluate
#             x_vector = scaler.inverse_transform(data_mean)[:, x_column_idx]
#             predict_vector = estimator.predict(data_mean)
#             predict_summary = pd.Series(predict_vector).agg(['mean', 'std', 'min', 'max'])
#             predict_decimal = fun_Decimalpoint(predict_summary['mean'])
#             predict_range = predict_summary['max'] - predict_summary['min']

#             predict_result = predict_summary.to_dict().copy()
#             predict_result['range'] = predict_range

#             if y_name is None:
#                 try:
#                     y_name = y.name
#                 except:
#                     y_name = 'Target'
            
#             fig = plt.figure()
#             plt.title(f"Influence of {y_name} by {x}\nstd: {round(predict_summary['std'], predict_decimal+1)},  range: {round(predict_range, predict_decimal+1)}")
#             plt.plot(x_vector, predict_vector, 'o-', markerfacecolor=(70/255,130/255,180/255,0.3), markersize=5)
#             predict_result['plot'] = fig
#             plt.close()
#             result[x] = predict_result

#             plot_temper[x] = {}
#             plot_temper[x]['predict_vector'] = predict_vector
#             plot_temper[x]['x_vector'] = x_vector

#         feature_influence = pd.DataFrame(result).T
#         self.feature_influence = feature_influence[['range', 'std', 'min', 'max']].applymap(lambda x: np.round(x, fun_Decimalpoint(x)))
#         self.feature_influence['plot'] = feature_influence['plot']
#         self.feature_influence.sort_values('range', ascending=False, inplace=True)
        
#         self.predict_frame = pd.DataFrame(plot_temper).T
#         # self.feature_influence = pd.concat([self.feature_influence, pd.DataFrame(plot_temper).T], axis=1 )

#         if plot is not False:
#             print(f'==== < Feature Influence Plot > ====')
#             print(f' → self.influence_plot')
#             ncols = len(self.feature_influence.index)
#             fig_ncols = 4 if ncols > 4 else ncols
#             fig_nrows = ((ncols // 4)+1) if ncols > 4 else 0.4

#             fig = plt.figure(figsize=(fig_ncols * 4, fig_nrows * 4))
#             fig.subplots_adjust(hspace=0.5)   # 위아래, 상하좌우 간격

#             for idx, x in enumerate(self.feature_influence.index, 1):
#                 plt.subplot(int(fig_nrows)+1, fig_ncols, idx)
#                 plt.title(f"Influence of {y_name} by {x}\nstd: {self.feature_influence.loc[x, 'std']},  range: {self.feature_influence.loc[x, 'range']}")
#                 plt.plot(plot_temper[x]['x_vector'], plot_temper[x]['predict_vector'], 'o-', markerfacecolor=(70/255,130/255,180/255,0.3), markersize=5)
#             plt.show()
#             self.influence_plot = fig

#         print(f'==== < Summary Feature Influence > ====')
#         print(f' → self.feature_influence')
#         print_DataFrame(self.feature_influence)
#         # return self.feature_influence

#     def compare(self, normal_data, abnormal_data, plot=False):
#         normal_predict = self.estimator.predict(self.scaler.transform(normal_data.to_frame().T))[0]
#         abnormal_predict = self.estimator.predict(self.scaler.transform(abnormal_data.to_frame().T))[0]
#         def calc_predict_from_predict_table(data, x):
#             idx_vector = np.argwhere(self.predict_frame['x_vector'][x] < float(data[x]))
#             if len(idx_vector):
#                 idx = np.max(idx_vector)
#             else:
#                 idx = 0

#             if len(idx_vector):
#                 try:
#                     y_grad = (self.predict_frame['predict_vector'][x][idx+1] - self.predict_frame['predict_vector'][x][idx])
#                     x_grad = (self.predict_frame['x_vector'][x][idx+1] - self.predict_frame['x_vector'][x][idx])
#                     predict = y_grad/x_grad * (self.predict_frame['x_vector'][x][idx+1] - float(data[x])) + self.predict_frame['predict_vector'][x][idx]
#                 except:
#                     predict = self.predict_frame['predict_vector'][x][-1]
#             else:
#                 predict = self.predict_frame['predict_vector'][x][0]
                
#             return predict

#         result = {}
#         for x in self.predict_frame.index:
#             normal_x_predict = calc_predict_from_predict_table(data=normal_data, x=x)
#             abnormal_x_predict = calc_predict_from_predict_table(data=abnormal_data, x=x)
#             result[x] = abnormal_x_predict - normal_x_predict

#         self.differences = round((abnormal_predict - normal_predict), fun_Decimalpoint(abnormal_predict - normal_predict))
#         self.normal_predict = round(normal_predict, fun_Decimalpoint(normal_predict))
#         self.abnormal_predict = round(abnormal_predict, fun_Decimalpoint(abnormal_predict))

#         self.explain_feature = pd.Series(result).sort_values(ascending=False).to_frame()
#         self.explain_feature.columns = ['explain_amount']
#         self.explain_feature = self.explain_feature.applymap(lambda x: round(float(x), fun_Decimalpoint(x)-1))
        
#         if plot is not False:
#             print(f'==== < Total Difference Normal to Abnormal Plot > ====')
#             print(f' → self.explain_plot')
#             self.explain_plot = plt.figure()
#             plt.title('Explain Difference Amount of Features\nbetween Normal and Abnormal Data')
#             self.explain_feature['explain_amount'].sort_values(ascending=True).plot.barh()
#             plt.show()
#             print()
#         print(f'==== < Total Difference Normal to Abnormal Data > ====')
#         print(f'Predicted Differences between Normal and Abnormal : {self.differences} (Abnormal: {self.abnormal_predict}, Normal: {self.normal_predict})')
#         print(f' → self.explain_feature')
#         print_DataFrame(self.explain_feature)
#         # return self.explain_feature













































"""
# test_df = pd.read_clipboard()
test_dict = {'y': {0: 10, 1: 13, 2: 20, 3: 7, 4: 15},
        'x1': {0: 2, 1: 4, 2: 5, 3: 2, 4: 4},
        'x2': {0: 'a', 1: 'a', 2: 'b', 3: 'b', 4: 'b'},
        'x3': {0: 10, 1: 8, 2: 5, 3: 12, 4: 7},
        'x4': {0: 'g1', 1: 'g2', 2: 'g1', 3: 'g2', 4: 'g3'}}

test_dict2 = {'y': [10, 17, 8, 15],
        'x1': ['a', 'b', 'b', 'a'],
        'x2': [2, 6, 7, 10],
        'x3': [0, 3, 5, 2],
        'x4': ['g2', 'g1', 'g2', 'g1']}
test_df = pd.DataFrame(test_dict)
test_df2 = pd.DataFrame(test_dict2)


test_df
pd.get_dummies(test_df, drop_first=True)

test_1 = test_df[['x2','x4']]
test_2 = test_df2[['x1','x4']]


OE1 = DS_OneHotEncoder()
df1 = OE1.fit_transform(test_df)

LE1 = DS_LabelEncoder()
df2 = LE1.fit_transform(test_df)


y1 = df1[['y']]
x1 = df1.iloc[:,1:]

y2 = df2[['y']]
x2 = df2.iloc[:,1:]

# Model
LR = LinearRegression()
LR.fit(x1, y1)

RF = RandomForestRegressor()
RF.fit(x2, y2)

GB = GradientBoostingRegressor()
GB.fit(x2, y2)



models = [('LR', LR), ('RF', RF), ('GB', GB)]
ES = VotingRegressor(models)
ES.fit(x2, y2)


# class Feature_Influence:
#     def __init__(self, model):
#         pass

#     def fit(self, X):
#         pass
    
#     def plot(self, X):
#         pass

# ***


# ------------------------------------------------------------------------
from itertools import product

test_num = test_df[['x1','x3']]

std_scale = StandardScaler()
std_scale.fit(test_num)
norm_test_num = std_scale.transform(test_num)

std_scale.inverse_transform(norm_test_num)


# if normalize:
X_mdl = x2
model = ES

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

'class' in str(type(MinMaxScaler()))
type(StandardScaler())
type('StandardScaler')

# class DF_Scaler:
# # StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
# def __init__(self, scaler):
#     if type(scaler) == str:
#         pass
#     elif "class 'sklearn.preprocessing._data." in str(type(scaler)):
#         pass











# case Product
def product_case(dictionary):
    if dictionary:
        from itertools import product

        product_case = []
        for k in fix_params:
            if type(fix_params[k]) == list:
                product_case.append(fix_params[k])
            else:
                product_case.append([fix_params[k]])
        return pd.DataFrame(eval('list(product(' + str(product_case)[1:-1] + '))'), columns=dictionary.keys())
    else:
        return False

class Quantile:
    def __init__(self, q):
        self.q = q
    
    def __call__(self, X):
        return X.quantile(self.q)




# fix_params = {'x4_g2': 0, 'x4_g3':[0, 1]}
fix_params = {}

n_points = 30

method = 'MinMax'
method = 'Quantile'
q_low = 0.05
q_upper = 0.95



q_l = Quantile(q_low)
q_u = Quantile(q_upper)
q_low_str = ('0' + str(int(q_low*100)) if q_low < 0.1 else str(int(q_low*100)) ) + '%'
q_upper_str = ('0' + str(int(q_upper*100)) if q_upper < 0.1 else str(int(q_upper*100)) ) + '%'

X_agg = X_mdl.agg(['mean','std','min','max', 'median', q_l, q_u])
X_agg.index = ['mean','std','min','max', '50%', q_low_str, q_upper_str]



if method == 'MinMax':
    mid = 'mean'
    limit_l = 'min'
    limit_u = 'max'
elif method == 'Quantile':
    mid = '50%'
    limit_l = q_low_str
    limit_u = q_upper_str


influence_init = pd.concat([X_agg.loc[[mid],:]]*n_points, ignore_index=True)
influence_linspace = pd.DataFrame(np.linspace(X_agg.loc[limit_l,:], X_agg.loc[limit_u,:], n_points), columns=X_mdl.columns)
grid_params = product_case(fix_params)
# grid_params

fix_params
grid_params


X_mdl
for xi, xc in enumerate(X_mdl):
    print(xc)




# influence_matrix ------------------------------------------------------
influence_matrix = {}

for xi, xc in enumerate(X_mdl):
    # print(xc)
    influence_df = influence_init.copy()
    influence_df[xc] = influence_linspace[xc]
    influence_df['group'] = mid

    if type(grid_params) != bool:    
        influence_grid_temper = influence_df.copy()

        if xc in grid_params.columns:
            params_df = grid_params.drop(xc, axis=1)
            params_df.drop_duplicates(inplace=True)
        else:
            params_df = grid_params.copy()

        for g in params_df.index:
            # print(xc, params_df.loc[g,:].to_dict())
            grid_temper = influence_grid_temper.copy()
            param_case = params_df.loc[g,:]
            grid_temper.loc[:,params_df.columns] = param_case.to_numpy()
            grid_temper['group'] = str(param_case.to_dict())

            influence_df = pd.concat([influence_df, grid_temper],axis=0)
        
    influence_matrix[xc] = influence_df
influence_matrix





# influence_predict ------------------------------------------------------
influence_predict = {}
for ic in influence_matrix:
    influence_predict[ic] = influence_matrix[ic][[ic]]
    influence_predict[ic]['predict'] = model.predict(influence_matrix[ic].iloc[:,:-1])
    influence_predict[ic]['group'] = influence_matrix[ic]['group']
    
influence_predict

# sns.relplot(data=influence_predict['x1'], x='x1', y='predict', kind='line', hue='group')


x_value = 'x1'
for gi, gv in influence_matrix[x_value].groupby('group'):
    plt_df = gv.drop('group', axis=1)

    if gi == 'mean':

        plt.plot(plt_df[x_value], model.predict(plt_df), '--', label=gi, c='red', linewidth=2, alpha=0.5)        # model Model
    else:
        plt.plot(plt_df[x_value], model.predict(plt_df), '-', label=gi, alpha=0.5)        # model Model

plt.legend()
plt.ylim([model.predict(X_mdl).min(), model.predict(X_mdl).max()])
plt.show()


pd.Series(model.feature_importances_, index=X_mdl.columns)





# LR.coef_
# plt.plot(influence_linspace['x1'], LR.predict(influence_matrix['x1']), 'o-')      # LR Model
# plt.plot(influence_linspace['x2'], LR.predict(influence_matrix['x2']), 'o-')      # LR Model
# plt.plot(influence_linspace['x3_a'], LR.predict(influence_matrix['x3_a']), 'o-')      # LR Model
# plt.plot(influence_linspace['x3_b'], LR.predict(influence_matrix['x3_b']), 'o-')      # LR Model

plt.plot(influence_linspace['x1'], RF.predict(influence_matrix['x1']), 'o-')        # RF Model
plt.plot(influence_linspace['x2'], RF.predict(influence_matrix['x2']), 'o-')        # RF Model
# plt.plot(influence_linspace['x3_a'], RF.predict(influence_matrix['x3_a']), 'o-')        # RF Model
# plt.plot(influence_linspace['x3_b'], RF.predict(influence_matrix['x3_b']), 'o-')        # RF Model

# pd.DataFrame( np.hstack([test_mean_df['x1'].to_numpy().reshape(-1,1), LR.predict(test_mean_df).reshape(-1,1)]), columns=['x1', 'x1_predict'])






# X_np = X_mdl.to_numpy()
# X_np

# def rbf_basis(X_mdl, d=5, sigma='auto'):
#     X_agg = X_mdl.agg(['mean','std','min','max'])
#     if sigma == 'auto':
#         sigma =  X_agg.loc['std',:]
#     return np.exp(-((X_mdl - X_agg.loc['mean',:])**2 / (2 * sigma**2)))

# pd.DataFrame(np.linspace(X_agg.loc['min',:], X_agg.loc['max',:], 10),columns=X_mdl.columns)

"""




















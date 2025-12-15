import warnings
import numpy as np

from sklearn.model_selection import PredefinedSplit
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from skopt import BayesSearchCV, space
from sklearn.utils.class_weight import compute_sample_weight

# RandomForestClassifier
# GradientBoostingClassifier
# XGBClassifier
# LGBMClassifier
# CatBoostClassifier



class TorchStateDict():
    def __init__(self, state_dict):
        self.state_dict = state_dict
    
    def state_dict(self):
        return self.state_dict
    
    def __call__(self):
        return self.state_dict
    
    def __repr__(self):
        return "<class 'torch.state_dict()'>"


# BayesOptLogger
class BayesOptLogger():
    def __init__(self, max_iter=None, verbose=1):
        warnings.filterwarnings( 'ignore', message="X does not have valid feature names")
        self.max_iter = max_iter
        self.count = 0
        self.last_state = None
        
        self.history = {'params':[], 'valid_loss':[], 'valid_score':[]}
        self.verbose=verbose

    def reset_count(self):
        self.count = 0
        self.last_state = None

    def logging(self, optim_result, verbose=None):
        # 'fun' 'func_vals' 'models' 'random_state' 'space' 'specs' 'x' 'x_iters'
        verbose = self.verbose if verbose is None else verbose
        
        self.history['params'].append(tuple(optim_result.x))  # 현재 iteration의 파라미터 값
        self.history['valid_loss'].append(optim_result.fun)
        self.history['valid_score'].append(-optim_result.fun)  # fun은 최소화 대상이므로, 음수로 바꾸면 점수
        
        self.count += 1
        max_iter_str = ''
        if self.max_iter is not None:
            max_iter_str = f"/{self.max_iter}"
            
        if verbose > 0:
            self.last_state = f"Iter {self.count}{max_iter_str} | Valid_Loss: {self.history['valid_loss'][-1]:.4f} | Params: {self.history['params'][-1]}"
            print(f"\r{self.last_state}", end="")
    
    def __call__(self, optim_result):
        self.logging(optim_result)
    
    def __repr__(self):
        return f"<BayesOpt Logger: {self.last_state}>"

# ModelPerformanceEvaluation
class ModelPerformanceEvaluation():
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    
    def __init__(self, data_preprocessor, target_task='regression'
                # ml_models=['RandomForest', 'GradientBoosting', 'XGB', 'LGBM', 'CatBoost'],
                ,ml_bayes_opt =True
                ,dl_bayes_opt = False
                ,ml_models=['RandomForest']
                ,dl_models=[]
                ,ml_kwargs = {'n_iter':2}
                ,dl_kwargs = {'epochs':300}
                ,verbose = 0
                ,random_state=None
                ):
        
        # [COMMON PART]
        self.data_preprocessor = data_preprocessor
        self.data_preprocessor.fit_tensor_dataloader()
        
        self.target_task_model_kind = None
        if 'reg' in target_task.lower():
            self.target_task_model_kind = 'Regressor'
        elif 'class' in target_task.lower():
            self.target_task_model_kind = 'Classifier'
        
        self.results = {}
        self.bayes_results = {}   
        self.verbose = verbose 
        self.random_state = random_state
        
        # [ML PART]
        if True:
            self.ml_dataset = None
            self.ml_models = ml_models
            self.ml_kwargs = ml_kwargs
            self.ml_bayes_opt = {}
            for model_name in self.ml_models:
                self.results[model_name] = {'best_estimator':None, 'train_loss':None, 'valid_loss':None, 'test_loss':None, 'best_params':{}}
                self.bayes_results[model_name] = {} 
                if ml_bayes_opt:
                    self.ml_bayes_opt[model_name] = None
            
            self.ml_classes = {
            'Regressor':{
                'RandomForest':RandomForestRegressor, 'GradientBoosting':GradientBoostingRegressor,
                'XGB': XGBRegressor, 'LGBM': LGBMRegressor, 'CatBoost': CatBoostRegressor
                },
            'Classifier':{
                'RandomForest':RandomForestClassifier, 'GradientBoosting':GradientBoostingClassifier,
                'XGB': XGBClassifier, 'LGBM': LGBMClassifier, 'CatBoost': CatBoostClassifier
                },
            }
            self.ml_default_params = {
                'RandomForest': {'n_jobs':-1, 'verbose': False, 'random_state':random_state}
                ,'GradientBoosting': { 'verbose': False, 'random_state':random_state}
                ,'XGB': {'n_jobs':-1, 'verbosity': 0, 'random_state':random_state}
                ,'LGBM': {'n_jobs':-1, 'verbose': -1, 'logging_level':'Silent', 'random_state':random_state}
                ,'CatBoost': {'thread_count':-1, 'verbose': False, 'random_seed':random_state}
            }
            self.ml_optimize_params_range = {
                    'RandomForest':{'n_estimators': (50, 500), 'max_depth': (3, 30), 'min_samples_split': (2, 20), 'min_samples_leaf': (1, 10), 'max_features': (0.1, 1.0)}
                    ,'GradientBoosting':{'n_estimators': (50, 500), 'learning_rate': (0.01, 0.3), 'max_depth': (3, 15), 'min_samples_split': (2, 20), 'min_samples_leaf': (1, 10), 'subsample': (0.5, 1.0)}
                    ,'XGB':{'n_estimators': (50, 1000), 'max_depth': (3, 15), 'learning_rate': (0.01, 0.3), 'min_child_weight': (1, 10), 'subsample': (0.5, 1.0), 'colsample_bytree': (0.5, 1.0), 'gamma': (0.0, 5.0), 'reg_alpha': (0.0, 1.0), 'reg_lambda': (0.0, 1.0)}
                    ,'LGBM':{'num_leaves': (20, 200), 'max_depth': (-1, 15), 'learning_rate': (0.01, 0.3), 'n_estimators': (50, 1000), 'min_child_samples': (5, 50), 'subsample': (0.5, 1.0), 'colsample_bytree': (0.5, 1.0), 'reg_alpha': (0.0, 1.0), 'reg_lambda': (0.0, 1.0)}
                    ,'CatBoost':{'iterations': (50, 1000), 'depth': (3, 15), 'learning_rate': (0.01, 0.3), 'l2_leaf_reg':(1.0, 10.0), 'rsm':(0.5,1.0), 'bagging_temperature': (0.0, 5.0), 'border_count': (32, 255)}
                }
            self.bayes_opt_params = {
                'Common': {'n_jobs':-1, 'n_points':1,  'verbose':0, 'random_state':random_state} | ml_kwargs
                ,'Regressor': {'scoring': 'neg_mean_squared_error'}
                ,'Classifier':{'scoring': 'neg_log_loss'}
            }
        
        # [DL PART]
        if True:
            self.dl_dataset = None
            self.dl_models = {}
            self.dl_kwargs = dl_kwargs
            
            
            for idx, model in enumerate(dl_models):
                
                model_idx = f'TorchModel_{idx}'
                self.dl_models[model_idx] = model
                self.results[model_idx] = {'best_estimator':None, 'train_loss':None, 'valid_loss':None, 'test_loss':None, 'best_params':{}}
    
    # ML
    def make_ml_dataset(self):
        transformed_data = self.data_preprocessor.transformed_data
        train_y, valid_y, tests_y = [dataset[0] for dataset in transformed_data.values()]
        train_X, valid_X, tests_X = [np.concatenate(dataset[1:],axis=1) for dataset in transformed_data.values()]

        pre_find_split_idx = np.zeros(len(train_X) + len(valid_X))
        pre_find_split_idx[:len(train_X)] = -1      # training : -1, validation : 0
        pre_finded_split = PredefinedSplit(pre_find_split_idx)
        
        train_valid_X = np.concatenate([train_X, valid_X], axis=0)
        train_valid_y = np.concatenate([train_y, valid_y], axis=0).ravel()

        self.ml_dataset = {'train': (train_X, train_y)
                        ,'valid': (valid_X, valid_y)
                        ,'test':(tests_X, tests_y)
                        ,'train_valid': (train_valid_X, train_valid_y)
                        ,'pre_finded_split': pre_finded_split}
        
        return self.ml_dataset
    
    def run_ml_learning_each(self, dataset, model_name, ml_target_class, sample_weight=None):
        # additional arguments
        add_base_options = {}
        add_fit_options = {}
        if self.target_task_model_kind == 'Classifier':
            if model_name in ['RandomForest','LGBM']:
                add_base_options['class_weight'] = 'balanced'
            elif model_name in ['GradientBoosting','XGB','CatBoost']:
                add_fit_options['sample_weight'] = sample_weight
        
        # metric
        metric_name = self.bayes_opt_params[self.target_task_model_kind]['scoring'].replace('neg_','')
        metric_fun = eval(f"metrics.{metric_name}")
        
        # base_model
        base_model = ml_target_class[model_name](**self.ml_default_params[model_name], **add_base_options)
        
        # ml_bayes_opt
        if len(self.ml_bayes_opt) > 0:
            # bayes_opt_logger
            bayes_opt_logger = BayesOptLogger(max_iter=self.bayes_opt_params['Common']['n_iter']
                                                ,verbose=self.verbose-1)
            # ml_bayes_opt
            self.ml_bayes_opt[model_name] = BayesSearchCV(estimator=base_model
                        ,search_spaces=self.ml_optimize_params_range[model_name]
                        ,cv=self.ml_dataset['pre_finded_split']
                        , **(self.bayes_opt_params['Common'] | self.bayes_opt_params[self.target_task_model_kind]) 
                        )
            self.ml_bayes_opt[model_name].fit(*dataset
                                            ,callback=bayes_opt_logger 
                                            ,**add_fit_options)
            
            # train_results save
            self.bayes_results[model_name]['valid_loss'] = bayes_opt_logger.history['valid_loss']
            self.bayes_results[model_name]['params'] = bayes_opt_logger.history['params']
            best_estimator = self.ml_bayes_opt[model_name].best_estimator_
            self.results[model_name]['best_params'] = dict(self.ml_bayes_opt[model_name].best_params_)
            
            if self.verbose > 1:
                print()
        else:
            base_model.fit(*dataset, **add_fit_options)
            best_estimator = base_model
            
        # best_model performance evaludate
        for dataset_name in ['train', 'valid', 'test']:
            predict = best_estimator.predict(self.ml_dataset[dataset_name][0])
            self.results[model_name][f'{dataset_name}_loss'] = metric_fun(self.ml_dataset[dataset_name][1].ravel(), predict)
        
        if self.verbose:
            print(f"(Loss) train: {self.results[model_name]['train_loss']:.4f} | valid: {self.results[model_name]['valid_loss']:.4f} | test: {self.results[model_name]['test_loss']:.4f}", end="")
        
        # save info
        self.results[model_name]['best_estimator'] = best_estimator
        
    def run_ml_learning(self):
        if self.ml_dataset is not None:
            ml_target_class = self.ml_classes[self.target_task_model_kind]
            
            # class category weightes
            sample_weight = None
            if self.target_task_model_kind == 'Classifier':
                # 1) train 부분만으로 sample_weight 계산
                train_sample_weight = compute_sample_weight(class_weight='balanced'
                                                       ,y=self.ml_dataset['train'][1])
                # 2) 전체 길이의 sample_weight 생성 (validation 부분은 1로 채움)
                sample_weight = np.ones(len(self.ml_dataset['train_valid'][1]))
                sample_weight[:len(train_sample_weight)] = train_sample_weight
            
            # training dataset
            train_X, train_y = self.ml_dataset['train']
            valid_X, valid_y = self.ml_dataset['valid']
            train_valid_X = np.concatenate([train_X, valid_X], axis=0)
            train_valid_y = np.concatenate([train_y, valid_y], axis=0).ravel()
            dataset = (train_valid_X, train_valid_y)
            
            # model learning
            for model_name in self.ml_models:
                if self.verbose > 0:
                    print(f"< Learning {model_name} >")
                    self.run_ml_learning_each(dataset, model_name, ml_target_class, sample_weight)
                print()
    
    # DL
    def make_dl_dataset(self):
        self.dl_dataset = self.data_preprocessor.tensor_dataloader
        return self.dl_dataset
    
    def run_dl_learning_each(self, model_name, model):
        model.train_model(self.dl_dataset['train'], self.dl_dataset['valid'], epochs=self.dl_kwargs['epochs'])
        
        if model.early_stop_loss is not None:
            model.set_best_model()
            # tm.set_best_model(verbose=self.verbose)
        model.test_model(self.dl_dataset['test'])
        
        # save_result
        self.results[model_name]['best_estimator'] = model.model
        self.results[model_name]['train_loss'] = model.train_losses[-1]
        self.results[model_name]['valid_loss'] = model.valid_losses[-1]
        self.results[model_name]['test_loss'] = model.test_losses[-1]
        self.results[model_name]['best_params'] = TorchStateDict(model.model.state_dict())
        
        if self.verbose:
            print(f"(Loss) train: {self.results[model_name]['train_loss']:.4f} | valid: {self.results[model_name]['valid_loss']:.4f} | test: {self.results[model_name]['test_loss']:.4f}", end="")
    
    def run_dl_learning(self):
        if self.dl_dataset is not None:
            
            # model learning
            for model_name, model in self.dl_models.items():
                if self.verbose > 0:
                    print(f"< Learning {model_name} >")
                    self.run_dl_learning_each(model_name, model)
                print()


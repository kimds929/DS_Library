import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import os
import time
from datetime import datetime
from six.moves import cPickle
from tqdm.notebook import tqdm
from IPython.display import clear_output, display, update_display
from collections import OrderedDict

from time import sleep


# epoch_time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# DeepLearning MDL Predict
class PredictDL():
    def __init__(self, model, input='torch', device='cpu'):
        self.model = model
        self.input = input
        self.device = device
    
    def predict(self, x):
        if self.input == 'torch':
            self.model.eval()
            return self.model(torch.FloatTensor(np.array(x)).to(self.device)).to('cpu').detach().numpy().ravel()


class TorchStateDict():
    def __init__(self, state_dict):
        self.state_dict = state_dict
    
    def state_dict(self):
        return self.state_dict
    
    def __call__(self):
        return self.state_dict
    
    def __repr__(self):
        return "<class 'torch.state_dict()'>"


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
    def __init__(self, min_iter=0, patience=4, optimize='miminize'):
        self.min_iter = min_iter  # 최소 반복 횟수 저장
        self.patience = np.inf if patience is None else patience
        self.optimize = optimize
        
        self.metrics = []       # (epoch, event, score, save, r_score, r_save)
        self.metrics_frame = pd.DataFrame()
        self.patience_scores = []
        self.optimum = (0, np.inf if 'min' in optimize else -np.inf, '', None, None)    # (epoch, score, save, r_score, r_save)
        self.plot = None
    
    def reset_patience_scores(self):
        self.patience_scores = []
    
    def load(self, EarlyStopping):
        self.metrics = EarlyStopping.metrics
        self.metrics_frame = EarlyStopping.metrics_frame
        self.optimize = EarlyStopping.optimize
        self.optimum = EarlyStopping.optimum
        self.patience = EarlyStopping.patience
        self.patience_scores = EarlyStopping.patience_scores
        self.plot = EarlyStopping.plot
        print("Successfully Load EarlyStopping Object!")
    
    def generate_plot(self, figsize=None, yscale='symlog', verbose=0):
        """
        yscale  
        """
        metrics_frame = self.metrics_frame
        label_score = metrics_frame.columns[2]
        label_r_score = metrics_frame.columns[4]

        # plot        
        if verbose == 'plot' or verbose=='all':
            clear_output(wait=True)

        if figsize is None:
            self.plot = plt.figure()
        else:
            self.plot = plt.figure(figsize=figsize)
        
        # reference_score
        if (~pd.isna(metrics_frame.iloc[:,4])).sum() > 0:
            plt.plot(metrics_frame['epoch'], metrics_frame[label_r_score], alpha=0.5, color='orange', label=label_r_score)
            
        plt.plot(metrics_frame['epoch'], metrics_frame[label_score], alpha=0.5, color='steelblue', label=label_score)
        plt.legend(loc='upper right')
        
        metrics_colors = ['steelblue', 'gold', 'red', 'green']
        # (pandas) .groupby 
        #       - observed=False (default) : 모든 카테고리(데이터에 없는 값 포함)를 결과에 포함
        #       - observed=True : 실제로 데이터에 존재하는 카테고리만 결과에 포함
        for me, (mgi, mgv) in enumerate(metrics_frame.groupby('event', observed=False)):    
            plt.scatter(mgv['epoch'], mgv[label_score], color=metrics_colors[me])            
        for mi, mg in metrics_frame[metrics_frame['event'] != ''].iterrows():
            event_name = 'p' if mg['event'] == 'patience' else ('★' if mg['event']=='optimum' else ('break' if mg['event'] == 'break' else ''))
            plt.text(mg['epoch'], mg[label_score], event_name)
        plt.yscale(yscale)
        plt.xlabel('epoch')
        plt.ylabel('score')
        # plt.yscale('symlog')
        if verbose == 'plot' or verbose=='all':
            plt.show()
            time.sleep(sleep)
        else:
            plt.close()
        return self.plot

    def early_stop(self, score, save=None, label=None,
                   reference_score=None, reference_save=None, reference_label=None,
                   verbose=0, sleep=0, save_all=False):
        
        result = 'none'
        epoch = len(self.metrics)+1
        label_score = 'valid_score' if label is None else label
        label_r_score = 'train_score' if reference_label is None else reference_label
        
        # min_iter 이전에는 patience 로직을 건너뛰고 optimum만 체크
        if epoch < self.min_iter:
            if 'min' in self.optimize:
                if score < self.optimum[1]:
                    self.patience_scores = []
                    result = 'optimum'
            elif 'max' in self.optimize:
                if score > self.optimum[1]:
                    self.patience_scores = []
                    result = 'optimum'
        
        # min_iter 이후 patience에 맞게 학습하여 early_stop
        else:
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
            
        if save is not None and 'collections.OrderedDict' in str(type(save)):
            save = TorchStateDict(save)
        
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

        self.metrics_frame = metrics_frame.copy()
        
        # plot
        self.generate_plot(verbose=verbose)
        
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
        
        return result




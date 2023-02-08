import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
from IPython.display import clear_output
from collections import OrderedDict



# epoch_time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# DeepLearning MDL Predict
class PredictDL():
    def __init__(self, model, input='torch'):
        self.model = model
        self.input = input
    
    def predict(self, x):
        if self.input == 'torch':
            return self.model.predict(torch.FloatTensor(np.array(x))).numpy().ravel()



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
                   verbose=0, sleep=0, save_all=False):
        
        result = 'none'
        epoch = len(self.metrics)+1
        label_score = 'valid_score' if label is None else label
        label_r_score = 'train_score' if reference_label is None else reference_label
        
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
            plt.plot(metrics_frame['epoch'], metrics_frame[label_r_score], 'o-', alpha=0.5, color='orange', label=label_r_score)
            
        plt.plot(metrics_frame['epoch'], metrics_frame[label_score], alpha=0.5, color='steelblue', label=label_score)
        plt.legend(loc='upper right')
        
        metrics_colors = ['steelblue', 'gold', 'red', 'green']
        for me, (mgi, mgv) in enumerate(metrics_frame.groupby('event')):
            plt.scatter(mgv['epoch'], mgv[label_score], color=metrics_colors[me])            
        for mi, mg in metrics_frame[metrics_frame['event'] != ''].iterrows():
            event_name = 'p' if mg['event'] == 'patience' else ('★' if mg['event']=='optimum' else ('break' if mg['event'] == 'break' else ''))
            plt.text(mg['epoch'], mg[label_score], event_name)
        plt.xlabel('epoch')
        plt.ylabel('score')
        # plt.yscale('symlog')
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




class AutoML(torch.nn.Module):
    '''
    【 Required Library 】torch, from collections import OrderedDict
    【 Required Customized Class 】
    
     < Method >
     . __init__:
     . create_architecture:
     . create_model:
     . forward:
     . predict:
    '''
    def __init__(self, X, y=None, hidden_layers=3, hidden_nodes=None, structure_type=None,
                 layer_structure={'Linear':'io','BatchNorm1d': 'o', 'ReLU': None}):
        super(AutoML, self).__init__()
        self.x_shape = X.shape
        self.y_ndim = None if y is None else y.ndim
        self.y_shape = None if y is None else y.shape
        
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.structure_type = structure_type
        
        self.model_dict = OrderedDict()
        
        self.layer_structure = layer_structure
        self.create_architecture()      # AutoML Architecture
        self.create_model()      # AutoML Architecture
        
        self.model = torch.nn.Sequential(self.model_dict)
        
        self.predicts = {}
    
    def create_architecture(self):
        n = self.x_shape[1]
        
        if self.hidden_nodes is None:
            # AutoML with Hidden Layer 
            hidden_nodes = [n]
            
            if self.structure_type is None or self.structure_type == 1:
                for i in range(self.hidden_layers):
                    n *= 2
                    hidden_nodes.append(n)
                self.hidden_nodes = hidden_nodes
            elif self.structure_type == 2:
                for i in range( (self.hidden_layers-1)//2):
                    n *= 2
                    hidden_nodes.append(n)
                self.hidden_nodes = hidden_nodes.copy()
                if (self.hidden_layers-1) % 2 == 1:
                    self.hidden_nodes.append(n*2)
                self.hidden_nodes = self.hidden_nodes + hidden_nodes[::-1]
            
        else:
            # AutoML with Hidden Nodes 
            self.hidden_layers = len(self.hidden_nodes)
            self.hidden_nodes = [n] + self.hidden_nodes
        
        if self.y_ndim is None:
            self.hidden_nodes.append(1)
        else:
            self.hidden_nodes.append(1 if  self.y_ndim == 1 else self.y_shape[1])
   
    def create_model(self, layer_structure=None):
        layer_structure = self.layer_structure if layer_structure is None else layer_structure
            
        for hi, hn in enumerate(self.hidden_nodes):
            if hi < len(self.hidden_nodes)-2:
                n_input = self.hidden_nodes[hi]
                n_output = self.hidden_nodes[hi+1]
                
                for ls, ls_io in layer_structure.items():
                    if ls_io == 'io':
                        self.model_dict.update({f"l{hi}_{ls}": eval(f"torch.nn.{ls}({n_input}, {n_output})") })
                    elif ls_io == 'i':
                        self.model_dict.update({f"l{hi}_{ls}": eval(f"torch.nn.{ls}({n_input})") })
                    elif ls_io == 'o':
                        self.model_dict.update({f"l{hi}_{ls}": eval(f"torch.nn.{ls}({n_output})") })
                    elif ls_io is None:
                        self.model_dict.update({f"l{hi}_{ls}": eval(f"torch.nn.{ls}()") })
            elif hi == len(self.hidden_nodes)-2:
                self.model_dict.update({f"nn{hi}": torch.nn.Linear(self.hidden_nodes[hi], self.hidden_nodes[hi+1])})

    def forward(self, x, training=True):
        if training is True:
            for layer_name, layer in zip(self.model_dict.keys(), self.model):
                x = layer(x)
                self.predicts[layer_name] = x
            
        elif training is False:
            with torch.no_grad():
                for layer_name, layer in zip(self.model_dict.keys(), self.model):
                    x = layer(x)
                    self.predicts[layer_name] = x
        return x

    def predict(self, x):
        return self.forward(x, training=False)

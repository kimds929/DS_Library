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






# DataSet ########################################################################
from six.moves import cPickle
import os
import time
import numpy as np
import pandas as pd
import torch

from datetime import datetime
from tqdm.notebook import tqdm
from IPython.display import clear_output, display, update_display
from sklearn.model_selection import train_test_split



class TorchDataLoader():
    def __init__(self, *args, split_size=(0.7, 0.1, 0.2), random_state=None, **kwargs):
        self.args = args
        assert (np.array(list(map(len, self.args)))/len(self.args[0])).all() == True, 'Arguments must have same length'
        self.idx = np.arange(len(self.args[0]))
        
        self.split_size = [s/np.sum(split_size) for s in split_size]
        
        self.train_test_split_size = None
        self.train_valid_split_size = None
        
        if len(self.split_size) == 2:
            self.train_test_split_size = self.split_size
        elif len(self.split_size) == 3:
            self.train_test_split_size = [self.split_size[0]+self.split_size[1], self.split_size[2]]
            self.train_valid_split_size = [s/self.train_test_split_size[0] for s in self.split_size[:2]]
        
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.torch_data = None
        self.dataset = None
        self.dataloader = None
        
    def split(self, dtypes=None, random_state=None):
        random_state = self.random_state if random_state is None else random_state
        self.train_idx, self.test_idx = train_test_split(self.idx, test_size=self.train_test_split_size[-1], random_state=random_state)
        self.index = (self.train_idx, self.test_idx)
        if self.train_valid_split_size is not None:
            self.train_idx, self.valid_idx = train_test_split(self.train_idx, test_size=self.train_valid_split_size[-1], random_state=random_state)
            self.index = (self.train_idx, self.valid_idx, self.test_idx)
        
        [print(len(index), end=', ') for index in self.index]
        print()
        if dtypes is None:
            self.torch_data = tuple([tuple([torch.tensor(arg[idx]) for idx in self.index]) for arg in self.args])
        else:
            self.torch_data = tuple([tuple([torch.tensor(arg[idx]).type(dtype) for idx in self.index]) for arg, dtype in zip(self.args, dtypes)])
    
    def make_dataset(self, dtypes=None, random_state=None):
        if self.torch_data is None:
            self.split(dtypes, random_state)
            
        self.dataset = tuple([torch.utils.data.TensorDataset(*data) for data in zip(*self.torch_data)])

    def make_dataloader(self, dtypes=None, random_state=None, **kwargs):
        if self.dataset is None:
            self.make_dataset(dtypes, random_state)
        if len(kwargs) > 0:
            self.kwargs = kwargs
            
        self.dataloader = tuple([torch.utils.data.DataLoader(dataset, **self.kwargs) for dataset in self.dataset])
        
        for sample in self.dataloader[0]:
            break
        self.sample = sample




class TorchModeling():
    """
    self.__init__(model, device)
    self.compile(optimizer, (loss_function))
    self.train_model(train_loader, (valid_loader), epochs, (loss_function), (metrics_function))
    
    loss_function(model, batch, optimizer)
    meteic_function(model, batch)
    """
    def __init__(self, model, device='cpu'):
        self.now_date = datetime.strftime(datetime.now(), '%y%m%d_%H')

        self.model = model.to(device)
        self.device = device
        self.model_t = None
        self.train_t = 1

        self.train_losses = []
        self.train_metrics = []
        self.valid_losses = []
        self.valid_metrics = []
        self.test_losses = []
        self.test_metrics = [] 

        self.train_info = []
        self.test_info = []
    
    def get_save_path(self):
        return f"{os.getcwd()}/{self.now_date}_{self.model._get_name()}"

    def fun_decimal_point(self, value):
        if type(value) == str or type(value) == int:
            return value
        else:
            if value == 0:
                return 3
            try:
                point_log10 = np.floor(np.log10(abs(value)))
                point = int((point_log10 - 3)* -1) if point_log10 >= 0 else int((point_log10 - 2)* -1)
            except:
                point = 0
            return np.round(value, point)

    def compile(self, optimizer, loss_function=None, metrics_function=None, scheduler=None,
                early_stop_loss=None, early_stop_metrics=None):
        """
        loss_function(model, x, y) -> loss
        """
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics_function = metrics_function
        self.scheduler = scheduler
        self.early_stop_loss = early_stop_loss
        self.early_stop_metrics = early_stop_metrics

    def recompile(self, optimizer=None, loss_function=None, metrics_function=None, scheduler=None,
                early_stop_loss=None, early_stop_metrics=None):
        if scheduler is not None:
            self.scheduler = scheduler
            self.scheduler.optimizer = self.optimizer

        if optimizer is not None:
            self.optimizer = optimizer

            if self.scheduler is not None:
                self.scheduler.optimizer = self.optimizer

        if loss_function is not None:
            self.loss_function = loss_function
        
        if metrics_function is not None:
            self.metrics_function = metrics_function

        if early_stop_loss is not None:
            self.early_stop_loss.patience = early_stop_loss.patience
            self.early_stop_loss.optimize = early_stop_loss.optimize
            early_stop_loss.load(self.early_stop_loss)
            self.early_stop_loss = early_stop_loss

        if early_stop_metrics is not None:
            self.early_stop_metrics.patience = early_stop_metrics.patience
            self.early_stop_metrics.optimize = early_stop_metrics.optimize
            early_stop_metrics.load(self.early_stop_metrics)
            self.early_stop_metrics = early_stop_metrics

    def train_model(self, train_loader, valid_loader=None, epochs=10, loss_function=None, metrics_function=None,
                tqdm_display=False, early_stop=True, save_parameters=False, display_earlystop_result=False, optimizer_step=True):
        loss_function = self.loss_function if loss_function is None else loss_function
        metrics_function = self.metrics_function if metrics_function is None else metrics_function
        final_epcohs = self.train_t + epochs - 1
        # [START of Epochs Loop] ############################################################################################
        epochs_iter = tqdm(range(self.train_t, self.train_t + epochs), desc="Epochs", total=epochs, position=0, leave=True) if tqdm_display else range(self.train_t, self.train_t + epochs)
        for epoch in epochs_iter:
            print_info = {}

            # train Loop --------------------------------------------------------------
            self.model.train()
            train_epoch_loss = []
            train_epoch_metrics = []
            train_iter = tqdm(enumerate(train_loader), desc="Train Batch", total=len(train_loader), position=1, leave=False) if tqdm_display else enumerate(train_loader)
            for batch_idx, (batch) in train_iter:
                batch_device = (batch_data .to(self.device) for batch_data in batch)
                loss = loss_function(self.model, batch_device, self.optimizer)
                
                if optimizer_step is True:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
                with torch.no_grad():
                    train_epoch_loss.append( loss.to('cpu').detach().numpy() )
                    if self.metrics_function is not None:
                        train_epoch_metrics.append( self.metrics_function(self.model, batch_device) )

            with torch.no_grad():
                print_info['train_loss'] = np.mean(train_epoch_loss)
                self.train_losses.append(print_info['train_loss'])
                if self.metrics_function is not None:
                    print_info['train_metrics'] = np.mean(train_epoch_metrics)
                    self.train_metrics.append(print_info['train_metrics'])

            # scheduler ---------------------------------------------------------
            if self.scheduler is not None:
                self.scheduler.step()

            with torch.no_grad():
                # valid Loop ---------------------------------------------------------
                if valid_loader is not None and len(valid_loader) > 0:
                    self.model.eval()
                    valid_epoch_loss = []
                    valid_epoch_metrics = []
                    valid_iter = tqdm(enumerate(valid_loader), desc="Valid Batch", total=len(valid_loader), position=1, leave=False) if tqdm_display else enumerate(valid_loader)
                    for batch_idx, (batch) in valid_iter:
                        batch_device = (batch_data .to(self.device) for batch_data in batch)
                        
                        loss = loss_function(self.model, batch_device)
                    
                        valid_epoch_loss.append( loss.to('cpu').detach().numpy() )
                        if self.metrics_function is not None:
                            valid_epoch_metrics.append( self.metrics_function(self.model, batch_device) )

                    print_info['valid_loss'] = np.mean(valid_epoch_loss)
                    self.valid_losses.append(print_info['valid_loss'])
                    if self.metrics_function is not None:
                        print_info['valid_metrics'] = np.mean(valid_epoch_metrics)
                        self.valid_metrics.append(print_info['valid_metrics'])
            
                # print_info ---------------------------------------------------------
                self.train_info.append(print_info)
                print_sentences = ",  ".join([f"{k}: {str(self.fun_decimal_point(v))}" for k, v in print_info.items()])
                
                # print(f"[Epoch: {epoch}/{final_epcohs}] {print_sentences}")
                if final_epcohs - epoch + 1 == epochs:
                    display(f"[Epoch: {epoch}/{final_epcohs}] {print_sentences}", display_id="epoch_result")
                else:
                    update_display(f"[Epoch: {epoch}/{final_epcohs}] {print_sentences}", display_id="epoch_result")

                # early_stop ---------------------------------------------------------
                early_stop_TF = None
                if self.early_stop_loss is not None:
                    score = print_info['valid_loss'] if (valid_loader is not None and len(valid_loader) > 0) else print_info['train_loss']
                    reference_score = print_info['train_loss'] if (valid_loader is not None and len(valid_loader) > 0) else None
                    # params = self.model.state_dict() if save_parameters else None
                    params = self.model.state_dict()
                    early_stop_TF = self.early_stop_loss.early_stop(score=score, reference_score=reference_score, save=params, verbose=0)

                    if save_parameters:
                        path_save_loss = f"{self.get_save_path()}_earlystop_loss.pth"
                        cPickle.dump(self.early_stop_loss, open(path_save_loss, 'wb'))      # save earlystop loss

                if self.metrics_function is not None and self.early_stop_metrics is not None:
                    score = print_info['valid_metrics'] if (valid_loader is not None and len(valid_loader) > 0) else print_info['train_metrics']
                    reference_score = print_info['train_metrics'] if (valid_loader is not None and len(valid_loader) > 0) else None
                    # params = self.model.state_dict() if save_parameters else None
                    params = self.model.state_dict()
                    self.early_stop_loss.early_stop(score=score, reference_score=reference_score, save=params, verbose=0)

                    if save_parameters:
                        path_save_metrics = f"{self.get_save_path()}_earlystop_metrics.pth"
                        cPickle.dump(self.early_stop_metrics, open(path_save_metrics, 'wb'))      # save earlystop metrics

                # save_parameters ---------------------------------------------------------
                if save_parameters:
                    path_save_weight = f"{self.get_save_path()}_weights.pth"
                    cPickle.dump(self.model.state_dict(), open(path_save_weight, 'wb'))      # save earlystop weights

                # step update ---------------------------------------------------------
                self.train_t += 1
                self.model_t = self.train_t

                # early_stop break ---------------------------------------------------------
                if early_stop is True and early_stop_TF == 'break':
                        break
        
        if display_earlystop_result:
            if self.early_stop_loss is not None:
                display(self.early_stop_loss.plot)
            if self.metrics_function is not None and self.early_stop_metrics is not None:
                display(self.early_stop_metrics.plot)
    # [END of Epochs Loop] ############################################################################################

    def set_best_model(self, t=None, pth_path=None):
        if self.early_stop_loss is None:
            print('There is no save point. (early_stop)')
        elif (t is not None) and (pth_path is not None):
            self.model_t = t
            self.model.load(pth_path)
            print('<All keys matched successfully>')
        else:
            optimum_point = self.early_stop_loss.optimum
            self.model_t = optimum_point[0]
            self.model.load_state_dict(optimum_point[2].state_dict)
            print('<All keys matched successfully>')         
    
    def test_model(self, test_loader, tqdm_display=False, verbose=0):
        with torch.no_grad():
            print_info = {"epoch":self.model_t-1}
            # test Loop ---------------------------------------------------------
            if test_loader is not None and len(test_loader) > 0:
                self.model.eval()
                
                test_batch_lens = []
                test_epoch_loss = []
                test_epoch_metrics = []
                test_iter = tqdm(enumerate(test_loader), desc="Valid Batch", total=len(test_loader), position=1, leave=False) if tqdm_display else enumerate(test_loader)
                for batch_idx, (batch) in test_iter:
                    batch_len = len(batch[0])
                    batch_device = (batch_data.to(self.device) for batch_data in batch)
                    
                    loss = self.loss_function(self.model, batch_device)

                    test_batch_lens.append( batch_len )
                    test_epoch_loss.append( loss.to('cpu').detach().numpy() )
                    if self.metrics_function is not None:
                        test_epoch_metrics.append( self.metrics_function(self.model, batch_device) )

                print_info['test_loss'] = (np.array(test_epoch_loss) * np.array(test_batch_lens)).sum() / sum(test_batch_lens)
                self.test_losses.append(print_info['test_loss'])
                if self.metrics_function is not None:
                    print_info['test_metrics'] = (np.array(test_epoch_metrics) * np.array(test_batch_lens)).sum() / sum(test_batch_lens)
                    self.test_metrics.append(print_info['test_metrics'])
            print_sentences = ",  ".join([f"{k}: {str(self.fun_decimal_point(v))}" for k, v in print_info.items() if k != 'epoch'])
            if verbose > 0:
                print(f"[{self.model_t-1} epoch test performances] {print_sentences}")
            self.test_info.append(print_info)
        return print_info

# def gaussian_loss(model, x, y):
#     mu, logvar = model(x)
#     std = torch.exp(0.5*logvar)
#     loss = torch.nn.functional.gaussian_nll_loss(mu, y, std**2)
#     # loss = loss_gaussian(mu, y, std**2)
#     return loss

# model = BNN_DirectEnsemble2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=5)
# # model = BNN_Model_2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=5)

# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# tm = TorchModeling(model=model, device=device)
# tm.compile(optimizer=optimizer
#             ,loss_function = gaussian_loss
#             , scheduler=scheduler
#             , early_stop_loss = EarlyStopping(patience=5)
#             )
# # tm.early_stop_loss.reset_patience_scores()
# # tm.training(train_loader=train_loader, valid_loader=valid_loader, epochs=100, display_earlystop_result=True)
# tm.train_model(train_loader=train_loader, valid_loader=valid_loader, epochs=30, display_earlystop_result=True, early_stop=False)
# tm.test_model(test_loader=test_loader)





# class AutoML(torch.nn.Module):
#     '''
#     【 Required Library 】torch, from collections import OrderedDict
#     【 Required Customized Class 】
    
#      < Method >
#      . __init__:
#      . create_architecture:
#      . create_model:
#      . forward:
#      . predict:
#     '''
#     def __init__(self, X, y=None, hidden_layers=3, hidden_nodes=None, structure_type=None,
#                  layer_structure={'Linear':'io','BatchNorm1d': 'o', 'ReLU': None}):
#         super(AutoML, self).__init__()
#         self.x_shape = X.shape
#         self.y_ndim = None if y is None else y.ndim
#         self.y_shape = None if y is None else y.shape
        
#         self.hidden_layers = hidden_layers
#         self.hidden_nodes = hidden_nodes
#         self.structure_type = structure_type
        
#         self.model_dict = OrderedDict()
        
#         self.layer_structure = layer_structure
#         self.create_architecture()      # AutoML Architecture
#         self.create_model()      # AutoML Architecture
        
#         self.model = torch.nn.Sequential(self.model_dict)
        
#         self.predicts = {}
    
#     def create_architecture(self):
#         n = self.x_shape[1]
        
#         if self.hidden_nodes is None:
#             # AutoML with Hidden Layer 
#             hidden_nodes = [n]
            
#             if self.structure_type is None or self.structure_type == 1:
#                 for i in range(self.hidden_layers):
#                     n *= 2
#                     hidden_nodes.append(n)
#                 self.hidden_nodes = hidden_nodes
#             elif self.structure_type == 2:
#                 for i in range( (self.hidden_layers-1)//2):
#                     n *= 2
#                     hidden_nodes.append(n)
#                 self.hidden_nodes = hidden_nodes.copy()
#                 if (self.hidden_layers-1) % 2 == 1:
#                     self.hidden_nodes.append(n*2)
#                 self.hidden_nodes = self.hidden_nodes + hidden_nodes[::-1]
            
#         else:
#             # AutoML with Hidden Nodes 
#             self.hidden_layers = len(self.hidden_nodes)
#             self.hidden_nodes = [n] + self.hidden_nodes
        
#         if self.y_ndim is None:
#             self.hidden_nodes.append(1)
#         else:
#             self.hidden_nodes.append(1 if  self.y_ndim == 1 else self.y_shape[1])
   
#     def create_model(self, layer_structure=None):
#         layer_structure = self.layer_structure if layer_structure is None else layer_structure
            
#         for hi, hn in enumerate(self.hidden_nodes):
#             if hi < len(self.hidden_nodes)-2:
#                 n_input = self.hidden_nodes[hi]
#                 n_output = self.hidden_nodes[hi+1]
                
#                 for ls, ls_io in layer_structure.items():
#                     if ls_io == 'io':
#                         self.model_dict.update({f"l{hi}_{ls}": eval(f"torch.nn.{ls}({n_input}, {n_output})") })
#                     elif ls_io == 'i':
#                         self.model_dict.update({f"l{hi}_{ls}": eval(f"torch.nn.{ls}({n_input})") })
#                     elif ls_io == 'o':
#                         self.model_dict.update({f"l{hi}_{ls}": eval(f"torch.nn.{ls}({n_output})") })
#                     elif ls_io is None:
#                         self.model_dict.update({f"l{hi}_{ls}": eval(f"torch.nn.{ls}()") })
#             elif hi == len(self.hidden_nodes)-2:
#                 self.model_dict.update({f"nn{hi}": torch.nn.Linear(self.hidden_nodes[hi], self.hidden_nodes[hi+1])})

#     def forward(self, x, training=True):
#         if training is True:
#             for layer_name, layer in zip(self.model_dict.keys(), self.model):
#                 x = layer(x)
#                 self.predicts[layer_name] = x
            
#         elif training is False:
#             with torch.no_grad():
#                 for layer_name, layer in zip(self.model_dict.keys(), self.model):
#                     x = layer(x)
#                     self.predicts[layer_name] = x
#         return x

#     def predict(self, x):
#         return self.forward(x, training=False)
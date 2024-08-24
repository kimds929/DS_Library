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
    def __init__(self, model, device='cpu'):
        self.now_date = datetime.strftime(datetime.now(), '%y%m%d_%H')

        self.model = model.to(device)
        self.device = device
        self.t = 1

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

    def compile(self, optimizer, loss_function, metric_function=None, scheduler=None,
                early_stop_loss=None, early_stop_metrics=None):
        """
        loss_function(model, x, y) -> loss
        """
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics_function = metric_function
        self.scheduler = scheduler
        self.early_stop_loss = early_stop_loss
        self.early_stop_metrics = early_stop_metrics

    def recompile(self, optimizer=None, loss_function=None, metric_function=None, scheduler=None,
                early_stop_loss=None, early_stop_metrics=None):
        if scheduler is not None:
            self.scheduler = scheduler
            self.scheduler.optimizer = self.optimizer

        if optimizer is not None:
            self.optimizer = optimizer
            self.scheduler.optimizer = self.optimizer

        if loss_function is not None:
            self.loss_function = loss_function
        
        if metric_function is not None:
            self.metrics_function = metric_function

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

    def train_model(self, train_loader, valid_loader=None, epochs=10, tqdm_display=False,
                early_stop=True, save_parameters=False, display_earlystop_result=False):
        final_epcohs = self.t + epochs - 1
        # [START of Epochs Loop] ############################################################################################
        epochs_iter = tqdm(range(self.t, self.t + epochs), desc="Epochs", total=epochs, position=0, leave=True) if tqdm_display else range(self.t, self.t + epochs)
        for epoch in epochs_iter:
            print_info = {}

            # train Loop --------------------------------------------------------------
            self.model.train()
            train_epoch_loss = []
            train_epoch_metrics = []
            train_iter = tqdm(enumerate(train_loader), desc="Train Batch", total=len(train_loader), position=1, leave=False) if tqdm_display else enumerate(train_loader)
            for batch_idx, (batch_x, batch_y) in train_iter:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_function(self.model, batch_x, batch_y)
                loss.backward()
                self.optimizer.step()
            
                with torch.no_grad():
                    train_epoch_loss.append( loss.to('cpu').detach().numpy() )
                    if self.metrics_function is not None:
                        train_epoch_metrics.append( self.metric_f(self.model, batch_x, batch_y) )

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
                    for batch_idx, (batch_x, batch_y) in valid_iter:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        
                        loss = self.loss_function(self.model, batch_x, batch_y)
                    
                        valid_epoch_loss.append( loss.to('cpu').detach().numpy() )
                        if self.metrics_function is not None:
                            valid_epoch_metrics.append( self.metric_f(self.model, batch_x, batch_y) )

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
                    params = self.model.state_dict() if save_parameters else None
                    early_stop_TF = self.early_stop_loss.early_stop(score=score, reference_score=reference_score,save=params, verbose=0)

                    if save_parameters:
                        path_save_loss = f"{self.get_save_path()}_earlystop_loss.pth"
                        cPickle.dump(self.early_stop_loss, open(path_save_loss, 'wb'))      # save earlystop loss

                if self.metrics_function is not None and self.early_stop_metrics is not None:
                    score = print_info['valid_metrics'] if (valid_loader is not None and len(valid_loader) > 0) else print_info['train_metrics']
                    reference_score = print_info['train_metrics'] if (valid_loader is not None and len(valid_loader) > 0) else None
                    params = self.model.state_dict() if save_parameters else None
                    self.early_stop_loss.early_stop(score=score, reference_score=reference_score, save=params, verbose=0)

                    if save_parameters:
                        path_save_metrics = f"{self.get_save_path()}_earlystop_metrics.pth"
                        cPickle.dump(self.early_stop_metrics, open(path_save_metrics, 'wb'))      # save earlystop metrics

                # save_parameters ---------------------------------------------------------
                if save_parameters:
                    path_save_weight = f"{self.get_save_path()}_weights.pth"
                    cPickle.dump(self.model.state_dict(), open(path_save_weight, 'wb'))      # save earlystop weights

                # step update ---------------------------------------------------------
                self.t += 1

                # early_stop break ---------------------------------------------------------
                if early_stop is True and early_stop_TF == 'break':
                        break
        
        if display_earlystop_result:
            if self.early_stop_loss is not None:
                display(self.early_stop_loss.plot)
            if self.metrics_function is not None and self.early_stop_metrics is not None:
                display(self.early_stop_metrics.plot)
        # [END of Epochs Loop] ############################################################################################

    def test_model(self, test_loader, tqdm_display=False):
        with torch.no_grad():
            print_info = {"epoch":self.t-1}
            # test Loop ---------------------------------------------------------
            if test_loader is not None and len(test_loader) > 0:
                self.model.eval()
                test_epoch_loss = []
                test_epoch_metrics = []
                test_iter = tqdm(enumerate(test_loader), desc="Valid Batch", total=len(test_loader), position=1, leave=False) if tqdm_display else enumerate(test_loader)
                for batch_idx, (batch_x, batch_y) in test_iter:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    loss = self.loss_function(self.model, batch_x, batch_y)
                
                    test_epoch_loss.append( loss.to('cpu').detach().numpy() )
                    if self.metrics_function is not None:
                        test_epoch_metrics.append( self.metric_f(self.model, batch_x, batch_y) )

                print_info['test_loss'] = np.mean(test_epoch_loss)
                self.test_losses.append(print_info['test_loss'])
                if self.metrics_function is not None:
                    print_info['test_metrics'] = np.mean(test_epoch_metrics)
                    self.test_metrics.append(print_info['test_metrics'])
            print_sentences = ",  ".join([f"{k}: {str(self.fun_decimal_point(v))}" for k, v in print_info.items() if k != 'epoch'])
            print(f"[After {self.t-1} epoch test performances] {print_sentences}")
            self.test_info.append(print_info)
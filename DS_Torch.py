# DataSet ########################################################################
# import numpy as np
# import torch
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
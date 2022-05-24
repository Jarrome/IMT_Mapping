import numpy as np
import torch
import pdb

class Indexer:
    '''
        key are nparray: [k,]
        value are nparray: [k,]
    '''
    def __init__(self):
        self.indexer = dict()
        self.shape = [0]

    def __getitem__(self, key):
        '''
            returns -1 if not in

        '''
        key_ = key.cpu().numpy()
        value = np.vectorize(self.indexer.get)(key_,-1.)
        return torch.Tensor(value).to(key)

    def __setitem__(self, key, value):
        key = key.cpu().int()
        if type(value) == torch.Tensor:
            value = value.cpu()
        else:
            value = torch.ones_like(key)*value

        self.expand(key,value.int())


    def expand(self, key, value):
        key = key.cpu().int().numpy()
        value = value.cpu()

        #new_dict = dict(np.stack([key,value],axis=1))
        new_dict = dict(zip(key,value))
        self.indexer.update(new_dict)
        self.shape = [len(self.indexer)]

        #self.indexer = {**self.indexer, **new_dict}
        #self.shape = [len(self.indexer)]

    def remove(self, key):
        key = key.cpu().int().numpy()

        np.vectorize(self.indexer.pop)(key,-1)

        self.shape = [len(self.indexer)]

    def keys(self):
        keys = np.fromiter(self.indexer.keys(), dtype=int)
        return keys
        
    def clear(self):
        self.indexer.clear()

        self.shape = [0]

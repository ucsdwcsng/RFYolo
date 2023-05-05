from numpy import dtype
import torch


class ToTensor(object):
    def __init__(self):
        pass;
    
    def __call__(self, tensor):
        return torch.Tensor(tensor);
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
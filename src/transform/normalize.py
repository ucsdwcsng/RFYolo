import torch


class Normalize(object):
    def __init__(self, mean=0., std=1.):
        self.std = torch.Tensor(std);
        self.mean = mean
        
    def __call__(self, tensor):
        # data_mean,data_std = torch.std_mean(tensor,dim=0);
        tensor = (tensor + self.mean)/self.std;
        return tensor;
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
import torch
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import datetime

class StdOutLogger(object):
    def __init__(self, logpath, log_filename=None):
        self.logpath = logpath;
        self.log_filename = log_filename;
        
        if log_filename is None: 
            fname='logs_%s.txt' % datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        else:
            fname='logs_'+log_filename+'.txt'
        
        self.terminal = sys.stdout
        self.log = open(f'{self.logpath}/{fname}', "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.terminal.flush();
        self.log.flush();
        pass 
    
class TBLogger(object):    
    def __init__(self,logpath, log_filename=None) -> None:
        self.logpath = logpath;
        self.log_filename = log_filename;
        self.writer = SummaryWriter(logpath + '/' + ("tblog_"+ log_filename if log_filename else "tblog"));
        self.global_step=0;
        
    def add_histogram(self,values,name,step=0):
        self.writer.add_histogram(f"{name}",values,self.global_step+step);
    
    def add_scalars(self,traces,name,step=0):
        self.writer.add_scalars(f"{name}",traces,self.global_step+step);
        
    def add_weight_dist(self,w,name,step=0):
        for layer in w.keys():
            self.writer.add_histogram(f"{name}/{layer}",w[layer],self.global_step+step);
            
    def add_graph(self,net,input,verbose=False):
        self.writer.add_graph(net,input,verbose=verbose);
    
    def add_embedding(self,name,feature,labels=None,image=None,step=0):
        self.writer.add_embedding(feature,labels,image,self.global_step+step,tag=name);
        
    def add_imagegrid(self,name,image_grid,step=0):
        self.writer.add_image(name, image_grid,self.global_step+step);
    
    def save_weights(self,name,weights,step=0):
        _steps=self.global_step+step;
        torch.save(weights, f'{self.logpath}/{name}_{_steps}.pt');
        

def get_logpath(args):
    logpath = args.logdir + '/' + args.exp_label + '/' + args.dataset + '/' + args.task + '/' + args.model + '/' + args.log_filename + '/';
    return logpath;
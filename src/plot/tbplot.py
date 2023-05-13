import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

import os
from os.path import isfile, join, isdir

import numpy as np
from scipy.signal import savgol_filter

from src.utils import mkdir

def get_trace(path, smooth=0):
    
    trace=[];
    # trace_sm=[[]];
    
    files = [f for f in os.listdir(path) if isfile(join(path, f))];
    
    event_file= files[-1];
    events = LegacyEventFileLoader(f'{path}/{event_file}').Load();
    trace_name=f'test_accuracy/';
    for e in events:
        for v in e.summary.value:
            # print(f'{v.tag}');
            if v.tag == trace_name:
                # print(e.step, v.simple_value)
                trace.append(v.simple_value)
    
    trace_sm = savgol_filter(trace, smooth, 3) # window size, polynomial order 3    
    
    return trace_sm;

def calc_mean_std(trace):
    trace_mean=[];trace_std=[];
    if len(trace) ==3:
        trace[2] = [];
    min_rounds = np.min([len(trace[0]), len(trace[1]), len(trace[2])]);
    argmin_rounds = np.argmin([len(trace[0]), len(trace[1]), len(trace[2])]);
    
    if min_rounds < 80:
        # indx = [True]*3;
        # indx[argmin_rounds] = False;
        # trace_mean[argmin_rounds] = [np.mean([trace_mean[i] for i in np.where(indx)]) for j in range];
        # trace_std[argmin_rounds] = [np.mean([trace_std[i] for i in np.where(indx)]) for j in range];
        _trace_mean=[];
        min_rounds = np.min([len(trace[0]), len(trace[1])]);
        for r in range(min_rounds):
            point=[trace[0][r],trace[1][r]];
            _trace_mean.append(np.mean(point));
        trace[argmin_rounds] = _trace_mean;
    
    min_rounds = np.min([len(trace[0]), len(trace[1]), len(trace[2])]);
    
    for r in range(min_rounds):
        point=[trace[0][r],trace[1][r],trace[2][r]];
        trace_mean.append(np.mean(point));
        trace_std.append(np.std(point));
        
    return np.array(trace_mean),np.array(trace_std);


# plt.tick_params(labelsize=14);


def tbplot_s(exp_id, datasets, tasks, models, logfiles, savepath='plots', smooth=9, legend=None, xlabel=None, ylabel=None, xlim=None, ylim=None, log_basepath='log'):
    count=0;
    plt.figure(figsize=(4,3));
    for d in datasets:
        for t in tasks:
            for m in models:
                for lf in logfiles:                              
                    tbpath = f"{log_basepath}/{exp_id}/{d}/{t}/{m}/{lf}/tblog_{lf}/test_accuracy__accuracy";
                    trace = get_trace(tbpath,smooth=smooth);
                    r=[i for i in range(len(trace))];
                    
                    if legend:
                        assert(len(legend) == len(datasets)*len(tasks)*len(models)*len(logfiles));
                        l=legend[count];
                    else:
                        l = f'';
                        itemsets = [datasets,tasks,models,logfiles];
                        item = [d,t,m,lf];
                        for itemset_i in range(len(itemsets)):
                            itemset = itemsets[itemset_i];
                            if len(itemset) != 1:
                                l = f'{l}-{item[itemset_i]}';
                    
                    plt.plot(r, trace, linewidth=3.0 ,label=l);
                    if xlim:
                        plt.xlim(xlim);
                    if ylim:
                        plt.xlim(ylim);
                    if xlabel:
                        plt.xlabel(xlabel);
                    if ylabel:
                        plt.ylabel(ylabel);
                    
                    count+=1;
                    
                    plt.show();
    plt.legend();
    
    pngfile = f'';
    itemsets = [datasets,tasks,models,logfiles];
    for itemset_i in range(len(itemsets)):
        itemset = itemsets[itemset_i];
        if len(itemset) == 1:
            pngfile = f'{pngfile}-{itemset[0]}';
    
    
    mkdir(savepath);
    plt.savefig(f"{savepath}/{pngfile}.png",dpi=300, bbox_inches="tight");
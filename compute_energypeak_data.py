import os

import torch
import argparse

from src.datagen import EnergyPeakData, get_fcfs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/temp', help='Path to dataset')
    parser.add_argument('--cls', type=str, default='', help='subcategory to process')
    opt = parser.parse_args()
        
    return opt


def main(opt):
    
    datadir=opt.datadir;
    nChannel=1024;
    nProto=64;

    nstft=1024;
    mfft=1024;
    window_fn=torch.blackman_window;
    overlapf=1;
    k=95;
    # k=24;

    kernel_sizes = [1,4,16,64,128];
    cScale = (-30, 5);  # in dB
    # cScale = None;  # in dB
    isLog = True;
    
    cls = opt.cls;
    if cls:
        dirs = cls.split(':');
    else:
        dirs = [d for d in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, d))]
        
    for d in dirs:        
        print(f"Dir: {d}");
        subdatadir = f'{datadir}/{d}';
        
        fc,fs = get_fcfs(subdatadir);
        # epGenerator = EnergyPeakData(subdatadir,
        #                                     kernel_sizes = kernel_sizes,
        #                                     cScale = cScale,
        #                                     isLog = isLog,
        #                                     nChannel = nChannel,
        #                                     nProto = nProto,
        #                                     fs=fs
        #                                  );
        epGenerator = EnergyPeakData(subdatadir,
                                            kernel_sizes = kernel_sizes,
                                            cScale = cScale,
                                            isLog = isLog,
                                            nstft = nstft,
                                            mfft = mfft,
                                            window_fn = window_fn,
                                            overlapf = overlapf,
                                            k = k,                                            
                                            fs=fs
                                         );
        epGenerator.generate_energypeak_data();
    
    pass


if __name__ == '__main__':
    opt = parse_args();
    
    main(opt);
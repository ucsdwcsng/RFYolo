import os
import torch
import argparse

from src.datagen.multiChannelSpectrum import spectralData
from src.datagen import get_fcfs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/temp', help='Path to dataset')
    parser.add_argument('--cls', type=str, default='', help='subcategory to process')
    opt = parser.parse_args()
        
    return opt

def main(opt):
    
    datadir=opt.datadir;
    nstft=1024;
    mfft=1024;
    window_fn=torch.blackman_window;
    overlapf=1;
    k=24;
    
    cls = opt.cls;
    if cls:
        dirs = cls.split(':');
    else:
        dirs = [d for d in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, d))]
        
    for d in dirs:        
        print(f"Dir: {d}");
        subdatadir = f'{datadir}/{d}';
                
        fc,fs = get_fcfs(subdatadir);
        
        spectralGenerator = spectralData(subdatadir,fs=fs, 
                                                    nstft=nstft,
                                                    mfft=mfft,
                                                    window_fn=window_fn,
                                                    overlapf=overlapf,
                                                    k=k
                                         );
        spectralGenerator.generate_spectral_data();




if __name__ == '__main__':
    opt = parse_args();
    
    main(opt);
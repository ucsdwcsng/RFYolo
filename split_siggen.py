import os
import argparse

from src.datagen.multiChannelSpectrum import segment_largeiq



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/temp', help='Path to dataset')
    parser.add_argument('--cls', type=str, default='', help='subcategory to process')
    opt = parser.parse_args()
    
    return opt

def main(opt):
        
    tseg=1e-3+1e-5;
    datadir=opt.datadir;    
    
    cls = opt.cls;
    if cls:
        dirs = cls.split(':');
    else:
        dirs = [d for d in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, d))]
    
        
    for d in dirs:        
        print(f"Dir: {d}");
        subdatadir = f'{datadir}/{d}';
        
        iqfiles = [f for f in os.listdir(subdatadir) if f.endswith(".32cf")];
        iqfile=iqfiles[0];
        
        fname, _ = os.path.splitext(iqfile);
        metafile=f'{fname}.json';
        
        segment_largeiq(subdatadir,iqfile,metafile,tseg);



if __name__ == '__main__':
    opt = parse_args();
    
    main(opt);
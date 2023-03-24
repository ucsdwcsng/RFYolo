import argparse

from src.datagen.multiChannelSpectrum import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/temp', help='Path to dataset')
    # parser.add_argument('--cls', type=str, default='', help='subcategory to process')
    opt = parser.parse_args()
        
    return opt

def main(opt):
    
    datadir=opt.datadir;
    
    dirs = [d for d in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, d))]
    
    i=0;
    for d in dirs:        
        print(f"Dir: {d}");
        subdatadir = f'{datadir}/{d}';
        
        lblGenerator = boxData(subdatadir);
        lblGenerator.generate_label_data(i);
        i+=1;




if __name__ == '__main__':
    opt = parse_args();
    
    main(opt);
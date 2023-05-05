import os
import json


def get_fcfs(subdatadir):
        iqfiles = [f for f in os.listdir(subdatadir) if f.endswith(".32cf")];
        iqfile=iqfiles[0];
        fname, _ = os.path.splitext(iqfile);
        
        metafile=f'{fname}.json';
                
        fc,fs = load_dataset_param(subdatadir,metafile);
        return fc,fs;
            
def load_dataset_param(datadir,metafile):
    with open(f'{datadir}/{metafile}','r') as metafd:
        meta = json.load(metafd);
        fs = meta['txVec']['samplingRate_Hz'];
        fc = meta['rxObj']['freqCenter_Hz'];
    
    return fc,fs;
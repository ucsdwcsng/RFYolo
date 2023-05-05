import torch
import numpy as np

from PIL import Image

import os
import shutil
import json

import torch.nn.functional as F
import math

import matplotlib.colors as colors
import matplotlib.cm as cm

from src.utils import mkdir, Mag2DB
from src.channelizer import PolyphaseChannelizer, STFT

class EnergyPeakView:    
    def __init__(self, **kwargs):
        # self.channelizer = PolyphaseChannelizer(**kwargs);      
        self.channelizer = STFT(**kwargs);
        
        self.chanIQ = [];
        self.chanMag = [];
        
    def compute(self, iq):
        # self.reset();
        # if len(self.chanIQ) == 0:
        self.chanIQ = self.channelizer.process(iq);
        self.chanMag = abs(self.chanIQ);
    
    # def reset(self):
    #     self.chanIQ           = [];
        
    def convEnergy(self,kernel_size=1):
        kernel = torch.ones((kernel_size,kernel_size),dtype=self.chanMag.dtype,device=self.chanMag.device)/(kernel_size**2);
        padding = math.floor((kernel_size - 1) / 2);
        enePeak = F.conv2d(self.chanMag.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0),padding=padding);
        
        return enePeak.squeeze();
        
    
    
class EnergyPeakData(EnergyPeakView):    
    def __init__(self, datadir, kernel_sizes=[1], cScale=None, isLog=False, **kwargs) -> None:
        super().__init__(**kwargs);
        self.datadir = datadir;
        self.cScale = cScale;
        self.isLog = isLog;
        
        self.kernel_sizes = kernel_sizes;
                
        
    def generate_energypeak_data(self):
        
        fext = ".32cf"
        iqfiles = [f for f in os.listdir(f'{self.datadir}/iq') if f.endswith(fext)];
        
        mkdir(f'{self.datadir}/energy_conv');
        
        cmap = cm.get_cmap('jet');        
        normalise = colors.Normalize(self.cScale[0],self.cScale[1]) if self.cScale else colors.Normalize();
        
        i=0;
        for f in iqfiles:
            fname, _ = os.path.splitext(f);
            print(f"{i}: {fname}");            
            
            for k in self.kernel_sizes:
                # Check if file already processed
                if os.path.exists(f'{self.datadir}/energy_conv/{fname}.k{k}.png'):
                    continue;
            
                array = np.fromfile(f'{self.datadir}/iq/{f}', dtype=np.complex64).astype(np.complex128);
                iq = torch.from_numpy(array).cuda();
                
                with torch.no_grad():
                    self.compute(iq);       #channelize
                    enePeaks_k = self.convEnergy(kernel_size=k);
                    if self.isLog:
                        enePeaks_k = Mag2DB(enePeaks_k);                    
                    
                    
                ep_cmap = np.transpose(cmap(normalise(enePeaks_k.cpu().numpy()))*255,(1,0,2));
                img_ep = Image.fromarray(np.uint8(ep_cmap[:, :, :3]));
                img_ep.save(f'{self.datadir}/energy_conv/{fname}.k{k}.png');
            
            i+=1;
        
            
            
             
    
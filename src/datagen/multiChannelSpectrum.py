import torch
import numpy as np

from PIL import Image
# import imageio
# import imageio.plugins.ffmpeg as ffmpeg
# ffmpeg.get_ffmpeg_version(threads=64)

import os
import shutil
import json

import math

from src.utils import wrap_to_pi, mkdir

import matplotlib.colors as colors
import matplotlib.cm as cm

class spectralView:
    
    def __init__(self, fs, nstft=1024, mfft=1024, window_fn=torch.blackman_window, overlapf=1, k=1) -> None:
        self.fs             =   fs;
        self.nstft          =   nstft;
        self.mfft            =   mfft;
        self.overlapf        =   overlapf;
        self.k               =   k;
        
        self.window_fn          =   window_fn;
        
        self.stft           = [];
        self.nb             = [];
        
        self.mag = []; self.logmag = [];
        self.ang = [];
        self.ang_nom = [];
        self.ang_cor = [];
        self.ang_nom_cor = [];
        
        
    def compute(self, iq):
        self.reset();
        self.stft = torch.stft(iq, self.mfft, hop_length=round((1-self.overlapf)*self.nstft+self.k), win_length=self.nstft, \
            pad_mode='constant', normalized=True, window=self.window_fn(self.nstft), return_complex=True, center=False);
        self.stft = torch.fft.fftshift(self.stft,dim=0);
        self.nb = len(iq);
    
    def reset(self):
        self.stft           = [];
        self.nb             = [];
        
        self.mag = []; self.logmag = [];
        self.ang = [];
        self.ang_nom = [];
        self.ang_cor = [];
        self.ang_nom_cor = [];
    
    def magSpectrum(self):
        if len(self.mag) == 0:
            self.mag = torch.abs(self.stft)
            self.logmag = 10*torch.log10(self.mag);        
        return self.mag,self.logmag;
    
    def angSpectrum(self):
        if len(self.ang) == 0:
            self.ang = torch.angle(self.stft);        
        return self.ang;
    
    def angSpectrum_nom(self):
        if len(self.ang_nom) == 0:
            mag,_=self.magSpectrum();
            ang=self.angSpectrum();
            
            self.ang_nom = mag*ang;
        
        return self.ang_nom;
    
    def angSpectrum_cor(self):        
        if len(self.ang_cor) == 0:
            a = self.angSpectrum();
            maa = self.angSpectrum_pred();
            
            self.ang_cor = wrap_to_pi(a - maa);
        
        return self.ang_cor;
        
    def angSpectrum_pred(self):
        device = self.stft.device;
        pi = torch.tensor(np.pi,device=device);
        
        # Compute frequency and time information
        t = torch.arange(0, self.stft.shape[-1],dtype=torch.float64, device=device)*self.k;
        f = torch.fft.fftshift(torch.fft.fftfreq(self.nstft,dtype=torch.float64, device=device));
        # f = (torch.fft.fftfreq(self.nstft,dtype=torch.float64, device=device));
        
        f = (-f).flip(dims=(0,));
        w = 2*pi*f;

        w=w.unsqueeze(1);
        t=t.unsqueeze(0);
        aa = w @ t;
        maa = wrap_to_pi(aa);
        
        return maa;        
        
    def angSpectrum_nom_cor(self):
        mag,_=self.magSpectrum();
        ang_cor=self.angSpectrum_cor();
        
        if len(self.ang_nom_cor) == 0:
            self.ang_nom_cor = mag*ang_cor;
        
        return self.ang_nom_cor;


class boxData:
    def __init__(self, datadir) -> None:        
        self.datadir = datadir;
    
    def generate_label_data(self, lbl_id) -> None:
        fext = ".json"
        metafiles = [f for f in os.listdir(f'{self.datadir}/meta') if f.endswith(fext)];
        
        mkdir(f'{self.datadir}/labels');
        
        for f in metafiles:
            fname, _ = os.path.splitext(f);
            # Check if file already processed
            exists= os.path.exists(f'{self.datadir}/labels/{fname}.txt');
            if exists:
                # continue;
                pass;
            
            with open(f'{self.datadir}/meta/{f}','r') as meta_ifd:
                meta = json.load(meta_ifd);
                
            with open(f'{self.datadir}/labels/{fname}.txt','w') as meta_ofd:
                for i in range(len(meta)):
                    e = meta[i];
                    meta_ofd.write(f'{lbl_id} {e[0]} {e[1]} {e[2]} {e[3]}\n');
                


class spectralData(spectralView):    
    def __init__(self, datadir, **kwargs) -> None:
        super().__init__(**kwargs);
        self.datadir = datadir;
    
    def generate_spectral_data(self):
        
        fext = ".32cf"
        iqfiles = [f for f in os.listdir(f'{self.datadir}/iq') if f.endswith(fext)];
        
        mkdir(f'{self.datadir}/spectrum');
        
        i=0;
        for f in iqfiles:
            fname, _ = os.path.splitext(f);
            print(f"{i}: {fname}");            
            
            # Check if file already processed
            if os.path.exists(f'{self.datadir}/spectrum/{fname}.nca.png'):
                i+=1;
                continue;
            
            array = np.fromfile(f'{self.datadir}/iq/{f}', dtype=np.complex64).astype(np.complex128);
            iq = torch.from_numpy(array);
            
            self.compute(iq);
            
            _,s = self.magSpectrum();
            a = self.angSpectrum();
            na = self.angSpectrum_nom();
            maa = self.angSpectrum_pred();
            ca = self.angSpectrum_cor();
            nca = self.angSpectrum_nom_cor();
            
            cmap = cm.get_cmap('jet');
            
            s_cmap = np.transpose(cmap(colors.Normalize()(s.cpu().numpy()))*255,(1,0,2));
            a_cmap = np.transpose(cmap(colors.Normalize()(a.cpu().numpy()))*255,(1,0,2));
            maa_cmap = np.transpose(cmap(colors.Normalize()(maa.cpu().numpy()))*255,(1,0,2));
            na_cmap = np.transpose(cmap(colors.Normalize()(na.cpu().numpy()))*255,(1,0,2));
            ca_cmap = np.transpose(cmap(colors.Normalize()(ca.cpu().numpy()))*255,(1,0,2));
            nca_cmap = np.transpose(cmap(colors.Normalize()(nca.cpu().numpy()))*255,(1,0,2));
            
            img_s = Image.fromarray(np.uint8(s_cmap[:, :, :3]));
            img_a = Image.fromarray(np.uint8(a_cmap[:, :, :3]));
            img_maa = Image.fromarray(np.uint8(maa_cmap[:, :, :3]));
            img_na = Image.fromarray(np.uint8(na_cmap[:, :, :3]));
            img_ca = Image.fromarray(np.uint8(ca_cmap[:, :, :3]));
            img_nca = Image.fromarray(np.uint8(nca_cmap[:, :, :3]));
            
            img_s.save(f'{self.datadir}/spectrum/{fname}.s.png');
            img_a.save(f'{self.datadir}/spectrum/{fname}.a.png');
            # img_maa.save(f'{self.datadir}/spectrum/{fname}.maa.png');
            img_na.save(f'{self.datadir}/spectrum/{fname}.na.png');
            img_ca.save(f'{self.datadir}/spectrum/{fname}.ca.png');
            img_nca.save(f'{self.datadir}/spectrum/{fname}.nca.png');
            
            i+=1;
            
            
            
    
def segment_largeiq(inpdatadir, iqfile, metafile, tseg):
        
    boxes = parseMetadata(inpdatadir,metafile);
    fc,fs = load_dataset_param(inpdatadir,metafile);
    boxes = time_segment_boxes_n_norm(boxes,tseg,fc,fs);
    
    mkdir(f'{inpdatadir}/iq');
    mkdir(f'{inpdatadir}/meta');
    
    nb = round(fs*tseg);
    with open(f'{inpdatadir}/{iqfile}', "rb") as iqfd:
        # Iterate over chunks of n samples
        cnt=0;
        for _, chunk in enumerate(iter(lambda: iqfd.read(nb*8), b'')):
            # Convert binary data to complex64 numpy array
            data = np.frombuffer(chunk, dtype=np.complex64)
            # Write data to output file
            fname, fext = os.path.splitext(iqfile);
            output_file = f"{inpdatadir}/iq/{fname}_{cnt}{fext}";
            
            exists = os.path.exists(output_file);
            if exists:
                cnt+=1;
                continue;
            
            with open(output_file, "wb") as iq_of:
                iq_of.write(data.tobytes())
            
            # Metadata file
            with open(f'{inpdatadir}/meta/{fname}_{cnt}.json', 'w') as meta_of:
                json.dump(boxes[cnt], meta_of);
            
            cnt+=1;
                    
    
def time_segment_boxes_n_norm(boxes,tseg,fc,fs):
    
    tmax = max(boxes, key=lambda b:b.y2).y2;
    tmin = min(boxes, key=lambda b:b.y2).y2;
    
    fmin = fc - fs/2;
    fmax = fc + fs/2;
    
    segBoxes = [];
    for tsegi in range(math.ceil(tmax/tseg)):
        tseg_min = tsegi*tseg;
        tseg_max = (tsegi+1)*tseg;
        radio_box = Box2D(0,tseg_min,float('inf'),tseg_max,-1);            
                
        boxes_seg = [];
        for box in boxes:
            # If the input rectangle intersects with the boundary rectangle
            if box.intersects(radio_box):
                # Crop the input rectangle to the boundary rectangle
                box_c = box.clip(radio_box)
                # Add the cropped input rectangle to the output rectangles
                
                x,y = (box_c.x2+box_c.x1)/2,(box_c.y2+box_c.y1)/2;
                w,h = (box_c.x2-box_c.x1)  ,(box_c.y2-box_c.y1)  ;
                
                nx,ny = (x-fmin)/fs, (y-tseg_min)/tseg;
                nw,nh = w/fs       , h/tseg           ;
                
                if nw !=0 and nh !=0:
                    boxes_seg.append([nx,ny,nw,nh,box_c.id]);
        segBoxes.append(boxes_seg);
        
    return segBoxes;
        

def parseMetadata(datadir, metafile):
    with open(f'{datadir}/{metafile}','r') as metafd:
        meta = json.load(metafd);
        arrSig = meta['signalArray'];
        
        boxes = [];
        for i in range(len(arrSig)):
            sig = arrSig[i];
            sig_fc = sig['requiredMetadata']['reference_freq'];
            rad_fc = sig['centerFreq_Hz'];
            
            arrEne = sig['transmissionArray'];
            for j in range(len(arrEne)):
                ene = arrEne[j];0
                x1,y1 = ene['freq_lo'],ene['time_start'];
                x2,y2 = ene['freq_hi'],ene['time_stop'];
                
                box = Box2D(x1,y1,x2,y2, i);
                boxes.append(box);
    
    return boxes;
            
    
def load_dataset_param(datadir,metafile):
    with open(f'{datadir}/{metafile}','r') as metafd:
        meta = json.load(metafd);
        fs = meta['txVec']['samplingRate_Hz'];
        fc = meta['rxObj']['freqCenter_Hz'];
    
    return fc,fs;
    
def process_recursive(self):
    
    for root, dirs, files in os.walk(self.iqdir):
        # Create the corresponding output subdirectory in the output directory
        rel_path = os.path.relpath(root, self.iqdir)
        output_subdir = os.path.join(self.pngdir, rel_path)
        mkdir(output_subdir, exist_ok=True)

        # Process each file in the current subdirectory and store it in the output directory
        for filename in files:
            input_file = os.path.join(root, filename)
            output_file = os.path.join(output_subdir, filename)
            # Process the input file and store the result in the output file
            # For example:
            shutil.copyfile(input_file, output_file)
    
class Box2D:
    def __init__(self, x1, y1, x2, y2, id):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2;
        self.id = id;

    def intersects(self, other):
        # Get the coordinates of the rectangle and the frame
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2;
        fx1, fy1, fx2, fy2 = other.x1, other.y1, other.x2, other.y2;

        # Calculate the intersection between the rectangle and the frame
        ix1 = max(x1, fx1)
        iy1 = max(y1, fy1)
        ix2 = min(x2, fx2)
        iy2 = min(y2, fy2)

        # If there is no intersection, return False
        if ix1 > ix2 or iy1 > iy2:
            return False

        # Otherwise, return True
        return True

    def clip(self, other):
        
        _x1 = max(self.x1, other.x1)
        _y1 = max(self.y1, other.y1)
        _x2 = min(self.x2, other.x2)
        _y2 = min(self.y2, other.y2)
        
        return Box2D(_x1,_y1,_x2,_y2,self.id);
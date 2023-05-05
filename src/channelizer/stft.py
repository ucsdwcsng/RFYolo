import torch


class STFT:
    def __init__(self, fs, nstft=1024, mfft=1024, window_fn=torch.blackman_window, overlapf=1, k=1) -> None:
        self.fs             =   fs;
        self.nstft          =   nstft;
        self.mfft            =   mfft;
        self.overlapf        =   overlapf;
        self.k               =   k;
        
        self.window_fn          =   window_fn;

    def process(self, iq):
        stft = torch.stft(iq, self.mfft, hop_length=round((1-self.overlapf)*self.nstft+self.k), win_length=self.nstft, \
            pad_mode='constant', normalized=True, window=self.window_fn(self.nstft).to(device=iq.device), return_complex=True, center=False);
        stft = torch.fft.fftshift(stft,dim=0);
        nb = len(iq);
        return stft
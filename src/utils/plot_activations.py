import torch
import torchvision


def log_activation_maps(activations,logger,ids=None,step=0,acttype='act'):
    for name,val in activations.items():
        for i in range(len(val)):
            if ids:
                if i not in ids:
                    continue;
            val_chan = val[i].unsqueeze(1);
            # val_chan = normalize(val_chan);
            act_grid = torchvision.utils.make_grid(val_chan,nrow=16,normalize=True);
            logger.add_imagegrid(f'{i}/{name}/{acttype}',act_grid,step=step);



def normalize_grad(image):
    norm = (image - image.mean())/image.std()
    norm = norm * 0.1
    norm = norm + 0.5
    norm = norm.clip(0, 1)
    return norm

def normalize_act(image):
    norm = (image - image.mean())/image.std()
    norm = norm * 0.1
    norm = norm + 0.5
    norm = norm.clip(0, 1)
    return norm
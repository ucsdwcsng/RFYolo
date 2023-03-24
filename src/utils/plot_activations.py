import torch
import torchvision


def log_activation_maps(activations,logger,ids=None,step=0):
    for name,val in activations.items():
        for i in range(len(val)):
            if ids:
                if i not in ids:
                    continue;
            val_chan = val[i].unsqueeze(1);
            act_grid = torchvision.utils.make_grid(val_chan,nrow=16);
            logger.add_imagegrid(f'{i}/{name}',act_grid,step=step);

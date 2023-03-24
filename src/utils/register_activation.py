import torch
import torchvision
import magicattr

from .plot_activations import *

def get_activation(features,name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def register_extract_activation_hook(model, layers, activations):
    for name in layers:        
        magicattr.get(model, name).register_forward_hook(get_activation(activations,name));

def register_activation_hook(activations,model,model_name,dataset):
    
    if model_name in ('resnet50', 'resnet50-lrelu'):    
        layers=[
            'conv1',
            'layers[0][0].downsample[0]',
            'layers[0][0].downsample[2]',
            'layers[1][0].downsample[0]',
            'layers[1][0].downsample[2]',
            'layers[2][0].downsample[0]',
            'layers[2][0].downsample[2]',
            'layers[3][0].downsample[0]',
            'layers[3][0].downsample[2]',
            'layers[3][2].bn3'
        ];
    elif model_name in ('vgg16','vgg16-lrelu'):
        layers = [
            'features[2]',  
            'features[7]',  
            'features[14]',  
            'features[21]',  
            'features[28]'  
        ];
    elif model_name in ('vgg16_bn','vgg16_bn-lrelu'):
        layers = [
            'features[3]',  
            'features[4]',  
            'features[5]',  
            'features[10]',  
            'features[11]',  
            'features[12]',  
            'features[20]',  
            'features[21]',  
            'features[22]',  
            'features[30]',  
            'features[31]',  
            'features[32]',
            'features[40]',
            'features[41]',
            'features[42]',
        ];
    elif model_name in ('lenet5','lenet5-lrelu'):    
        layers = [
            'conv1',
            'conv2'
        ];
    elif model_name in ('yolov7-tiny'):
        layers=[
            'model[0].conv',
            'model[7].conv',
            'model[14].conv',
            'model[21].conv',
            'model[28].conv',
            'model[35].conv',
            'model[37].conv',
            'model[47].conv',
            'model[57].conv',
            'model[65].conv',
            'model[73].conv',
            'model[74].conv',
            'model[75].conv',
            'model[76].conv'
        ];
    register_extract_activation_hook(model,layers,activations);
    

def define_test_hook(activations,logger,device):
    def test_hook(x,step,ids=None):
        for i in range(len(x)):
            if ids:
                if i not in ids:
                    continue;
            inp_act_grid = torchvision.utils.make_grid(x[i],nrow=1);
            logger.add_imagegrid(f'{i}/original',inp_act_grid);

        log_activation_maps(activations,logger,ids,step);
               

    return test_hook;    
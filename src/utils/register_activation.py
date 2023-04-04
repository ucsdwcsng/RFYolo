import torch
import torchvision
import magicattr

from .plot_activations import *

def get_activation(features,name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def get_gradient(gradients,name):
    def hook(model, grad_in, grad_out):
        gradients[name] = grad_out[0].detach()
    return hook

def register_extract_activation_hook(model, layers, activations):
    for name in layers:        
        magicattr.get(model, name).register_forward_hook(get_activation(activations,name));

def register_extract_gradient_hook(model, layers, gradients):
    for name in layers:        
        magicattr.get(model, name).register_backward_hook(get_gradient(gradients,name));

def register_activation_hook(activations,gradients,model,model_name,dataset):
    
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
            'model[0].act',
            'model[7].act',
            'model[14].act',
            'model[21].act',
            'model[28].act',
            'model[35].act',
            'model[37].act',
            'model[47].act',
            'model[57].act',
            'model[65].act',
            'model[73].act',
            'model[74].act',
            'model[75].act',
            'model[76].act'
        ];
    register_extract_activation_hook(model,layers,activations);
    register_extract_gradient_hook(model,layers,gradients);
    

def define_test_hook(activations,gradients,logger,device):
    def test_hook(x,gradx,step,ids=None,batch_indx=0):
        for i in range(len(x)):
            if ids:
                if i not in ids:
                    continue;
            inp_act_grid = torchvision.utils.make_grid(x[i],nrow=1);
            logger.add_imagegrid(f'b{batch_indx}/d{i}/original/act',inp_act_grid);
            inp_grad_grid = torchvision.utils.make_grid(gradx[i],nrow=1);
            logger.add_imagegrid(f'b{batch_indx}/d{i}/original/grad',inp_grad_grid);

        log_activation_maps(activations,logger,ids,step,acttype='act');
        log_activation_maps(gradients,logger,ids,step,acttype='grad');
               

    return test_hook;    
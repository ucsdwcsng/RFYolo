import torch

class Guided_backprop():
    def __init__(self, model,nms_fn,nc):
        self.model = model
        self.image_reconstruction = None # store R0
        self.activation_maps = []  # store f1, f2, ... 
        self.model.eval()
        self.nms_fn = nms_fn;
        self.nc=nc;
        self.register_hooks()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            self.image_reconstruction = grad_in[0] 

        def forward_hook_fn(module, input, output):
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            grad = self.activation_maps.pop() 
            # for the forward pass, after the ReLU operation, 
            # if the output value is positive, we set the value to 1,
            # and if the output value is negative, we set it to 0.
            grad[grad > 0] = 1 
            
            # grad_out[0] stores the gradients for each feature map,
            # and we only retain the positive gradients
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            new_grad_in = positive_grad_out * grad

            return (new_grad_in,)


        # travese the modules，register forward hook & backward hook
        # for the ReLU
        for i in range(len(self.model.model)):
            if hasattr(self.model.model[i], "act"):
                self.model.model[i].act.register_forward_hook(forward_hook_fn)
                self.model.model[i].act.register_backward_hook(backward_hook_fn)

        # # register backward hook for the first conv layer
        first_layer = self.model.model[0].conv
        first_layer.register_backward_hook(first_layer_hook_fn)

    def visualize(self, input_image, target_class):
        
        # self.model.train();
        self.model.eval();
        
        bs = input_image.shape[0];
        
        # compute_loss_ota = ComputeLossOTA(self.model);
        
        
        input_image=input_image.requires_grad_();
        with torch.set_grad_enabled(True):
            # torch.autograd.gradcheck(lambda x : self.model(x)[0],(input_image));
            pred = self.model(input_image)[0];
            # pred = torch.cat((pred[0].view(bs,-1,self.nc+5),pred[1].view(bs,-1,self.nc+5),pred[2].view(bs,-1,self.nc+5)),1)
            # loss = compute_loss_ota(pred,input_image)
            _,pred_indx = self.nms_fn(pred);
            pred = pred[:,pred_indx[0].detach().cpu().numpy()];
        self.model.zero_grad()
        
        device=pred.device;
        
        if pred.numel()==0:
            self.image_reconstruction = torch.zeros_like(input_image,device=input_image.device);
            result = self.image_reconstruction
            return result
            
        pred_class = pred[..., 5:].argmax(dim=-1).item()
        
        grad_target_map = torch.zeros(pred.shape,dtype=torch.float,device=device);
        if target_class is not None:
            grad_target_map[..., target_class +5] = 1
        else:
            grad_target_map[..., pred_class +5] = 1
        
        pred.backward(grad_target_map)
        
        result = self.image_reconstruction
        return result



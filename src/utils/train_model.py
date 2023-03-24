import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .find_files import glob_re
import re

def train_model(net,train_dl,test_dl,scenario,
                       lr0=0.001,a=0.001,b=0.75,momentum=0.9,weight_decay=0.0001,num_epoch=10,batch_size=64,
                       loss_func = nn.CrossEntropyLoss(reduction='mean',label_smoothing=0.1),
                       logger=None,device=torch.device('cpu'),test_hook_once=None,step0=0,
                       resume=0):

    step=0;
    bs = batch_size
    N = len(train_dl.dataset)
    myiterations = (N//bs+1)*num_epoch
    lr_scheduler=exp_lr_decay(lr0,a,b);
        
    latest_epoch=0
    latest_minibatch=0
    if resume==1:
        print('-'*40)
        print('Resuming Training ...')
        if logger is None:
            print('Error: No logger specified ... cannot resume==1')
            print('Exitting ...')
            exit(1);
            
        logpath = logger.logpath;

        regex = f'.*checkpoint_{scenario}_(\d*).pt';
        checkpoint_files = glob_re(logpath,regex);
        ls_step =[int(re.match(regex,f).group(1)) for f in checkpoint_files];
        idx = np.argmax(ls_step);
        if checkpoint_files is None:
            print("Checkpoint not found ...")
            exit(1)
        
        latest_step = ls_step[idx];
        print(f'Checkpoint Found (steps: {latest_step})')
        latest_file = checkpoint_files[idx];
        checkpoint_weight = torch.load(f'{latest_file}');
        net.load_state_dict(checkpoint_weight);
        
        latest_epoch = int(latest_step/len(train_dl));
        latest_minibatch = latest_step%len(train_dl);
        
        step = latest_step;
    net.to(device);
    net.train();
            
    # net_param = [];
    # for name,param in net.named_parameters():
    #     if name[:6] == 'output':
    #         continue;
    #     net_param.append(param);
    
    # optimizer = torch.optim.SGD([
    #     {'params': net_param},
    #     {'params': net.fc.parameters()}], lr=lr0, momentum=momentum, weight_decay=weight_decay);
    optimizer = torch.optim.SGD(net.parameters(), lr=lr0, momentum=momentum, weight_decay=weight_decay, nesterov=True);
    # loss_func = nn.CrossEntropyLoss(reduction='mean',label_smoothing=0.1);
    
    # pg0,pg1,pg2=get_param_groups(net);
    # optimizer = torch.optim.SGD([
    #         {'params': pg0},  # add pg0 (other)
    #         {'params': pg1, 'weight_decay': weight_decay},  # add pg1 with weight_decay
    #         {'params': pg2}  # add pg2 (biases)
    #     ],lr=lr0, momentum=momentum, nesterov=True)
    
    loss_func = loss_func.to(device);
    
    test_loss,test_acc = test_model(net,test_dl,
                            loss_func,
                            device=device,
                            test_hook_once=test_hook_once);
    print(f'Initial \tTestLoss: {test_loss} \tTestAcc: {test_acc}')
    
    epoch_loss = [];
    for epoch in range(num_epoch):
        if resume==1:
            if epoch < latest_epoch:
                continue;
        print(f'----- EPOCH {epoch+1} -----')   
        
        pbar=enumerate(train_dl);
        pbar = tqdm(pbar,total=len(train_dl))  # progress bar
        batch_loss=[];batch_acc=[];
        for batch_idx, (data, labels) in pbar:
            if resume==1:
                if batch_idx < latest_minibatch:
                    continue;
                
            data, labels = data.to(device), labels.type(torch.LongTensor).to(device)
            
            net.zero_grad();
            output = net(data);
            loss = loss_func(output, labels)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            acc = 100*pred.eq(labels.data.view_as(pred)).long().sum()/batch_size;
            loss.backward()
            
            lr = lr_scheduler(step);
            optimizer.param_groups[0]['lr'] = lr;
            # optimizer.param_groups[1]['lr'] = 10*lr;
            optimizer.step()
            batch_loss.append(loss.item())
            batch_acc.append(acc)
            
            # print(f'lr: {lr} \ttrain_loss: {loss.item()}')
            mloss = sum(batch_acc)/len(batch_acc)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 3) % (
                '%g/%g' % (epoch, num_epoch - 1), mem, mloss, labels.shape[0], data.shape[-1])
            pbar.set_description(s)
            
            if logger:
                logger.add_scalars({"loss":loss.item()},f"train_loss/{scenario}",step=step+step0);
                logger.add_scalars({"acc":acc.item()},f"train_acc/{scenario}",step=step+step0);
            
            if step+1 in (np.array([0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])*myiterations).astype(int) or (step+1)%2000==0:
                net.eval();
                test_loss,test_acc = test_model(net,test_dl,
                                        loss_func,
                                        device=device,
                                        test_hook_once=test_hook_once,iter=step);
                net.train();
                print(f'lr: {lr} \tAvgTrainLoss: {sum(batch_loss)/len(batch_loss)} \tAvgTrainAcc: {sum(batch_acc)/len(batch_acc)} \tTestLoss: {test_loss} \tTestAcc: {test_acc}')
                if logger:
                    logger.save_weights(f"checkpoint_{scenario}",net.state_dict(),step=step+step0);
                    logger.add_weight_dist(net.state_dict(),f"{scenario}",step=step+step0);
                    logger.add_scalars({"loss":test_loss.item()},f"test_loss/{scenario}",step=step+step0);
                    logger.add_scalars({"accuracy":test_acc.item()},f"test_accuracy/{scenario}",step=step+step0);
            step+=1
        
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        print(f'AvgTrainLoss: {sum(batch_loss)/len(batch_loss)}')
    
    test_loss,test_acc = test_model(net,test_dl,
                            loss_func,
                            device=device,
                            test_hook_once=test_hook_once,
                            iter=step);
    print(f'Final \tTestLoss: {test_loss} \tTestAcc: {test_acc}')
    if logger:
        logger.save_weights(f"checkpoint_{scenario}",net.state_dict(),step=step+step0);
        logger.add_weight_dist(net.state_dict(),f"{scenario}",step=step+step0);
        logger.add_scalars({"loss":test_loss.item()},f"test_loss/{scenario}",step=step+step0);
        logger.add_scalars({"accuracy":test_acc.item()},f"test_accuracy/{scenario}",step=step+step0);
    return net.state_dict(), epoch_loss[-1], test_loss, step+step0


def exp_lr_decay(lr0,a,b):
    def exp_lr_fn(i):
        return lr0*((1+a*i)**(-1*b));
    return exp_lr_fn;

def test_model(net,test_dl,
                loss_func = nn.CrossEntropyLoss(reduction="sum"),
               device=torch.device('cpu'),test_hook_once=None,iter=0):
    net.to(device)
    net.eval()
    test_loss = torch.tensor(0.0,device=device);
    correct = torch.tensor(0.0,device=device);
    # loss_func = nn.CrossEntropyLoss(reduction="sum").to(device)
    loss_func = loss_func.to(device)
    
    print("Testing ...")
    with torch.no_grad():
        pbar=enumerate(test_dl)
        pbar = tqdm(pbar,total=len(test_dl))  # progress bar        
        for i,(data,target) in pbar:
            data, target = data.to(device), target.type(torch.LongTensor).to(device)
            output = net(data)
            test_loss += loss_func(output, target).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability            
            correct += pred.eq(target.data.view_as(pred)).long().sum()            
            if i == 1:
                if test_hook_once:
                    test_hook_once(data,iter,ids=[0]);
    test_loss /= len(test_dl.dataset)
    accuracy = 100. * correct / len(test_dl.dataset)
    return test_loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()


def get_param_groups(net):
    
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in net.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        else:
            pg0 += [p for p in v.parameters(recurse=False)];
    
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    
    return pg0,pg1,pg2;
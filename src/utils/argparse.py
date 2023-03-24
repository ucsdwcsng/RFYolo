import argparse
    
def args_parser():
    parser = argparse.ArgumentParser();
    
    # random seed
    parser.add_argument('--seed', type=int, default=202210, help="random number generator seed (default: 202210)")
    # train param
    parser.add_argument('--num_epoch', type=int, default=10, help="the number of epochs")
    parser.add_argument('--batch_size', type=int, default=10, help="batch size")
    parser.add_argument('--lr0', type=float, default=0.01, help="init learning rate")
    parser.add_argument('--lr_a', type=float, default=0.01, help="learning rate exp decay param a")
    parser.add_argument('--lr_b', type=float, default=0.01, help="learning rate exp decay param b")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument('--w_decay', type=float, default=0.5, help="weight decay")
    parser.add_argument('--lbl_sm', type=float, default=0.01, help="label smoothing parameter")

    # model arguments
    parser.add_argument('--model', type=str, default='latent-mapper', help="name of model to use")
    parser.add_argument('--wavelet', type=str, default='haar', help="Wavelet to use")
    # dataset arguments
    parser.add_argument('--dataset', type=str, default='siggen/2S', help="name of dataset")
    parser.add_argument('--num_workers', type=int, default=8, help="num_workers for dataloader")    
    # Task
    parser.add_argument('--task', type=str, default='feature_extract', help="Task to run")
    
    parser.add_argument('--init_weights', type=int, default=0, help="Load pretrained weights?")
    parser.add_argument('--resume_train', type=int, default=0, help="Resume training from interruption?")
    
    # parser.add_argument('--savedir', type=str, default='../save/', help='save directory')
    parser.add_argument('--datadir', type=str, default='../data/', help='data directory')
    parser.add_argument('--logdir', type=str, default='../logs/', help='logs directory')
    parser.add_argument('--ptdir', type=str, default='../pretrain/', help='pretrained model dir')
    parser.add_argument('--log_filename', type=str, help='The log file name')
        
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    
    # parser.add_argument('--load_initial', type=str, default='', help='define initial model path')
    parser.add_argument('--exp_label', type=str, default='test', help='Experiment Label')
    
    args = parser.parse_args();
    return args;
import os
import argparse
import time

import torch
from logger import create_logger
from transforms import Transforms
from const import SingletonString
from utils import argparse2bool, list_of_ints, load_model, mkdir, save_model, seed_everything


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
float_formatter = "{:.4f}".format


def parse_args():
    parser = argparse.ArgumentParser(description='Boundary unlearning main procedure')
    ######################### general parameters ################################
    parser.add_argument('--logname', default='trail', help='name of output log file')
    parser.add_argument('--log_dir', default='.', help='Folder where all logs are stored (default: .)')
    parser.add_argument('--out_dir', default='./outs/cifar10_-1_FGCLS-2/', help='Folder where all outputs are stored (default: .)')
    parser.add_argument('--exp_name', default='logs', help='Subfolder where all logs are stored (default: logs)')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--cuda', type=str, default=3, help="Choose the GPU device")
    # parser.add_argument('--verbose', type=argparse2bool, default=True, help='whether to print the log')
    
    ### dataset related parameters
    parser.add_argument('--dataset', type=str, default='cifar10', #
                        choices=['mnist', 'cifar10', 'cifar100', 'stl10', 'imagenet', 'vggface2', 'svhn', 'fashionmnist'],
                        help='dataset for training (default: cifar10)')
    parser.add_argument('--select_classes', type=list_of_ints, default='3,8', 
                        help='selected classes for small dataset (default: 3,8)')
    parser.add_argument('--num_samples', type=int, default=-1, metavar='N', 
                        help='number of samples to use (default: -1 (all))')
    parser.add_argument('--valid_size', type=int, default=10000, metavar='V', 
                        help='number of validation samples (default: 5000)')
    parser.add_argument('--num_to_forget', type=int, default=10000, metavar='Nf', 
                        help='number of samples to forget (default: 5000)')
    parser.add_argument('--forget_classes', type=list_of_ints, default=[2], metavar='Cf', 
                        help='classes to forget (default: None)')
        
    ## model related parameters
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=[# 'RF', 'DT', 'SVM', 'LR', 
                                'mlps', 'logistic', 'simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'densenet'],
                        help='model architecture (default: resnet18)')
    parser.add_argument('--pretrained', type=argparse2bool, default=True, help='whether to use pretrained model')
    
    parser.add_argument('--unlearn_method', type=str, default='boundary_shrink',  #'boundary_expand', #
                        choices=['boundary_shrink', 'boundary_expand'], help='unlearning method')
    parser.add_argument('--resume', type=str, default=None, help='path to the model to be unlearned')
    
    ######################### model (un-)learning related parameters ################################
    parser.add_argument('--lossfn', type=str, default='ce', choices=['ce', 'bce', 'mse'])
    parser.add_argument('--start_raw', type=argparse2bool, default=False, help='whether to start from raw sample')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 50)')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping (default: 30)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', 
                        help='input batch size for training (default: 256)')
    
    parser.add_argument('--optim', type=str, default="Adam", choices=['Adam', 'SGD'])
    parser.add_argument('--scheduler', default='None', #'CosineAnnealingWarmRestarts', 
                        choices = ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR', 'None'],
                        help='Pytorch Scheduler name: (default: The one used for train, in args_re')
    
    parser.add_argument('--maxlr', type=float, default=0.001, help='the maximum learning rate (default: 0.001)')
    parser.add_argument('--minlr', type=float, default=0.001, help='the minimum learning rate (default: 0.0001)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD (default: 0.9)')
    # parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay (default: 5e-5)')
    # parser.add_argument('--lamb', type=float, default=1e-7, help='regularization parameter (default: 1e-8)')

    ## model training with data augmentation or mixup
    parser.add_argument("--train_transform", type=str, default="normal", 
                        help="data augmentation for preprocessing the training data",
                        choices=["none", "normal", "random", "randerase", "augafn", "baseaug", 
                                 "cutout", "cutmix", "autoaug", "augmix", "randaug", "test"],)
    parser.add_argument("--test_transform", type=str, default="test", 
                        help="data augmentation for preprocessing the testing data",
                        choices=["none", "normal", "random", "randerase", "augafn", "baseaug", 
                                 "cutout", "cutmix", "autoaug", "augmix", "randaug", "test" ],)
    
    # parser.add_argument('--mixup', type=argparse2bool, default=False, help='using the mixup as the data augmentation')
    parser.add_argument('--mixup_type', default='none', 
                        choices = ['none', 'mixup', 'cutmix'], help='Online mixup type for data augmentation')
    parser.add_argument('--mixup_prob', type=float, default=0.5, metavar='P', 
                        help='Prob. for mixup (default: 0.5)')
    parser.add_argument('--mixup_alpha', default=1.0, type=float, help='mixup interpolation coefficient (default: 1.0)')

    args = parser.parse_args()
    return args


def construct_name(args):
    name = f"{args.dataset}_{args.arch}_seed-{args.seed}"
    if args.num_to_forget is not None:
        name += f"_Nf-{args.num_to_forget}"
    if args.forget_classes is not None:
        name += f"_Cf-{str(args.forget_classes)}"
    name+=f"_ep-{args.epochs}_bs-{args.batch_size}"
    name += f"_lr-[{str(args.minlr).replace('.','_')}-{str(args.maxlr).replace('.','_')}]"
    # name += f"_wd-{str(args.weight_decay).replace('.','_')}"
    name += f"_opt-{args.optim}"
    name += f"_trTrans-{args.train_transform}_tsTrans-{args.test_transform}"
    # if args.lamb is not None:
    #     name += f"_regular_{args.lamb}"
    # if args.mixup_type == "cutmix":
    #     name += f"_cmixp-{str(args.mixup_prob).replace('.','_')}"
    #     name += f"_cmixa-{str(args.mixup_alpha).replace('.','_')}"
    if args.scheduler != "CosineAnnealingWarmRestarts":
        name += f"_sched-{args.scheduler}"
    return name


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    args.procedure = args.unlearn_method
    outpath = SingletonString()
    outpath.content = args.out_dir
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    
    # args.resume = "./outs/cifar10_-1_FGCLS-2/model_bases/cifar10_resnet18/ORG/cifar10_resnet18_seed-0_Nf-10000_Cf-[2]_ep-40_bs-256_lr-[0_001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"
    
    
    from data_tool import DataStore, DataLoaderTool, construct_data
    DataStore.create_basic_folders()
    logpath = os.path.join(args.log_dir, args.exp_name)
    mkdir(logpath)
    logger = create_logger(logpath, 'main_proc_unlearn_' + args.logname)
    logger.info(args)
    
    logger.info("Load dataset and transform")
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    processor = Transforms(dt_mean, dt_std, dt_size)
    train_transform = processor.get_transform(args.train_transform, num_classes)
    test_transform = processor.get_transform(args.test_transform, num_classes)
    
    unlearn_method = args.unlearn_method
    ## load dataset based on the configuration
    dataset_conf = { "num_samples": args.num_samples, "select_classes": args.select_classes}
    train_data, test_data = DataLoaderTool.load_dataset(args.dataset, train_transform, test_transform, **dataset_conf)
    data, data_unlearn, data_remain = construct_data(train_data, test_data, train_size=args.num_samples, valid_size=args.valid_size, 
                                                     forget_size=args.num_to_forget, forget_classes=args.forget_classes)
    
    from unlearners import *
    from model_bases import DeepModels
    from schema import BasicUnlearnSchema
    from boundary_unlearn import boundary_unlearning
    from tool import PathGenerator as pather
    
    logname_mod = args.logname + '_' + args.unlearn_method
    model_zero = DeepModels(args.arch, feature_dims, num_classes, logpath, logname_mod, args.pretrained)
    model_zero.load(args.resume)
    model_zero._model.to(device)
    logger.info(" +++++++++++ Original Model Test +++++++++++ ")
    train_acc, train_auc = BasicUnlearnSchema._test_model_(model_zero, data['train'], args.batch_size, args.lossfn, device)
    train_remain_acc, train_remain_auc = BasicUnlearnSchema._test_model_(model_zero, data['remain'], args.batch_size, args.lossfn, device)
    valid_acc, valid_auc = BasicUnlearnSchema._test_model_(model_zero, data['valid'], args.batch_size, args.lossfn, device)
    test_acc, test_auc = BasicUnlearnSchema._test_model_(model_zero, data['test'], args.batch_size, args.lossfn, device)
    forget_acc, forget_auc = BasicUnlearnSchema._test_model_(model_zero, data['forget'], args.batch_size, args.lossfn, device)
    remain_acc, remain_auc = BasicUnlearnSchema._test_model_(model_zero, data_remain, args.batch_size, args.lossfn, device)
    
    logger.info(" ###### [Train, Valid, Test, Forget, Remain-Data] "
            + f"  Acc: {train_acc:.4f}, {train_remain_acc:.4f}, {valid_acc:.4f}, {test_acc:.4f}, {forget_acc:.4f}, {remain_acc:.4f} ######")
            # + f"| Auc: {train_auc:.4f}, {train_remain_auc:.4f}, {valid_auc:.4f}, {test_auc:.4f}, {forget_auc:.4f}, {remain_auc:.4f} ###### ")
    
    model_fn = ""
    start_time = time.time()
    train_forget_loader = DataLoader(data['forget'], args.batch_size, shuffle=True)
    if args.unlearn_method == 'boundary_shrink':
        bound = 0.01
        step = 1 / 255
        iter = 2
        poison_epoch = 10
        model_ = boundary_unlearning.boundary_shrink(model_zero._model, train_forget_loader, device, bound, step, iter, poison_epoch)
        suffix = f"bound_{bound}_step_{step}_iter_{iter}_poison_epoch_{poison_epoch}"
        model_fn = f'boundary_shrink_unlearn_model_{suffix}.pth'
    elif args.unlearn_method == 'boundary_expand':
        model_ = boundary_unlearning.boundary_expanding(model_zero._model, args.maxlr, args.epochs, train_forget_loader, args.optim, device)
        suffix = f'lr_{args.maxlr}_epochs_{args.epochs}'
        model_fn = f'boundary_expand_unlearn_model_{suffix}.pth'
    logger.info(f" +++++++++++ Unlearning Done [at {(time.time() - start_time):.2f} s]! +++++++++++ ")
    
    out_path = pather.get_target_model_path(f"{args.dataset}_{args.arch}/{args.unlearn_method}/")
    model_outpath = path.join(out_path, construct_name(args) + model_fn)
    logger.info(f" +++++++++++ Model saved at: {model_outpath} +++++++++++ ")
    save_model(model_, model_outpath)
    
    logger.info(f" +++++++++++ Forget classes: {args.forget_classes} +++++++++++ ")
    model_zero._model = model_
    train_remain_acc, _ = BasicUnlearnSchema._test_model_(model_zero, data['remain'], args.batch_size, args.lossfn, device)
    train_forget_acc, _ = BasicUnlearnSchema._test_model_(model_zero, data['forget'], args.batch_size, args.lossfn, device)
    test_remain_acc, _ = BasicUnlearnSchema._test_model_(model_zero, data['test'], args.batch_size, args.lossfn, device)
    test_forget_acc, _ = BasicUnlearnSchema._test_model_(model_zero, data_unlearn['test'], args.batch_size, args.lossfn, device)
    valid_remain_acc, _ = BasicUnlearnSchema._test_model_(model_zero, data['valid'], args.batch_size, args.lossfn, device)
    valid_forget_acc, _ = BasicUnlearnSchema._test_model_(model_zero, data_unlearn['valid'], args.batch_size, args.lossfn, device)
    
    logger.info(" ###### Remain [Train, Valid, Test]:"
                        + f"  Acc: {train_remain_acc:.4f}, {valid_remain_acc:.4f}, {test_remain_acc:.4f} ######")
    logger.info(" ###### Forget [Train, Valid, Test]:"
                        + f"  Acc: {train_forget_acc:.4f}, {valid_forget_acc:.4f}, {test_forget_acc:.4f} ######")
        
    logger.info("********************************* \n\n")
    print("Done!")
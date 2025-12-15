import os
import argparse
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from logger import create_logger
from model_deep import get_model
from transforms import Transforms
from const import SingletonString
from utils import argparse2bool, list_of_ints, load_model, save_model, seed_everything
# from model_deep import get_model

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '3.0'
float_formatter = "{:.4f}".format


def parse_args():
    parser = argparse.ArgumentParser(description='Robust Unlearn Model')
    parser.add_argument('--logname', default='trail', help='name of output log file')
    parser.add_argument('--log_dir', default='.', help='Folder where all logs are stored (default: .)')
    parser.add_argument('--out_dir', default='./outs_sz/cifar10_5000/', help='Folder where all outputs are stored (default: .)') # 
    parser.add_argument('--exp_name', default='rob_log', help='Subfolder where all logs are stored (default: logs)')
    parser.add_argument('--verbose', type=argparse2bool, default=True, help='whether to print the log')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--cuda', type=int, default=3, help='cuda device (default: 0)')
    
    ## dataset and model related parameters
    parser.add_argument('--dataset', type=str, default='cifar10', #'mnist', #'small_cifar2', #'small_mnist_2', #
                        choices=['mnist', 'cifar10', 'stl10', 'imagenet', 'vggface2'
                                # 'small_mnist', 'small_mnist_2', 'small_cifar2', 'small_cifar5', 'small_cifar10', 
                                # 'tiny_imagenet5', 'tiny_imagenet_pretrain', 'tiny_image_finetune', 
                                # 'adult', 'accident', 'location', 
                                ],
                        help='dataset for training (default: cifar10)')
    parser.add_argument('--select_classes', type=list_of_ints, default='3,8', help='selected classes for small dataset (default: 3,8)')
    parser.add_argument('--num_samples', type=int, default=-1, help='number of samples per class selected for small dataset')
    parser.add_argument('--num_workers', type=int, default=16, metavar='N', help='number of workers for data loading (default: 4)')
    parser.add_argument('--valid_size', type=int, default=5000, metavar='V', help='number of validation samples (default: 5000)')
    parser.add_argument('--top_k', type=int, default=-1, metavar='K', help='number of selected informative samples (default: 5000)')
    parser.add_argument('--num_to_forget', type=int, default=5000, metavar='Nf', help='number of samples to forget (default: 5000)')
    parser.add_argument('--forget_classes', type=list_of_ints, default=None, metavar='Cf', help='classes to forget (default: None)')
    
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=[# 'RF', 'DT', 'SVM', 'LR', 
                                'mlps', 'logistic', 'simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'densenet'],
                        help='model architecture (default: resnet18)')
    parser.add_argument('--model_path', type=str, default="", help='path of the model to unlearn')
    
    # ##TODO: temporal parameters, will be removed later
    # parser.add_argument('--model_num', type=int, default=1, help="Number of original model")
    # parser.add_argument('--unlearn_num', type=int, default=1, help="Number of unlearned model")
    # parser.add_argument('--unlearn_size', type=int, default=1000, help="Size of the sample to be unlearned")
    
    ## unlearning related parameters
    parser.add_argument('--epochs', type=int, default=10, metavar='E', help='number of epochs for unlearning (default: 20)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='B', help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate (default: 0.001)')
    parser.add_argument('--lr_feedback', type=float, default=0.001, help='initial learning rate for feedback (default: 0.001)')
    parser.add_argument('--lam_reg', type=float, default=10.0, help='regularization parameter for adversarial part (default: 0.1)')    
    parser.add_argument('--strength', type=float, default=1.0, help='strength of the feedback (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (default: 0.0005)')
    parser.add_argument('--class_wise', type=argparse2bool, default=True, help='whether to use class-wise feedback')
    parser.add_argument('--generalize', type=argparse2bool, default=True, help='whether to contain generalization loss')
    parser.add_argument('--save_checkpoint', type=argparse2bool, default=False, help='whether to save the checkpoint')
    
    ## running parameter
    parser.add_argument('--protocal', type=str, default='DISTILL', #'LASTK', #'FT', #
                        choices=['FT', 'LASTK', 'RL', 'GA', 'DISTILL'])
    parser.add_argument('--lossfn', type=str, default='ce', choices=['ce', 'bce', 'mse'])
    parser.add_argument('--optim', type=str, default="Adam", choices=['Adam', 'SGD'])
    parser.add_argument('--patience', type=int, default=30, help='patience for early stopping (default: 15)')
    parser.add_argument('--scheduler', default='CosineAnnealingLR', #'None', # 'CosineAnnealingWarmRestarts', #'LRScheduler', #
                        choices = ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR', 'LRScheduler', 'None'],
                        help='Pytorch Scheduler name: (default: The one used for train')

    ## protocal parameters
    parser.add_argument('--last_k', type=int, default=3, 
                        help="Number of last layers to be used for retraining for unlearning (default: 1)")
    parser.add_argument('--re_init', type=argparse2bool, default=True, 
                        help="Whether to re-initialize the weight of last layers for LastKlayer unlearning")   
    
    ## parameters for the different protocals
    parser.add_argument('--finetune_over_retain', type=argparse2bool, default=True, 
                        help="Whether to finetune the model for retain data for the GA and RL protocal")
    parser.add_argument('--correct_senstive', type=argparse2bool, default=True, 
                        help="Whether to correct the prediction over retain data for the DISTILL protocal")
    
    ### data augmentation related parameters
    parser.add_argument("--preproc_train_transform", type=str, default="normal",
                        choices=["none", "normal", "random", "randerase", "augafn", 
                                 "baseaug", "cutout", "cutmix", "autoaug", "augmix", "randaug", "test"], 
                        help="data augmentation for preprocessing the training data",)
    parser.add_argument("--preproc_test_transform", type=str, default="test",
                        choices=["none", "normal", "random", "randerase", "augafn", 
                                 "baseaug", "cutout", "cutmix", "autoaug", "augmix", "randaug", "test" ], 
                        help="data augmentation for preprocessing the testing data",)
    
    parser.add_argument("--online_train_aug", type=str, default="none", #"cutout", ##"randaug", #"random", #
                        help="online data augmentation for training",
                        choices=["none", "normal", "random", "randerase",  "augafn", "baseaug", 
                                 "cutout", "mixup", "cutmix", "autoaug", "augmix", "randaug", "test" ],)
    parser.add_argument("--online_forget_aug", type=str, default="none", 
                        help="data augmentation for forgetting",
                        choices=["none", "normal", "baseaug",  "randerase", 
                                 "cutout", "mixup", "cutmix", "autoaug", "augmix", "randaug", "test" ],)
    
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha for distillation loss")
    parser.add_argument("--T", type=float, default=3, help="temperature for distillation loss")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers to be used for distillation")
    parser.add_argument("--freeze", type=argparse2bool, default=False, help="freeze the layers for distillation")
    
    ## adversarial loss related parameters
    parser.add_argument('--adv_loss', type=str, default='none', # 'udp', #'fgsm', #'vat', #
                        choices=['none', 'vat', 'fgsm', 'udp', 'trades', 'pgd',], help='adversarial loss type')
    parser.add_argument('--eps', type=float, default=0.002, help='epsilon for adversarial loss: (Hidden Space Adv. Training!) (default: 0.001)')
    parser.add_argument('--clip_min', type=float, default=0.0, help='clip_min for adversarial loss (default: 0.0)')
    parser.add_argument('--clip_max', type=float, default=1.0, help='clip_max for adversarial loss (default: 1.0)')
    parser.add_argument('--sanity_check', type=argparse2bool, default=False, help='sanity_check for adversarial loss (default: False)')
    parser.add_argument('--step_size', type=float, default=0.01, help='step_size for adversarial  Loss (default: 0.01)')  # 1e-6
    parser.add_argument('--num_iters', type=int, default=20, help='iteration times for computing adv noise for adversarial loss (default: 1000)')
    ##### specifc parameters for some adversarial loss
    parser.add_argument('--norm', type=str, default='inf', choices=['inf', '1', '2'], help='norm for FGSMAdvloss (default: inf)')
    parser.add_argument('--targeted', type=argparse2bool, default=False, help='targeted for (FGSMAdvLoss and SimVAAdvLoss) (default: False)')
    parser.add_argument('--rs', type=argparse2bool, default=False, help='rs for (UDPAdvLoss and TradesAdvLoss) (default: False)')
    parser.add_argument('--use_alpha_scheduler', type=argparse2bool, default=False, 
                        help='use the alpha scheduler for (UDPAdvLoss and TradesAdvLoss) (default: False)')
    parser.add_argument('--sample_iters', type=str, default='none', choices=['none', 'uniform'], help='sample_iters for UDPAdvLoss (default: none)')
    parser.add_argument('--grad_sign', type=argparse2bool, default=False, help='grad_sign for PGDAdvLoss (default: False)')
    
    ## feedback loss related parameters
    parser.add_argument('--feedback', type=str, default='none', # "snn", #'snn', # 'entropy', #'ot', #'contrast', #
                        choices=['none', 'contrast', 'snn', 'mmd', 'ot', 'entropy'], help='feedback loss type')
    parser.add_argument('--format', type=str, default='diff', choices=['diff', 'ratio'], help='format for feedback loss (default: diff)')
    parser.add_argument('--dynamic_weight', type=argparse2bool, default=False, help='dynamic_weight for feedback loss (default: False)')
    parser.add_argument('--margin', type=float, default=1.0, help='margin for contrastive loss (default: 1.0)')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature for adversarial loss (default: 0.05)')
    parser.add_argument('--kernel_type', type=str, default='rbf', help='kernel_type for MMDLoss (default: rbf)')
    parser.add_argument('--kernel_mul', type=float, default=2.0, help='kernel_mul for MMDLoss (default: 2.0)')
    parser.add_argument('--kernel_num', type=int, default=5, help='kernel_num for MMDLoss (default: 5)')
    parser.add_argument('--fix_sigma', type=argparse2bool, default=False, help='fix_sigma for MMDLoss (default: False)')
    
    args = parser.parse_args()
    return args


def get_adv_loss(args):
    from losses import VATAdvLoss, FGSMAdvLoss, UDPAdvLoss, PGDAdvLoss, TradesAdvLoss, SimBAAdvLoss
    option = args.adv_loss
    if option == 'none':
        return None
    elif option == 'vat':
        return VATAdvLoss(args.eps, args.step_size, args.num_iters, args.sanity_check)
    elif option == 'fgsm':
        norm = np.inf if args.norm == 'inf' else int(args.norm)
        return FGSMAdvLoss(args.eps, norm, args.clip_min, args.clip_max, args.targeted, args.sanity_check)
    elif option == 'udp':
        return UDPAdvLoss(args.eps, args.step_size, args.num_iters, args.clip_min, args.clip_max,
                                args.rs, args.use_alpha_scheduler, args.sample_iters, args.sanity_check)
    elif option == 'pgd':
        return PGDAdvLoss(args.eps, args.step_size, args.num_iters, args.grad_sign, args.sanity_check)
    elif option == 'trades':
        return TradesAdvLoss(args.eps, args.step_size, args.num_iters, args.clip_min, args.clip_max, 
                                args.rs, args.use_alpha_scheduler, args.sanity_check)
    elif option == 'simba':  # simple-blackbox-attack
        return SimBAAdvLoss(args.eps, args.num_iters, args.targeted, args.sanity_check)
    else:
        raise ValueError("Invalid adv_loss type")

def get_feedback_loss(args, device):
    from losses import ContrastiveLoss, TriSoftNearestNeighborsLoss, TriMMDLoss, TriWassersteinLoss, TriEntropyLoss
    option = args.feedback
    if option == 'none':
        return None
    elif option == 'contrast':
        return ContrastiveLoss(args.temperature, args.margin, args.class_wise, args.generalize, device)
    elif option == 'snn':
        return TriSoftNearestNeighborsLoss(args.temperature, args.generalize, device)
    elif option == 'mmd':
        assert args.format in ['diff', 'ratio'], "Invalid format for MMD loss"
        return TriMMDLoss(args.format, args.kernel_type, args.kernel_mul, args.kernel_num, 
                            args.fix_sigma, args.class_wise, args.generalize, device)
    elif option == 'ot':
        assert args.format in ['diff', 'ratio'], "Invalid format for OT loss"
        return TriWassersteinLoss(args.format, args.class_wise, args.generalize, device)
    elif option == 'entropy':
        # assert args.format in ['dist', 'mean'], "Invalid format for entropy distance loss"
        return TriEntropyLoss('dist', args.class_wise, args.generalize, device)
    else:
        raise ValueError("Invalid feedback type")

def data_model_test(model, data, batch_size, num_workers, device, transform=None):
    from learner import complete_test
    test_loader = DataLoader(data, batch_size, False, num_workers=num_workers)
    acc, auc = complete_test(model, test_loader, device, transform)
    return acc, auc


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    args.unlearn_method = 'robun'
    ## set output path
    outs = SingletonString()
    outs.content = args.out_dir
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # mod_fn = 'cifar10_default_resnet18_seed-1_Nf-0_split-0.2_lr-[0_001-0_001]_wd-5e-05_sched-None.pt'
    # args.model_path = './output/outs_10000_rob/cifar10/model_bases/ORG0/' + mod_fn

    # args.model_path = "./outs/cifar10_-1/model_bases/cifar10_resnet18/ORG/" + \
    #     "cifar10_resnet18_seed-3407_Nf-5000_ep-20_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-testregular_0.0_sched-None.pth"

    # args.model_path = "./outs/cifar10_10000/model_bases/cifar10_resnet18/ORG/" + \
    #      "cifar10_resnet18_seed-1_Nf-1000_ep-40_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-testregular_0.0_sched-None.pth"

    # args.model_path = "outs_sz/cifar10_5000/model_bases/cifar10_resnet18/ORG/cifar10_resnet18_seed-0_Nf-1000_ep-40_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-testregular_0.0_sched-None.pth"
    
    args.model_path = "outs_sz/cifar10_-1_5000/model_bases/cifar10_resnet18/ORG/cifar10_resnet18_seed-3407_Nf-5000_ep-20_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"
    
    from data_tool import DataLoaderTool, DataStore, construct_data
    DataStore.create_basic_folders()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    logpath = os.path.join(args.log_dir, args.exp_name)
    logger = create_logger(logpath, 'main_robun_' + args.logname)
    logger.info(args)

    logger.info("load dataset and transform")
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    processor = Transforms(dt_mean, dt_std, dt_size)
    train_transform = processor.get_transform(args.preproc_train_transform, num_classes)
    test_transform = processor.get_transform(args.preproc_test_transform, num_classes)
    transform = None

    num_samples = args.num_samples
    dataset_conf = { "num_samples": num_samples, "select_classes": args.select_classes}
    train_data, test_data = DataLoaderTool.load_dataset(args.dataset, train_transform, test_transform, **dataset_conf)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    data, data_unlearn, data_retain = construct_data(train_data, test_data, train_size=args.num_samples, valid_size=args.valid_size, 
                                                     forget_size=args.num_to_forget, forget_classes=args.forget_classes)

    ##  import after setting the out path
    # from losses import *
    from tool import PathGenerator as pather
    from models import RobustUnlearn, RobustForget, DistillUnlearn

    logger.info("Get adversarial and feedback loss")
    adv_loss = get_adv_loss(args)    
    feedback_loss = get_feedback_loss(args, device)
    if args.adv_loss in ['vat', 'fgsm']:
        if args.online_train_aug != "none" and args.preproc_train_transform == "none":
            args.online_train_aug = "normal"
        if args.online_forget_aug != "none"  and args.preproc_train_transform == "none":
            args.online_forget_aug = "normal"

    ## load original model and test
    logger.info(f"Load original model from @ {args.model_path}")
    init_mod_ = get_model(args.arch, feature_dims, num_classes, pretrained=True).to(device)
    model_ = load_model(args.model_path).to(device)
    # init_mod_ = deepcopy(model_)

    ## apply different unlearn models
    unlearner = None
    if args.protocal in ['FT', 'LASTK']:
        unlearner = RobustUnlearn
    elif args.protocal in ['GA', 'RL']:
        unlearner = RobustForget
    elif args.protocal == 'DISTILL':
        unlearner = DistillUnlearn
    else:
        raise ValueError("Invalid protocal")

    ## unlearn model initialization
    logger.info("Init unlearn model")
    robun = unlearner(logpath, args.logname, args.out_dir, out_name='model_robun')
    robun.logger.info(args)
    logger.info(f" +++++++  Selected top-K informative samples: {args.top_k}")
    

    logger.info("Set parameters and transform configurations")
    # model setting
    robun.set_params(
        args.epochs, args.batch_size, args.lr, args.lr_feedback,
        args.lam_reg, args.strength, args.weight_decay, args.momentum,
        args.save_checkpoint, args.class_wise, args.seed, args.dynamic_weight
    )

    trans_params = '_'.join([args.preproc_train_transform, args.preproc_test_transform, args.online_train_aug, args.online_forget_aug])

    logger.info("EVALUATION: Original model over original dataset")
    train_acc, train_auc = data_model_test(model_, data['train'], args.batch_size, args.num_workers, device, transform)
    valid_acc, valid_auc = data_model_test(model_, data['valid'], args.batch_size, args.num_workers, device, transform)
    test_acc, test_auc = data_model_test(model_, data['test'], args.batch_size, args.num_workers, device, transform)
    forget_acc, forget_auc = data_model_test(model_, data['forget'], args.batch_size, args.num_workers, device, transform)
    retain_acc, retain_auc = data_model_test(model_, data_retain, args.batch_size, args.num_workers, device, transform)
    logger.info(" ###### [Train, Valid, Test, Forget, Remain-Data] "
                        + f"  Acc: {train_acc:.4f}, {valid_acc:.4f}, {test_acc:.4f}, {forget_acc:.4f}, {retain_acc:.4f} ######")
    #                   + f"| Auc: {train_auc:.4f}, {valid_auc:.4f}, {test_auc:.4f}, {forget_auc:.4f}, {retain_auc:.4f} ###### ")
    if args.forget_classes is not None:
        logger.info(f" +++++++++++ Forget Class: {args.forget_classes} +++++++++++ ")
        for type_, dt_ in data_unlearn.items():
            acc_dt, auc_dt = data_model_test(model_, dt_, args.batch_size, args.num_workers, device, transform)
            logger.info(f"##### [ {type_:6} ]: ACC: {acc_dt:.4f}, AUC: {auc_dt:.4f}")
        #                   + f"| Auc: {forget_auc:.4f}, {retain_auc:.4f} ###### ")

    ### set data augmentation for online training and forgetting
    if args.online_train_aug in ["random", "randaug", "augmix", "autoaug", "autoafn", "cutout", "randerase"]:
        num_cls = len(np.unique(data['retain'].dataset.dataset.targets))
        data['retain'].transform = processor.get_transform( args.online_train_aug, num_cls)
    else:
        robun.config_transform(dt_mean, dt_std, dt_size)
    if args.online_forget_aug != "none":
        num_cls = len(np.unique(data['forget'].dataset.dataset.targets))
        data['forget'].transform = processor.get_transform( args.online_forget_aug, num_cls)

    mod_ = model_.to(device)
    ## set dataset for unlearning
    logger.info("Set data for unlearning")
    robun.set_data(data, num_classes, args.batch_size, args.num_workers)
    ## evaluation the original model
    # robunlearn.evaluation(mod_, args.lossfn, device, 'none')
    acc_org, auc_org = robun.model_evaluate(mod_, test_loader, device)

    logger.info("Perform unlearning")
    ## perform forget-sample unlearn
    ckpt_path = pather.get_checkpoint_path(f"{args.dataset}_{args.num_samples}_{args.arch}")
    if unlearner == RobustUnlearn:
        unlearn_mod, ckpt = robun.unlearn(
            mod_, args.protocal, adv_loss, feedback_loss, args.patience, args.lossfn, 
            args.scheduler, args.optim, args.online_train_aug, args.online_forget_aug, device,
            last_k=args.last_k, re_init=args.re_init, suffix=trans_params, top_k=args.top_k,
            model_type=args.arch, checkpoint_path=ckpt_path
        )
    elif unlearner == RobustForget:
        unlearn_mod, ckpt = robun.unlearn(
            mod_, args.protocal, args.finetune_over_retain, adv_loss, feedback_loss, 
            args.patience, args.lossfn, args.scheduler, args.optim, args.online_train_aug, args.online_forget_aug, 
            device, checkpoint_path=ckpt_path, model_type=args.arch, suffix=trans_params,
        )
    else: # DistillUnlearn
        # num_layers = 2
        if args.num_layers > 0:
            ref_mod_params_ = list(mod_.parameters())
            mod_params_ = list(init_mod_.parameters())
            for i in range(min(args.num_layers, len(ref_mod_params_))):
                mod_params_[i].data.copy_(ref_mod_params_[i].data)
            
            if args.freeze:
                for i, param in enumerate(mod_params_):
                    if i >= args.num_layers:
                        break
                    param.requires_grad = False
        if args.num_layers == -1:
            init_mod_ = deepcopy(mod_)
            
        # prune_amount = 0.1
        # layers_to_copy = ['conv1', 'layer1']
        # robun.init_student_model(mod_, init_mod_, layers_to_copy)
        # robun.preprocess(mod_, prune_amount, device)
        
        unlearn_mod, ckpt = robun.unlearn(
            mod_, init_mod_, adv_loss, feedback_loss, args.patience,   # args.correct_senstive, 
            args.scheduler, args.optim, args.online_train_aug, args.online_forget_aug, 
            device, suffix=trans_params, model_type=args.arch, checkpoint_path=ckpt_path,
            alpha=args.alpha, temperature=args.T
        )
        if args.freeze:
            robun._suffix_ = f"{robun._suffix_}_alpha-{args.alpha}_temp-{args.T}_num_layers-{args.num_layers}_freeze"
        else:
            robun._suffix_ = f"{robun._suffix_}_alpha-{args.alpha}_temp-{args.T}_num_layers-{args.num_layers}"

    outfn = f"{args.arch}_{args.protocal}_{trans_params}_adv-{adv_loss}_feedback-{feedback_loss}_{robun._suffix_}_best.pt"
    # outfn = f"{args.arch}_{args.protocal}_{trans_params}_adv_{args.adv_loss}_feedback_{feedback_loss}_{robun._suffix_}.pt"
    out_path = os.path.join(robun.path['model'], outfn)
    save_model(unlearn_mod, out_path)
    logger.info(f" ++++++++++++++++++ Model saved @ {out_path} +++++++++++++++++++")    

    logger.info("Evaluation: Unlearned model test")
    for type_ in ['train', 'retain', 'valid', 'test', 'forget']: #data.items():
        acc_dt, auc_dt = data_model_test(unlearn_mod, data[type_], args.batch_size, args.num_workers, device, transform)
        logger.info(f"##### [ {type_:6} ]: ACC: {acc_dt:.4f}")   # , AUC: {auc_dt:.4f}

    ## evaluation the unlearned model
    # robun.evaluation(unlearn_mod, args.lossfn, device, "none")
    acc_unlearn, auc_unlearn = robun.model_evaluate(unlearn_mod, test_loader, device)
    logger.info(f"Accuracy: Original: {acc_org:.4f} --> Unlearned: {acc_unlearn:.4f}")

    ## EVALUATION: using the test data to evaluate the model
    if args.forget_classes is None:
        logger.info(" +++++++++++ Unlearned Model Test +++++++++++ ")
        train_acc, train_auc =  data_model_test(unlearn_mod, data['train'], args.batch_size, args.num_workers, device, transform)
        train_retain_acc, train_retain_auc = data_model_test(unlearn_mod, data['retain'], args.batch_size, args.num_workers, device, transform)
        valid_acc, valid_auc =  data_model_test(unlearn_mod, data['valid'], args.batch_size, args.num_workers, device, transform)
        test_acc, test_auc =  data_model_test(unlearn_mod, data['test'], args.batch_size, args.num_workers, device, transform)
        forget_acc, forget_auc = data_model_test(unlearn_mod, data['forget'], args.batch_size, args.num_workers, device, transform)
        retain_acc, retain_auc = data_model_test(unlearn_mod, data_retain, args.batch_size, args.num_workers, device, transform)
        logger.info(
            " ###### [Train, Train_rem, Valid, Test, Forget, Remain-Data] "
            + f"  Acc: {train_acc:.4f}, {train_retain_acc:.4f}, {valid_acc:.4f}, {test_acc:.4f}, {forget_acc:.4f}, {retain_acc:.4f} ######"
        )
        #     + f"| Auc: {train_auc:.4f}, {valid_auc:.4f}, {test_auc:.4f}, {forget_auc:.4f}, {retain_auc:.4f}, {retain_auc:.4f}###### ")
    else:
        logger.info(f" +++++++++++ Forget classes: {args.forget_classes} +++++++++++ ")
        train_retain_acc, _ =  data_model_test(unlearn_mod, data['retain'], args.batch_size, args.num_workers, device, transform)
        train_forget_acc, _ =  data_model_test(unlearn_mod, data['forget'], args.batch_size, args.num_workers, device, transform)
        test_retain_acc, _ = data_model_test(unlearn_mod, data['test'], args.batch_size, args.num_workers, device, transform)
        test_forget_acc, _ = data_model_test(unlearn_mod, data_unlearn['test'], args.batch_size, args.num_workers, device, transform)
        valid_retain_acc, _ = data_model_test(unlearn_mod, data['valid'], args.batch_size, args.num_workers, device, transform)
        valid_forget_acc, _ = data_model_test(unlearn_mod, data_unlearn['valid'], args.batch_size, args.num_workers, device, transform)
        
        logger.info(" ###### Remain [Train, Valid, Test]:"
                            + f"  Acc: {train_retain_acc:.4f}, {valid_retain_acc:.4f}, {test_retain_acc:.4f} ######")
        logger.info(" ###### Forget [Train, Valid, Test]:"
                            + f"  Acc: {train_forget_acc:.4f}, {valid_forget_acc:.4f}, {test_forget_acc:.4f} ######")

    logger.info("********************************* \n\n")
    print("Done!")

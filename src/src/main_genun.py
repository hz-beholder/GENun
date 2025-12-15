import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from logger import create_logger
from model_deep import get_model
from transforms import Transforms
from const import SingletonString
from utils import argparse2bool, list_of_ints, load_model, save_model, seed_everything

float_formatter = "{:.4f}".format

def parse_args():
    parser = argparse.ArgumentParser(description='Generalizable-M0 Unlearning')
    parser.add_argument('--logname', default='trail', help='name of output log file')
    parser.add_argument('--log_dir', default='./logs/cifar10_-1_genun', help='Folder where all logs are stored (default: .)')
    parser.add_argument('--out_dir', default='./outs/cifar10_-1_genun', help='Folder where all outputs are stored (default: .)') 
    parser.add_argument('--exp_name', default='genun_log', help='Subfolder where all logs are stored (default: rob_log)')
    parser.add_argument('--verbose', type=argparse2bool, default=True, help='whether to print the log')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=6, help='cuda device (default: 0)')
    
    ## dataset and model related parameters
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset for training (default: cifar10)',
                        choices=['mnist', 'cifar10', 'cifar100', 'stl10', 'imagenet', 'vggface2', 'svhn', 'fashionmnist'])
    parser.add_argument('--select_classes', type=list_of_ints, default='3,8', help='selected classes for small dataset (default: 3,8)')
    parser.add_argument('--num_samples', type=int, default=-1, help='number of samples per class selected for small dataset')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N', help='number of workers for data loading (default: 4)')
    parser.add_argument('--valid_size', type=int, default=5000, metavar='V', help='number of validation samples (default: 5000)')
    # parser.add_argument('--top_k', type=int, default=-1, metavar='K', help='number of selected informative samples (default: 5000)')
    parser.add_argument('--num_to_forget', type=int, default=1000, metavar='Nf', help='number of samples to forget (default: 5000)')
    parser.add_argument('--forget_classes', default=None, metavar='Cf', help='classes to forget (default: None)')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='sample ratio for the retain dataset (default: 0.5)')
    
    parser.add_argument('--arch', type=str, default='resnet18', help='model architecture (default: resnet18)',
                        choices=['mlps', 'logistic', 'simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'densenet', 'vit', 'vgg','ALLCNN'])
    parser.add_argument('--model_path', type=str, default="", help='path of the model to unlearn')
    
    ## unlearning related parameters
    parser.add_argument('--epochs', type=int, default=10, metavar='E', help='number of epochs for unlearning (default: 20)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='B', help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate (default: 0.001)')
    # parser.add_argument('--lr_feedback', type=float, default=0.001, help='initial learning rate for feedback (default: 0.001)')
    # parser.add_argument('--generalize', type=argparse2bool, default=True, help='whether to contain generalization loss')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (default: 0.0005)')
    parser.add_argument('--class_wise', type=argparse2bool, default=True, help='whether to use class-wise feedback')
    parser.add_argument('--save_checkpoint', type=argparse2bool, default=False, help='whether to save the checkpoint')
    
    parser.add_argument('--over_forget', type=argparse2bool, default=False, help='whether using the forget-set for finetuning (weighted GA)')
    parser.add_argument('--regularizer', type=str, default='l1', choices=['none', 'l1', 'l2', 'l1+l2', 'l1_diff', 'l2_diff'], help='weights regularization type')
    parser.add_argument('--gamma', type=float, default=1e-5, help='regularization parameter for weight regularization (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=1.0, help='trade-off between loss of retain and forget data (default: 0.5)')
    parser.add_argument('--no_reg_epochs', type=int, default=0, help='number of epochs without regularization (default: 0)')
    parser.add_argument('--dynamic_weight', type=argparse2bool, default=False, help='dynamic_weight for feedback loss (default: False)')

    # ## running parameter
    parser.add_argument('--protocal', type=str, default='GenUn', choices=['DU', 'LASTK', 'DISTILL', 'GenUn'])
    parser.add_argument('--lossfn', type=str, default='ce', choices=['ce', 'bce', 'mse'])
    parser.add_argument('--optim', type=str, default="adam", choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping (default: 10)')
    parser.add_argument('--scheduler', default='CosineAnnealingWarmRestarts',  #'CosineAnnealingLR', #'None', # 'LRScheduler', #
                        choices = ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR', 'LRScheduler', 'None'],
                        help='Pytorch Scheduler name: (default: The one used for train')

    ### data augmentation related parameters
    parser.add_argument("--preproc_train_transform", type=str, default="normal",
                        choices=["none", "normal", "random", "randerase", "augafn", 
                                 "baseaug", "cutout", "cutmix", "autoaug", "augmix", "randaug", "test"], 
                        help="data augmentation for preprocessing the training data",)
    parser.add_argument("--preproc_test_transform", type=str, default="test",
                        choices=["none", "normal", "random", "randerase", "augafn", 
                                 "baseaug", "cutout", "cutmix", "autoaug", "augmix", "randaug", "test" ], 
                        help="data augmentation for preprocessing the testing data",)
    
    parser.add_argument("--online_train_aug", type=str, default="none", 
                        help="online data augmentation for training",
                        choices=["none", "normal", "random", "randerase",  "augafn", "baseaug", 
                                 "cutout", "mixup", "cutmix", "autoaug", "augmix", "randaug", "test" ],)
    parser.add_argument("--online_forget_aug", type=str, default="none", #'random',  #
                        help="data augmentation for forgetting",
                        choices=["none", "normal", "random", "randerase", "augafn", "baseaug", 
                                 "cutout", "mixup", "cutmix", "autoaug", "augmix", "randaug", "test" ],)
    
    # parser.add_argument('--adv_lamb', type=float, default=1.0, help='regularization parameter for adversarial part (default: 1.0)')    
    # parser.add_argument('--feed_lamb', type=float, default=1.0, help='strength parameter of the feedback part (default: 1.0)')
    # parser.add_argument('--classify_lamb', type=float, default=1.0, help='weight parameter of the classification loss (default: 1.0)')

    # ## protocal parameters
    # parser.add_argument('--last_k', type=int, default=3, 
    #                     help="Number of last layers to be used for retraining for unlearning (default: 1)")
    # parser.add_argument('--re_init', type=argparse2bool, default=True, 
    #                     help="Whether to re-initialize the weight of last layers for LastKlayer unlearning")   
    
    ## parameters for the different protocals
    # parser.add_argument('--finetune_over_retain', type=argparse2bool, default=True, 
    #                     help="Whether to finetune the model for retain data for the GA and RL protocal")
    # # parser.add_argument('--correct_senstive', type=argparse2bool, default=True, 
    # #                     help="Whether to correct the prediction over retain data for the DISTILL protocal")
    
    # ## parameters for the distillation loss
    # parser.add_argument("--alpha", type=float, default=0.5, help="alpha for distillation loss")
    # parser.add_argument("--T", type=float, default=3, help="temperature for distillation loss")
    # # parser.add_argument("--num_layers", type=int, default=2, help="number of layers to be used for distillation")
    
    # ## adversarial loss related parameters
    # parser.add_argument('--adv_loss', type=str, default='none', #'fgsm', # 'udp', #'fgsm', #'vat', #
    #                     choices=['none', 'vat', 'fgsm', 'udp', 'trades', 'pgd',], help='adversarial loss type')
    # parser.add_argument('--eps', type=float, default=0.002, help='epsilon for adversarial loss: (Hidden Space Adv. Training!) (default: 0.001)')
    # parser.add_argument('--clip_min', type=float, default=0.0, help='clip_min for adversarial loss (default: 0.0)')
    # parser.add_argument('--clip_max', type=float, default=1.0, help='clip_max for adversarial loss (default: 1.0)')
    # parser.add_argument('--sanity_check', type=argparse2bool, default=False, help='sanity_check for adversarial loss (default: False)')
    # parser.add_argument('--step_size', type=float, default=0.01, help='step_size for adversarial  Loss (default: 0.01)')  # 1e-6
    # parser.add_argument('--num_iters', type=int, default=20, help='iteration times for computing adv noise for adversarial loss (default: 1000)')
    # ##### specifc parameters for some adversarial loss
    # parser.add_argument('--norm', type=str, default='inf', choices=['inf', '1', '2'], help='norm for FGSMAdvloss (default: inf)')
    # parser.add_argument('--targeted', type=argparse2bool, default=False, help='targeted for (FGSMAdvLoss and SimVAAdvLoss) (default: False)')
    # parser.add_argument('--rs', type=argparse2bool, default=False, help='rs for (UDPAdvLoss and TradesAdvLoss) (default: False)')
    # parser.add_argument('--use_alpha_scheduler', type=argparse2bool, default=False, 
    #                     help='use the alpha scheduler for (UDPAdvLoss and TradesAdvLoss) (default: False)')
    # parser.add_argument('--sample_iters', type=str, default='none', choices=['none', 'uniform'], help='sample_iters for UDPAdvLoss (default: none)')
    # parser.add_argument('--grad_sign', type=argparse2bool, default=False, help='grad_sign for PGDAdvLoss (default: False)')
    
    # ## feedback loss related parameters
    # parser.add_argument('--feedback', type=str, default='none', #'snn',
    #                     choices=['none', 'contrast', 'snn', 'mmd', 'ot', 'entropy', 'softdis', 'kl', 'l2', 'mse', 'cosine', 'max_entropy'], help='feedback loss type')
    # parser.add_argument('--format', type=str, default='diff', choices=['diff', 'ratio'], help='format for feedback loss (default: diff)')
    # parser.add_argument('--margin', type=float, default=1.0, help='margin for contrastive loss (default: 1.0)')
    # parser.add_argument('--temperature', type=float, default=0.2, help='temperature for adversarial loss (default: 0.05)')
    # parser.add_argument('--kernel_type', type=str, default='rbf', help='kernel_type for MMDLoss (default: rbf)')
    # parser.add_argument('--kernel_mul', type=float, default=2.0, help='kernel_mul for MMDLoss (default: 2.0)')
    # parser.add_argument('--kernel_num', type=int, default=5, help='kernel_num for MMDLoss (default: 5)')
    # parser.add_argument('--fix_sigma', type=argparse2bool, default=False, help='fix_sigma for MMDLoss (default: False)')
    
    args = parser.parse_args()
    return args


# def get_adv_loss(args):
#     from losses import VATAdvLoss, FGSMAdvLoss, UDPAdvLoss, PGDAdvLoss, TradesAdvLoss, SimBAAdvLoss
#     option = args.adv_loss
#     if option == 'none':
#         return None
#     elif option == 'vat':
#         return VATAdvLoss(args.eps, args.step_size, args.num_iters, args.sanity_check)
#     elif option == 'fgsm':
#         norm = np.inf if args.norm == 'inf' else int(args.norm)
#         return FGSMAdvLoss(args.eps, norm, args.clip_min, args.clip_max, args.targeted, args.sanity_check)
#     elif option == 'udp':
#         return UDPAdvLoss(args.eps, args.step_size, args.num_iters, args.clip_min, args.clip_max,
#                                 args.rs, args.use_alpha_scheduler, args.sample_iters, args.sanity_check)
#     elif option == 'pgd':
#         return PGDAdvLoss(args.eps, args.step_size, args.num_iters, args.grad_sign, args.sanity_check)
#     elif option == 'trades':
#         return TradesAdvLoss(args.eps, args.step_size, args.num_iters, args.clip_min, args.clip_max, 
#                                 args.rs, args.use_alpha_scheduler, args.sanity_check)
#     elif option == 'simba':  # simple-blackbox-attack
#         return SimBAAdvLoss(args.eps, args.num_iters, args.targeted, args.sanity_check)
#     else:
#         raise ValueError("Invalid adv_loss type")

# def get_feedback_loss(args, device):
#     from losses import ContrastiveLoss, TriSoftNearestNeighborsLoss, TriMMDLoss, TriWassersteinLoss, TriEntropyLoss
#     option = args.feedback
#     if option == 'none':
#         return None
#     elif option == 'contrast':
#         return ContrastiveLoss(args.temperature, args.margin, args.class_wise, args.generalize, device)
#     elif option == 'snn':
#         return TriSoftNearestNeighborsLoss(args.temperature, args.generalize, device)
#     elif option == 'mmd':
#         assert args.format in ['diff', 'ratio'], "Invalid format for MMD loss"
#         return TriMMDLoss(args.format, args.kernel_type, args.kernel_mul, args.kernel_num, 
#                             args.fix_sigma, args.class_wise, args.generalize, device)
#     elif option == 'ot':
#         assert args.format in ['diff', 'ratio'], "Invalid format for OT loss"
#         return TriWassersteinLoss(args.format, args.class_wise, args.generalize, device)
#     elif option == 'entropy':
#         # assert args.format in ['dist', 'mean'], "Invalid format for entropy distance loss"
#         return TriEntropyLoss('dist', args.class_wise, args.generalize, device)
#     else:
#         raise ValueError("Invalid feedback type")

def model_testing(model, data, data_retain, data_unlearn, batch_size, num_workers, 
                  device, transform, forget_classes, logger):
    def __data_test__(model, data, batch_size, num_workers, device, transform=None):
        from learner import complete_test
        test_loader = DataLoader(data, batch_size, False, num_workers=num_workers)
        return complete_test(model, test_loader, device, transform)    #  acc, auc
    
    ## EVALUATION: using the test data to evaluate the model
    if forget_classes is None:
        train_acc, train_auc =  __data_test__(model, data['train'], batch_size, num_workers, device, transform)
        train_retain_acc, train_retain_auc = __data_test__(model, data['retain'], batch_size, num_workers, device, transform)
        valid_acc, valid_auc =  __data_test__(model, data['valid'], batch_size, num_workers, device, transform)
        test_acc, test_auc =  __data_test__(model, data['test'], batch_size, num_workers, device, transform)
        forget_acc, forget_auc = __data_test__(model, data['forget'], batch_size, num_workers, device, transform)
        retain_acc, retain_auc = __data_test__(model, data_retain, batch_size, num_workers, device, transform)
        logger.info(
            " ###### [Train, Train_rem, Valid, Test, Forget, Remain-Data] "
            + f"  Acc: {train_acc:.4f}, {train_retain_acc:.4f}, {valid_acc:.4f}, {test_acc:.4f}, {forget_acc:.4f}, {retain_acc:.4f} ######"
        #     + f"| Auc: {train_auc:.4f}, {valid_auc:.4f}, {test_auc:.4f}, {forget_auc:.4f}, {retain_auc:.4f}, {retain_auc:.4f}###### ")
        )
    else:
        train_retain_acc, _ =  __data_test__(model, data['retain'], batch_size, num_workers, device, transform)
        train_forget_acc, _ =  __data_test__(model, data['forget'], batch_size, num_workers, device, transform)
        test_retain_acc, _ = __data_test__(model, data['test'], batch_size, num_workers, device, transform)
        test_forget_acc, _ = __data_test__(model, data_unlearn['test'], batch_size, num_workers, device, transform)
        valid_retain_acc, _ = __data_test__(model, data['valid'], batch_size, num_workers, device, transform)
        valid_forget_acc, _ = __data_test__(model, data_unlearn['valid'], batch_size, num_workers, device, transform)
        
        logger.info(" ###### Remain [Train, Valid, Test]:"
                            + f"  Acc: {train_retain_acc:.4f}, {valid_retain_acc:.4f}, {test_retain_acc:.4f} ######")
        logger.info(" ###### Forget [Train, Valid, Test]:"
                            + f"  Acc: {train_forget_acc:.4f}, {valid_forget_acc:.4f}, {test_forget_acc:.4f} ######")


if __name__ == "__main__":
    args = parse_args()
    args.unlearn_method = 'genun'
    # args.model_path = "./outs/cifar10_-1_genun/model_bases/cifar10_resnet18/ORG/ORG_GENE_M_lr-0.0002.pt"

    ## set output path
    outs = SingletonString()
    outs.content = args.out_dir
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    from data_tool import DataLoaderTool, DataStore, construct_data
    DataStore.create_basic_folders()
    seed_everything(args.seed)
    
    logpath = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(logpath, exist_ok=True)
    logger = create_logger(logpath, f'main_{args.protocal}_' + args.logname)
    logger.info(" +++++++++++ Start Robust Unlearning Program +++++++++++ ")
    logger.info(args)
    
    ### datasert blocks
    transform = None
    dataset_conf = { "num_samples": args.num_samples, "select_classes": args.select_classes}
    trans_params = '_'.join([args.preproc_train_transform, args.preproc_test_transform, args.online_train_aug, args.online_forget_aug])
    
    logger.info("load dataset and transform")
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    if args.arch in ['densenet', 'vit']:   # 'resnet50',
        dt_size = 224
        feature_dims = (3, dt_size, dt_size)
    tran_process = Transforms(dt_mean, dt_std, dt_size)
    ## pre-processing transform setup
    train_trans = tran_process.get_transform(args.preproc_train_transform, num_classes)
    test_trans = tran_process.get_transform(args.preproc_test_transform, num_classes)
    train_data, test_data = DataLoaderTool.load_dataset(args.dataset, train_trans, test_trans, **dataset_conf)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    data, data_unlearn, data_retain = construct_data(train_data, test_data, train_size=args.num_samples, 
                valid_size=args.valid_size, forget_size=args.num_to_forget, forget_classes=args.forget_classes)
    
    ##  import classes after setting the out path
    from data_tool import flatten_subset
    from tool import PathGenerator
    # from models import DirectUnlearn, RobustUnlearn, RobustForget
    from models_genun import GeneModUnlearn

    # logger.info("Get adversarial and feedback loss")
    adv_losser = 'none' #get_adv_loss(args)
    feed_losser = 'none' #args.feedback
    # if args.adv_loss in ['vat', 'fgsm']:
    #     if args.online_train_aug != "none" and args.preproc_train_transform == "none":
    #         args.online_train_aug = "normal"
    #     if args.online_forget_aug != "none"  and args.preproc_train_transform == "none":
    #         args.online_forget_aug = "test"

    ## load original model and test
    logger.info(f"Load original model from @ {args.model_path}")
    # init_mod_ = None
    # init_mod_ = get_model(args.arch, feature_dims, num_classes, pretrained=True).to(device)
    model_ = load_model(args.model_path).to(device)

    ## pre-evaluation of the original model
    logger.info(" +++++++++++ Evaluation: Original model  +++++++++++ ")
    model_testing(model_, data, data_retain, None, args.batch_size, args.num_workers, device, transform, None, logger)
    
    ## apply different unlearn models
    logger.info("Initialize unlearn model")
    mu_er = GeneModUnlearn(logpath, args.logname, args.out_dir, name=f'model_{args.protocal}')

    # unlearner = None
    # if args.protocal in ['DU']:
    #     unlearner = DirectUnlearn
    # elif args.protocal in ['FT', 'LASTK']:
    #     unlearner = RobustUnlearn
    # elif args.protocal in ['GA', 'RL']:
    #     unlearner = RobustForget
    # else:
    #     raise ValueError("Invalid protocal")

    ## unlearn model initialization
    # robun = unlearner(logpath, args.logname, args.out_dir, out_name=f'model_{args.protocal}_S')
    # robun.logger.info(args)
    # logger.info(f" +++++++  Selected top-K informative samples: {args.top_k}")
    
    # model setting
    logger.info("Set parameters and basic configurations")
    
    # 支持多学习率尝试
    lr_list = [1e-5, 2e-3]  # 可自行扩展
    for lr in lr_list:
        args.lr = lr
        logger.info(f"==== Running GenUn with lr={lr} ====")
        mu_er.set_params(**vars(args))
        mu_er.config_transform(dt_mean, dt_std, dt_size)
        
        # ### set data augmentation for online training and forgetting
        # if args.online_train_aug in ["random", "randaug", "augmix", "autoaug", "autoafn", "cutout", "randerase"]:
        #     num_cls = len(np.unique(data['retain'].dataset.targets))
        #     data['retain'].dataset.transform = processor.get_transform(args.online_train_aug, num_cls)
        # else:
        #     robun.config_transform(dt_mean, dt_std, dt_size)
        # if args.online_forget_aug != "none":
        #     num_cls = len(np.unique(data['forget'].dataset.targets))
        #     data['forget'].dataset.transform = processor.get_transform(args.online_forget_aug, num_cls)
        
        if args.sample_ratio > 0 and args.sample_ratio < 1:
            from torch.utils.data import Subset
            num_sams = int(len(data['retain']) * args.sample_ratio)
            data['retain'] = flatten_subset(Subset(data['retain'], np.random.choice(len(data['retain']), num_sams, replace=False)))
            logger.info(f" +++++++++++ Remain data reduced to {num_sams} +++++++++++ ")

        ## set dataset for unlearning
        logger.info("Set unlearning data")
        mu_er.set_data(data, num_classes, args.batch_size, args.num_workers)
        ## evaluation the original model
        # robun.evaluation(mod_, args.lossfn, device, 'none')
        acc_org, auc_org = mu_er.model_evaluate(model_, test_loader, device)

        ## perform forget-sample unlearn
        logger.info(" +++++++++++++++ Start Unlearning (in process) +++++++++++++++")
        ckpt_path = PathGenerator.get_checkpoint_path(f"{args.dataset}_{args.num_samples}_{args.arch}")
        unlearn_mod, ckpt = mu_er.unlearn(
            model_, None, None, args.patience, args.scheduler, args.optim,
            args.online_train_aug, args.online_forget_aug, device, ckpt_path, 
            model_type=args.arch, over_forget=args.over_forget, suffix=trans_params,
        )

        # if unlearner == DirectUnlearn:
        #     feed_losser = args.feedback
        #     unlearn_mod, ckpt = mu_er.unlearn(
        #         model_, init_mod_, adv_losser, args.feedback, args.patience, args.lossfn,
        #         args.scheduler, args.optim, args.online_train_aug, args.online_forget_aug, 
        #         device, ckpt_path, is_finetune=args.finetune_over_retain, model_type=args.arch, 
        #         forget_classes=args.forget_classes )
        # elif unlearner == RobustUnlearn:
        #     feed_losser = get_feedback_loss(args, device)
        #     unlearn_mod, ckpt = mu_er.unlearn(
        #         model_, args.protocal, adv_losser, feed_losser, args.patience, args.lossfn, 
        #         args.scheduler, args.optim, args.online_train_aug, args.online_forget_aug, 
        #         device, ckpt_path, suffix=trans_params, ast_k=args.last_k, re_init=args.re_init, 
        #         ltop_k=args.top_k, model_type=args.arch, forget_classes=args.forget_classes )
        # elif unlearner == RobustForget:
        #     feed_losser = get_feedback_loss(args, device)
        #     unlearn_mod, ckpt = mu_er.unlearn(
        #         model_, args.protocal, args.finetune_over_retain, adv_losser, feed_losser, 
        #         args.patience, args.lossfn, args.scheduler, args.optim, 
        #         args.online_train_aug, args.online_forget_aug, device, ckpt_path, 
        #         model_type=args.arch, suffix=trans_params, forget_classes=args.forget_classes )
        # else:
        #     raise ValueError("Invalid unlearn method")

        modfn = f"{args.arch}_{args.protocal}_lr-{lr}_{trans_params}_adv-{adv_losser}_fdbk-{feed_losser}_overforget-{args.over_forget}_{mu_er._suffix_}_best.pt" 
        modout_path = os.path.join(mu_er.path['model'], modfn)
        save_model(unlearn_mod, modout_path)
        logger.info(f" ++++++++++++++++++ Model saved @ {modout_path} done! +++++++++++++++++++")    

        ## evaluation the unlearned model
        from learner import complete_test
        logger.info("\n" + "++++++++"*5)
        logger.info(" +++ Evaluation: Unlearned model test +++")
        # robun.evaluation(unlearn_mod, args.lossfn, device, "none")
        for sub_dat in ['train', 'retain', 'valid', 'test', 'forget']: # data.items():  ## 
            t_loader = DataLoader(data[sub_dat], args.batch_size, False, num_workers=args.num_workers)
            acc_dt, auc_dt = complete_test(unlearn_mod, t_loader, device, transform)
            logger.info(f"##### [ {sub_dat:6} ]: ACC: {acc_dt:.4f}")   # , AUC: {auc_dt:.4f}

        acc_unlearn, auc_unlearn = mu_er.model_evaluate(unlearn_mod, test_loader, device)
        logger.info(f"Accuracy: Original: {acc_org:.4f} --> Unlearned: {acc_unlearn:.4f}")

        logger.info(" +++++++++++ Unlearned Model Test +++++++++++ ")
        # logger.info(f" +++++++++++ Forget classes: {args.forget_classes} +++++++++++ ")
        model_testing(unlearn_mod, data, data_retain, data_unlearn, args.batch_size, args.num_workers, 
                      device, transform, args.forget_classes, logger)
        logger.info("********************************* \n\n")
    print("Done!")

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from logger import create_logger
from model_deep import get_model
from transforms import Transforms
from const import SingletonString
from utils import argparse2bool, list_of_ints, load_model, save_model, seed_everything

float_formatter = "{:.4f}".format

# ========== 1. JSD 函数 ==========
def JSDiv(p, q):
    """
    计算两个概率分布之间的Jensen-Shannon散度
    p, q: [B, C] 概率分布（已softmax）
    return: scalar 0 ~ ln(2)
    """
    m = 0.5 * (p + q)                                   
    # 注意：F.kl_div的输入顺序是(log_target, input)
    kl_pm = F.kl_div(torch.log(m.clamp_min(1e-12)), p, reduction='batchmean')  
    kl_qm = F.kl_div(torch.log(m.clamp_min(1e-12)), q, reduction='batchmean')  
    return 0.5 * (kl_pm + kl_qm)

# ========== 2. ZRF 函数 ==========
@torch.no_grad()
def UnLearningScore(student_model, teacher_model, forget_dl, device):
    """
    计算遗忘保留分数 (Zhu et al., 2023)
    越接近1表示遗忘效果越好
    """
    preds_s, preds_t = [], []
    for x, *_ in forget_dl:
        x = x.to(device)
        preds_s.append(F.softmax(student_model(x), dim=1).cpu())
        preds_t.append(F.softmax(teacher_model(x), dim=1).cpu())
    preds_s = torch.cat(preds_s, dim=0)
    preds_t = torch.cat(preds_t, dim=0)
    jsd = JSDiv(preds_s, preds_t).item()
    return 1 - jsd 

# ========== 3. 激活距离函数 ==========
@torch.no_grad()
def activation_distance(model1, model2, dataloader, device):
    """
    计算两个模型在相同输入上输出的激活距离
    """
    distances = []
    for x, *_ in dataloader:
        x = x.to(device)
        p1 = F.softmax(model1(x), dim=1)
        p2 = F.softmax(model2(x), dim=1)
        # 计算L2距离
        distances.append(torch.norm(p1 - p2, p=2, dim=1).cpu())
    return torch.cat(distances).mean().item()


def parse_args():
    parser = argparse.ArgumentParser(description='Robust Unlearn Model')
    parser.add_argument('--logname', default='trail', help='name of output log file')
    parser.add_argument('--log_dir', default='.', help='Folder where all logs are stored (default: .)')
    parser.add_argument('--out_dir', default='./outs/tiny_imagenet_-1/', help='Folder where all outputs are stored (default: .)') # 修改为tiny_imagenet
    parser.add_argument('--exp_name', default='rob_log', help='Subfolder where all logs are stored (default: logs)')
    parser.add_argument('--verbose', type=argparse2bool, default=True, help='whether to print the log')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')  # 修改为0
    parser.add_argument('--cuda', type=int, default=1, help='cuda device (default: 0)')
    parser.add_argument('--cuda_t', type=int, default=1, help='cuda device for teacher-model (default: 0)')
    
    ## dataset and model related parameters
    parser.add_argument('--dataset', type=str, default='tiny_imagenet',  # 修改为tiny_imagenet
                        choices=['mnist', 'cifar10', 'cifar100', 'stl10', 'imagenet', 'vggface2', 'svhn', 'fashionmnist', 'tiny_imagenet'],
                        help='dataset for training (default: tiny_imagenet)')
    parser.add_argument('--select_classes', type=list_of_ints, default=None, help='selected classes for small dataset (default: None)')  # 修改为None
    parser.add_argument('--num_samples', type=int, default=-1, help='number of samples per class selected for small dataset')
    parser.add_argument('--num_workers', type=int, default=32, metavar='N', help='number of workers for data loading (default: 32)')  # 增加workers
    parser.add_argument('--valid_size', type=int, default=10000, metavar='V', help='number of validation samples (default: 10000)')  # 修改为10000
    parser.add_argument('--top_k', type=int, default=-1, metavar='K', help='number of selected informative samples (default: 5000)')
    parser.add_argument('--num_to_forget', type=int, default=10000, metavar='Nf', help='number of samples to forget (default: 10000)')  # 修改为10000
    parser.add_argument('--forget_classes', type=list_of_ints, default=None, metavar='Cf', help='classes to forget (default: None)')
    parser.add_argument('--sample_ratio', type=float, default=0.5, help='sample ratio for the retain dataset (default: 0.5)')
    
    parser.add_argument('--arch', type=str, default='resnet50', # 修改为resnet50
                        choices=['mlps', 'logistic', 'simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'densenet', 'vgg', 'vit'],
                        help='model architecture (default: resnet50)')
    parser.add_argument('--model_path', type=str, default="./outs/tiny_imagenet_-1//model_bases/tiny_imagenet_resnet50/ORG/tiny_imagenet_resnet50_seed-0_Nf-10000_ep-50_bs-512_lr-[1e-05-0_0001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth", help='path of the model to unlearn')
    parser.add_argument('--sequential', type=argparse2bool, default=False, help='whether to use sequential model')
    
    ## unlearning related parameters
    parser.add_argument('--epochs', type=int, default=7, metavar='E', help='number of epochs for unlearning (default: 20)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='B', help='input batch size for training (default: 256)') 
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate (default: 0.001)')
    parser.add_argument('--lr_feedback', type=float, default=0.0002, help='initial learning rate for feedback (default: 0.001)')
    parser.add_argument('--lam_reg', type=float, default=1.0, help='regularization parameter for adversarial part (default: 0.1)')    
    parser.add_argument('--strength', type=float, default=1.0, help='strength of the feedback (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay (default: 0.0005)')
    parser.add_argument('--class_wise', type=argparse2bool, default=True, help='whether to use class-wise feedback')
    parser.add_argument('--generalize', type=argparse2bool, default=True, help='whether to contain generalization loss')
    parser.add_argument('--save_checkpoint', type=argparse2bool, default=False, help='whether to save the checkpoint')
    
    ## running parameter
    parser.add_argument('--protocal', type=str, default='DISTILL', choices=['FT', 'LASTK', 'RL', 'GA', 'DISTILL'])
    parser.add_argument('--lossfn', type=str, default='ce', choices=['ce', 'bce', 'mse'])
    parser.add_argument('--optim', type=str, default="Adam", choices=['Adam', 'SGD'])
    parser.add_argument('--patience', type=int, default=30, help='patience for early stopping (default: 15)')
    parser.add_argument('--scheduler', default='CosineAnnealingWarmRestarts', #'CosineAnnealingLR', #'None', # 'LRScheduler', #
                        choices = ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR', 'LRScheduler', 'None'],
                        help='Pytorch Scheduler name: (default: The one used for train')

    parser.add_argument('--no_reg_epochs', type=int, default=0, help='number of epochs without regularization (default: 0)')
    parser.add_argument('--regularizer', type=str, default='none', choices=['none', 'l1', 'l2', 'l1l2'], help='regularizer type')
    parser.add_argument('--gamma', type=float, default=0.0, help='regularization parameter (default: 0.0)')
    parser.add_argument('--dynamic_regularization', type=argparse2bool, default=False, help='whether to use dynamic regularization')
    
    ## protocal parameters
    parser.add_argument('--last_k', type=int, default=3, 
                        help="Number of last layers to be used for retraining for unlearning (default: 1)")
    parser.add_argument('--re_init', type=argparse2bool, default=True, 
                        help="Whether to re-initialize the weight of last layers for LastKlayer unlearning")   
    
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
    parser.add_argument("--online_forget_aug", type=str, default="none", 
                        help="data augmentation for forgetting",
                        choices=["none", "normal", "baseaug",  "randerase", 
                                 "cutout", "mixup", "cutmix", "autoaug", "augmix", "randaug", "test" ],)
    
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha for distillation loss")
    parser.add_argument("--T", type=float, default=5, help="temperature for distillation loss")
    parser.add_argument("--num_layers", type=int, default=10, help="number of layers to be used for distillation")
    
    ## adversarial loss related parameters
    parser.add_argument('--adv_loss', type=str, default='none',
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
    parser.add_argument('--feedback', type=str, default='none', 
                        choices=['none', 'contrast', 'snn', 'mmd', 'ot', 'entropy'], help='feedback loss type')
    parser.add_argument('--format', type=str, default='diff', choices=['diff', 'ratio'], help='format for feedback loss (default: diff)')
    parser.add_argument('--dynamic_weight', type=argparse2bool, default=False, help='dynamic_weight for feedback loss (default: False)')
    parser.add_argument('--margin', type=float, default=1.0, help='margin for contrastive loss (default: 1.0)')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature for adversarial loss (default: 0.05)')
    parser.add_argument('--kernel_type', type=str, default='rbf', help='kernel_type for MMDLoss (default: rbf)')
    parser.add_argument('--kernel_mul', type=float, default=2.0, help='kernel_mul for MMDLoss (default: 2.0)')
    parser.add_argument('--kernel_num', type=int, default=5, help='kernel_num for MMDLoss (default: 5)')
    parser.add_argument('--fix_sigma', type=argparse2bool, default=False, help='fix_sigma for MMDLoss (default: False)')

    ## 新增：金标模型路径参数
    parser.add_argument('--gold_model_path', type=str, default='./outs/tiny_imagenet_-1/model_bases/tiny_imagenet_resnet50/RT/tiny_imagenet_resnet50_seed-0_Nf-10000_ep-20_bs-512_lr-[1e-05-0_0001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth', help='path of the gold model for evaluation metrics')
    
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
    args.unlearn_method = 'disun'
    ## set output path
    outs = SingletonString()
    outs.content = args.out_dir
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    device_aux = torch.device(f"cuda:{args.cuda_t}" if torch.cuda.is_available() else "cpu")
    
    # 原有注释掉的路径...
    # args.model_path = "./outs/cifar10_-1/model_bases/cifar10_resnet18/ORG/" + \
    #     "cifar10_resnet18_seed-3407_Nf-5000_ep-20_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-testregular_0.0_sched-None.pth"

    # 设置金标模型路径（如果没有提供，使用原始模型作为金标模型）
    if not args.gold_model_path:
        args.gold_model_path = args.model_path
    
    from data_tool import DataLoaderTool, DataStore, construct_data
    DataStore.create_basic_folders()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    logpath = os.path.join(args.log_dir, args.exp_name)
    logger = create_logger(logpath, 'main_disun_' + args.logname)
    logger.info(args)

    logger.info("load dataset and transform")
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    if args.arch in ['resnet50', 'densenet', 'vit']:
        dt_size = 224
        feature_dims = [3, dt_size, dt_size]
    processor = Transforms(dt_mean, dt_std, dt_size)
    train_transform = processor.get_transform(args.preproc_train_transform, num_classes)
    test_transform = processor.get_transform(args.preproc_test_transform, num_classes)
    transform = None

    num_samples = args.num_samples
    dataset_conf = { "num_samples": num_samples, "select_classes": args.select_classes}
    train_data, test_data = DataLoaderTool.load_dataset(args.dataset, train_transform, test_transform, **dataset_conf)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    data, data_unlearn, data_remain = construct_data(train_data, test_data, train_size=args.num_samples, valid_size=args.valid_size, 
                                                     forget_size=args.num_to_forget, forget_classes=args.forget_classes)

    ##  import after setting the out path
    from data_tool import flatten_subset
    from tool import PathGenerator as pather
    from models_disun import DistillUnlearn

    logger.info("Get adversarial and feedback loss")
    adv_loss = None #get_adv_loss(args)    
    feedback_loss = None #get_feedback_loss(args, device)
    if args.adv_loss in ['vat', 'fgsm']:
        if args.online_train_aug != "none" and args.preproc_train_transform == "none":
            args.online_train_aug = "normal"
        if args.online_forget_aug != "none"  and args.preproc_train_transform == "none":
            args.online_forget_aug = "normal"

    ## load original model and test
    logger.info(f"Load original model from @ {args.model_path}")
    init_mod_ = get_model(args.arch, feature_dims, num_classes, pretrained=True).to(device)
    model_ = load_model(args.model_path).to(device_aux)
    
    ## 加载金标模型
    logger.info(f"Load gold model from @ {args.gold_model_path}")
    gold_model = load_model(args.gold_model_path).to(device)

    ## unlearn model initialization
    logger.info("Init unlearn model")
    disun = DistillUnlearn(logpath, args.logname, args.out_dir)
    disun.logger.info(args)
    logger.info(f" +++++++  Selected top-K informative samples: {args.top_k}")
    
    logger.info("Set parameters and transform configurations")
    # model setting
    disun.set_params(
        args.epochs, args.batch_size, args.lr, args.lr_feedback,
        args.lam_reg, args.strength, args.weight_decay, args.momentum,
        args.save_checkpoint, args.class_wise, args.seed, args.dynamic_weight,
        args.no_reg_epochs, args.regularizer, args.gamma, args.dynamic_regularization
    )

    trans_params = '_'.join([args.preproc_train_transform, args.preproc_test_transform, args.online_train_aug, args.online_forget_aug])

    logger.info("EVALUATION: Original model over original dataset")
    train_acc, train_auc = data_model_test(model_, data['train'], args.batch_size, args.num_workers, device_aux, transform)
    retain_acc, retain_auc = data_model_test(model_, data['retain'], args.batch_size, args.num_workers, device_aux, transform)
    valid_acc, valid_auc = data_model_test(model_, data['valid'], args.batch_size, args.num_workers, device_aux, transform)
    test_acc, test_auc = data_model_test(model_, data['test'], args.batch_size, args.num_workers, device_aux, transform)
    forget_acc, forget_auc = data_model_test(model_, data['forget'], args.batch_size, args.num_workers, device_aux, transform)
    remain_acc, remain_auc = data_model_test(model_, data_remain, args.batch_size, args.num_workers, device_aux, transform)
    logger.info(" ###### [Train, ReTain, Valid, Test, Forget, Remain-Data] "
                        + f"  Acc: {train_acc:.4f}, {retain_acc:.4f}, {valid_acc:.4f}, {test_acc:.4f}, {forget_acc:.4f}, {remain_acc:.4f} ######")
    
    if args.forget_classes is not None:
        logger.info(f" +++++++++++ Forget Class: {args.forget_classes} +++++++++++ ")
        for type_, dt_ in data_unlearn.items():
            acc_dt, auc_dt = data_model_test(model_, dt_, args.batch_size, args.num_workers, device, transform)
            logger.info(f"##### [ {type_:6} ]: ACC: {acc_dt:.4f}, AUC: {auc_dt:.4f}")

    ### set data augmentation for online training and forgetting
    if args.online_train_aug in ["random", "randaug", "augmix", "autoaug", "autoafn", "cutout", "randerase"]:
        num_cls = len(np.unique(data['retain'].dataset.dataset.targets))
        data['retain'].transform = processor.get_transform( args.online_train_aug, num_cls)
    else:
        disun.config_transform(dt_mean, dt_std, dt_size)
    if args.online_forget_aug != "none":
        num_cls = len(np.unique(data['forget'].dataset.dataset.targets))
        data['forget'].transform = processor.get_transform( args.online_forget_aug, num_cls)
    
    ### retain data downsampling for faster unlearning
    if args.sample_ratio > 0 and args.sample_ratio < 1:
        from torch.utils.data import Subset
        num_sams = int(len(data['retain']) * args.sample_ratio)
        data['retain'] = flatten_subset(Subset(data['retain'], np.random.choice(len(data['retain']), num_sams, replace=False)))
        logger.info(f" +++++++++++ Remain data reduced to {num_sams} +++++++++++ ")
    
    mod_ = model_ #.to(device_aux)
    ## set dataset for unlearning
    logger.info("Set data for unlearning")
    disun.set_data(data, num_classes, args.batch_size, args.num_workers)
    ## evaluation the original model
    acc_org, auc_org = disun.model_evaluate(mod_, test_loader, device_aux)

    ## perform forget-sample unlearn
    logger.info("Perform unlearning")
    ckpt_path = pather.get_checkpoint_path(f"{args.dataset}_{args.num_samples}_{args.arch}")
    
    # DistillUnlearn
    ref_mod_params_ = list(mod_.parameters())
    mod_params_ = list(init_mod_.parameters())
    for i in range(min(args.num_layers * 2, len(ref_mod_params_))):
        mod_params_[i].data.copy_(ref_mod_params_[i].data)

    disun._suffix_ = f"{disun._suffix_}_alpha-{args.alpha}_temp-{args.T}_num_layers-{args.num_layers}_sample_r-{args.sample_ratio}"
    logger.info(f" +++++++++++ start Unlearning +++++++++++ ")
    if args.sequential:
        logger.info("construct softlabel dataset based on the teacher's prediction")
        train_loader = DataLoader(data['retain'], args.batch_size, True, num_workers=args.num_workers)
        teacher_data = disun.construct_softlabel_dataset(mod_, train_loader, device)
        train_loader = DataLoader(teacher_data, args.batch_size, True, num_workers=args.num_workers)
        del mod_, model_
        logger.info("Unlearn the model using the sequential method")
        unlearn_mod, ckpt = disun.unlearn_sequential(
            init_mod_, train_loader, adv_loss, args.patience, args.scheduler, args.optim, args.online_train_aug,
            alpha=args.alpha, temperature=args.T)
        disun._suffix_ += "_seq"
    else:
        logger.info("Unlearn the model with distillation")
        unlearn_mod, ckpt = disun.unlearn(
            mod_, init_mod_, adv_loss, feedback_loss, args.patience,
            args.scheduler, args.optim, args.online_train_aug, args.online_forget_aug, 
            device, device_aux, suffix=trans_params, 
            model_type=args.arch, checkpoint_path=ckpt_path,
            alpha=args.alpha, temperature=args.T, 
        )

    outfn = f"{args.arch}_{args.protocal}_{trans_params}_adv-{adv_loss}_feedback-{feedback_loss}_{disun._suffix_}_best.pt"
    out_path = os.path.join(disun.path['model'], outfn)
    save_model(unlearn_mod, out_path)
    logger.info(f" ++++++++++++++++++ Model saved @ {out_path} +++++++++++++++++++")    

    logger.info("Evaluation: Unlearned model test")
    for type_ in ['train', 'retain', 'valid', 'test', 'forget']:
        acc_dt, auc_dt = data_model_test(unlearn_mod, data[type_], args.batch_size, args.num_workers, device, transform)
        logger.info(f"##### [ {type_:6} ]: ACC: {acc_dt:.4f}")

    ## evaluation the unlearned model
    acc_unlearn, auc_unlearn = disun.model_evaluate(unlearn_mod, test_loader, device)
    logger.info(f"Accuracy: Original: {acc_org:.4f} --> Unlearned: {acc_unlearn:.4f}")

    ## EVALUATION: using the test data to evaluate the model
    if args.forget_classes is None:
        logger.info(" +++++++++++ Unlearned Model Test +++++++++++ ")
        train_acc, train_auc =  data_model_test(unlearn_mod, data['train'], args.batch_size, args.num_workers, device, transform)
        train_retain_acc, train_retain_auc = data_model_test(unlearn_mod, data['retain'], args.batch_size, args.num_workers, device, transform)
        valid_acc, valid_auc =  data_model_test(unlearn_mod, data['valid'], args.batch_size, args.num_workers, device, transform)
        test_acc, test_auc =  data_model_test(unlearn_mod, data['test'], args.batch_size, args.num_workers, device, transform)
        forget_acc, forget_auc = data_model_test(unlearn_mod, data['forget'], args.batch_size, args.num_workers, device, transform)
        retain_acc, retain_auc = data_model_test(unlearn_mod, data_remain, args.batch_size, args.num_workers, device, transform)
        logger.info(
            " ###### [Train, Train_rem, Valid, Test, Forget, Remain-Data] "
            + f"  Acc: {train_acc:.4f}, {train_retain_acc:.4f}, {valid_acc:.4f}, {test_acc:.4f}, {forget_acc:.4f}, {retain_acc:.4f} ######"
        )
    else:
        logger.info(f" +++++++++++ Forget classes: {args.forget_classes} +++++++++++ ")
        train_retain_acc, _ =  data_model_test(unlearn_mod, data['retain'], args.batch_size, args.num_workers, device, transform)
        train_forget_acc, _ =  data_model_test(unlearn_mod, data['forget'], args.batch_size, args.num_workers, device, transform)
        test_remain_acc, _ = data_model_test(unlearn_mod, data['test'], args.batch_size, args.num_workers, device, transform)
        test_forget_acc, _ = data_model_test(unlearn_mod, data_unlearn['test'], args.batch_size, args.num_workers, device, transform)
        valid_remain_acc, _ = data_model_test(unlearn_mod, data['valid'], args.batch_size, args.num_workers, device, transform)
        valid_forget_acc, _ = data_model_test(unlearn_mod, data_unlearn['valid'], args.batch_size, args.num_workers, device, transform)
        
        logger.info(" ###### Remain [Train, Valid, Test]:"
                            + f"  Acc: {train_retain_acc:.4f}, {valid_remain_acc:.4f}, {test_remain_acc:.4f} ######")
        logger.info(" ###### Forget [Train, Valid, Test]:"
                            + f"  Acc: {train_forget_acc:.4f}, {valid_forget_acc:.4f}, {test_forget_acc:.4f} ######")

    # ================= 新增：三大遗忘指标计算 =================
    logger.info("========== Unlearning Metrics ==========")
    
    # 创建随机初始化的教师模型用于ZRF计算
    random_teacher = get_model(args.arch, feature_dims, num_classes, pretrained=False).to(device)
    
    # 创建遗忘集的数据加载器
    forget_dl = DataLoader(data['forget'], batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers)
    
    # 计算三个指标
    with torch.no_grad():
        # 1. JSD (金标模型 vs 遗忘后模型)
        preds_g, preds_u = [], []
        for x, *_ in forget_dl:
            x = x.to(device)
            preds_g.append(F.softmax(gold_model(x), dim=1).cpu())
            preds_u.append(F.softmax(unlearn_mod(x), dim=1).cpu())
        preds_g = torch.cat(preds_g, dim=0)
        preds_u = torch.cat(preds_u, dim=0)
        jsd_val = JSDiv(preds_g, preds_u).item()
        
        # 2. 激活距离 (金标模型 vs 遗忘后模型)
        act_dist = activation_distance(gold_model, unlearn_mod, forget_dl, device)
        
        # 3. ZRF (遗忘后模型 vs 随机初始化教师模型)
        zrf_val = UnLearningScore(unlearn_mod, random_teacher, forget_dl, device)
        zrf_val_best = UnLearningScore(gold_model, random_teacher, forget_dl, device)

    
    logger.info(f"ActDist(gold,student): {act_dist:.4f}")
    logger.info(f"JSD(gold,student) : {jsd_val:.4f}")
    logger.info(f"ZRF(student,teacher) : {zrf_val:.4f}")   # 越接近越好
    logger.info(f"ZRF(gold,teacher) : {zrf_val_best:.4f}")
    

    # ================= 结束 =================
    
    logger.info("********************************* \n\n")
    print("Done!")
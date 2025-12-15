import os
import argparse
import time
from copy import deepcopy

import torch
from logger import create_logger
from transforms import Transforms
from const import SingletonString
from utils import argparse2bool, list_of_ints, load_model, mkdir, save_model, seed_everything


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['CUDA_COREDUMP_SHOW_PROGRESS'] = "1"
os.environ['CUDA_ENABLE_COREDUMP_ON_EXCEPTION']='1'
float_formatter = "{:.4f}".format


def parse_args():
    parser = argparse.ArgumentParser(description='Model learning and unlearning procudure')
    ######################### general parameters ################################
    parser.add_argument('--logname', default='trail', help='name of output log file')
    parser.add_argument('--log_dir', default='.', help='Folder where all logs are stored (default: .)')
    parser.add_argument('--out_dir', default='./outs/', help='Folder where all outputs are stored (default: .)')
    parser.add_argument('--exp_name', default='logs', help='Subfolder where all logs are stored (default: logs)')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--cuda', type=str, default=0, help="Choose the GPU device")
    # parser.add_argument('--verbose', type=argparse2bool, default=True, help='whether to print the log')
    
    ### dataset related parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100', 'stl10', 'imagenet', 'vggface2', 'svhn', 'fashionmnist','tiny_imagenet','imagenet'],
                        help='dataset for training (default: cifar10)')
    parser.add_argument('--select_classes', type=list_of_ints, default='3,8', 
                        help='selected classes for small dataset (default: 3,8)')
    parser.add_argument('--num_samples', type=int, default=-1, metavar='N', 
                        help='number of samples to use (default: -1 (all))')
    parser.add_argument('--valid_size', type=int, default=5000, metavar='V', 
                        help='number of validation samples (default: 5000)')
    parser.add_argument('--num_to_forget', type=int, default=5000, metavar='Nf', 
                        help='number of samples to forget (default: 5000)')
    parser.add_argument('--forget_classes', type=list_of_ints, default=None, metavar='Cf', 
                        help='classes to forget (default: None)')
        
    ## model related parameters
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['mlps', 'logistic', 'simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'densenet', 'vit', 'vgg','ALLCNN','resnet50'],
                        help='model architecture (default: resnet18)')
    parser.add_argument('--pretrained', type=argparse2bool, default=True, help='whether to use pretrained model')
    
    parser.add_argument('--unlearn_method', type=str, default='FT', # 'none', #'L1FT', #'salun', #
                choices=[ "ORG", "RT", "FT", "RL", "GA", "WGA", "LKL", "FRA", "UI", "CR", "L1FT", "SALUN", "salun",
                    'none', 'scratch', 'finetune', 'l1_sparse', 'random_label', 'negative_gradient', 'graident_ascent',
                    'weighted_gradient_ascent', 'lastklayer', 'fisherapprox', 'influence_unlearn', 'certified_remove'],
                help='unlearning method (default: none (learning original model from scratch))')
    parser.add_argument('--resume', type=str, default=None, help='path to the model to be unlearned')
    
    ######################### model (un-)learning related parameters ################################
    parser.add_argument('--lossfn', type=str, default='ce', choices=['ce', 'bce', 'mse'])
    parser.add_argument('--start_raw', type=argparse2bool, default=False, help='whether to start from raw sample')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--no-reg-epochs', type=int, default=0, metavar='N', help='number of epochs to train without regularization (default: 0)')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping (default: 30)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', 
                        help='input batch size for training (default: 256)')
    
    parser.add_argument('--optim', type=str, default="Adam", choices=['Adam', 'SGD'])
    parser.add_argument('--scheduler', default='None', #'CosineAnnealingWarmRestarts', 
                        choices = ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR', 'None'],
                        help='Pytorch Scheduler name: (default: The one used for train, in args_re')
    
    parser.add_argument('--maxlr', type=float, default=0.001, help='the maximum learning rate (default: 0.001)')
    parser.add_argument('--minlr', type=float, default=0.001, help='the minimum learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-05, help='weight decay (default: 0)')   
    parser.add_argument('--lamb', type=float, default=1e-6, help='regularization parameter (default: 1e-8)')

    ## model training with data augmentation or mixup
    parser.add_argument("--train_transform", type=str, default="normal", 
                        help="data augmentation for preprocessing the training data",
                        choices=["none", "normal", "random", "randerase", "augafn", "baseaug", "mixup",
                                 "cutout", "cutmix", "autoaug", "augmix", "randaug", "test"],)
    parser.add_argument("--test_transform", type=str, default="test", 
                        help="data augmentation for preprocessing the testing data",
                        choices=["none", "normal", "random", "randerase", "augafn", "baseaug", "mixup", 
                                 "cutout", "cutmix", "autoaug", "augmix", "randaug", "test"],)
    
    # parser.add_argument("--save_general_model", type=argparse2bool, default=False, help='whether save the well-generalize model (Acc_tr = Acc_val)')
    
    # parser.add_argument('--mixup', type=argparse2bool, default=False, help='using the mixup as the data augmentation')
    parser.add_argument('--mixup_type', default='none', 
                        choices = ['none', 'mixup', 'cutmix'], help='Online mixup type for data augmentation')
    parser.add_argument('--mixup_prob', type=float, default=1.0, metavar='P', 
                        help='Prob. for mixup (default: 1.0)')
    parser.add_argument('--mixup_alpha', default=1.0, type=float, help='mixup interpolation coefficient (default: 1.0)')

    ### for lastklayer unlearning
    parser.add_argument('--last_k', type=int, default=2, help="Number of last layers to be used for retraining / influence updating for unlearning")
    parser.add_argument('--re_init', type=argparse2bool, default=True, 
                        help="Whether to re-initialize the weight of last layers for LastKlayer unlearning")
    
    ## for last-k-layer retrain unlearning method
    # parser.add_argument('--maxL-rf', type=int, default=3, metavar='UL', help='Layers to retrain upperbound (default: 3)')
    # parser.add_argument('--minL-rf', type=int, default=1, metavar='LL', help='Layers to retrain lowerbound (default: 1)')
    # parser.add_argument('--stepL-rf', type=int, default=1, metavar='LS', help='Layers to retrain step size (default: 1)')
    
    ## for l1-sparse
    parser.add_argument('--regularization', type=str, default='none', choices=['none', 'l1', 'l2', 'l1+l2'], help='regularization type')
    parser.add_argument('--dynamic_regular', type=argparse2bool, default=False, help='whether to use dynamic regularization')
 
    ### for salun
    parser.add_argument('--mask_threshold', type=float, default=0.5, help='mask threshold for salun')
    
    ### for fisherapprox unlearning
    parser.add_argument('--fisher_type', type=str, default='golaker', choices=['golaker', 'variate'], 
                        help='Fisher type for Fisher Approximation')
    parser.add_argument('--alpha', type=float, default=0.2, help='noise ratio for WoodFisher')
    
    ### for influence unlearning
    parser.add_argument('--recursion_depth', type=int, default=2, help='recursion depth for influence unlearning')
    parser.add_argument('--damp', type=float, default=0.01, help='damping factor for influence unlearning')
    parser.add_argument('--scale', type=float, default=25.0, help='scale factor for influence unlearning')
    parser.add_argument('--r_averaging', type=int, default=1, help='recursion averaging for influence unlearning')
    
     ## for certificated remove model
    parser.add_argument('--std', type=float, default=10.0, help='standard deviation for objective perturbation')
    parser.add_argument('--train_mode', type=str, default='ovr', choices=['ovr', 'binary'], 
                        help="train mode (default: ovr) for certificated remove model's lr classifier")
    parser.add_argument('--train_seperate', type=argparse2bool, default=False, 
                        help="whether to train seperate classifier for each class one vs rest")
    parser.add_argument("--sample-rate", type=float, default=0.001, metavar="SR", 
                        help="sample rate used for batch construction (default: 0.001)", )
    # parser.add_argument('--num-steps', type=int, default=100, help='number of optimization steps')  ## with equal to epochs
    parser.add_argument('--tolerance', type=float, default=1e-4, help='tolerance for optimization')
    
    # for WGA
    parser.add_argument('--wga_beta', type=float, default=5.0, help='beta exponent for Weighted Gradient Ascent (default: 5.0)')
    
    
    ######################### defense related parameters ################################
    ## parameters used for DP defense (Privacy_Engine)
    parser.add_argument('--is_dp_defense', type=argparse2bool, default=False)
    # parser.add_argument('--top_k', type=int, default=4, choices=[0, 1, 2, 3, 4],  help=" 0 (label), 4 (no defense)")
    parser.add_argument("--delta", type=float, default=1e-5, metavar="D", help="Target delta (default: 1e-5)", )
    parser.add_argument("--sigma", type=float, default=1.2, metavar="S", help="Noise multiplier (default 1.0)", )
    parser.add_argument("-c", "--max-per-sample-grad_norm", type=float, default=1.0, metavar="C", 
                        help="Clip per-sample gradients to this norm (default 1.0)", )
    parser.add_argument("--secure_rng", action="store_true", default=False,
                        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost", )
    parser.add_argument("--epsilon", type=float, default=50.0, metavar="E", help="Target epsilon (default: 8.0)", )
    parser.add_argument("--clip", type=float, default=1.0, metavar="CL", help="Clipping bound (default: 1.0)", )
    
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
    name += f"_wd-{str(args.weight_decay).replace('.','_')}"
    name += f"_opt-{args.optim}"
    name += f"_trTrans-{args.train_transform}_tsTrans-{args.test_transform}"
    if args.lamb is not None:
        name += f"_regular_{args.lamb}"
    if args.mixup_type == "cutmix":
        name += f"_cmixp-{str(args.mixup_prob).replace('.','_')}"
        name += f"_cmixa-{str(args.mixup_alpha).replace('.','_')}"
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
    # args.resume = "./outs/cifar10_-1_genun/model_bases/cifar10_resnet18/ORG/ORG_GENE_M_lr-0.0002.pt"
    
    # args.resume = "./outs/cifar10_-1/model_bases/cifar10_resnet18/ORG/"+\
    #     "cifar10_resnet18_seed-3407_Nf-5000_ep-20_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-testregular_0.0_sched-None.pth"
    # args.resume = "./outs/cifar100_-1/model_bases/cifar100_vit/ORG/"+\
    #     "cifar100_vit_seed-0_Nf-5000_ep-40_bs-256_lr-[0_001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"
    
    # args.resume = "./outs/cifar100_-1/model_bases/cifar100_vit/ORG/"+\
    #       "cifar100_vit_seed-0_Nf-5000_ep-20_bs-256_lr-[0_0001-0_0002]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"
    
    
    #args.resume = "./outs/cifar10_-1_genun/model_bases/cifar10_resnet18/ORG/" + \
    #        "cifar10_resnet18_seed-0_Nf-5000_ep-25_bs-256_lr-[0_0001-0_0002]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"

    from data_tool import DataStore, DataLoaderTool, construct_data
    DataStore.create_basic_folders()
    logpath = os.path.join(args.log_dir, args.exp_name)
    mkdir(logpath)
    logger = create_logger(logpath, 'main_proc_unlearn_' + args.logname)
    logger.info(args)
    
    logger.info("Load dataset and transform")
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    if args.arch in ['resnet50', 'densenet', 'vit']:
        dt_size = 224
        feature_dims = [3, dt_size, dt_size]
    processor = Transforms(dt_mean, dt_std, dt_size)
    train_transform = processor.get_transform(args.train_transform, num_classes)
    test_transform = processor.get_transform(args.test_transform, num_classes)
    
    unlearn_method = args.unlearn_method
    ## load dataset based on the configuration
    dataset_conf = { "num_samples": args.num_samples, "select_classes": args.select_classes}
    train_data, test_data = DataLoaderTool.load_dataset(args.dataset, train_transform, test_transform, **dataset_conf)
    data, data_unlearn, data_retain = construct_data(train_data, test_data, train_size=args.num_samples, valid_size=args.valid_size, 
                                                     forget_size=args.num_to_forget, forget_classes=args.forget_classes)
    # 增加日志，输出各 split 的样本数
    logger.info(f"train: {len(data['train'])}, valid: {len(data['valid'])}, test: {len(data['test'])}, forget: {len(data['forget'])}, retain: {len(data['retain'])}")
    if args.forget_classes is not None:
        logger.info(f"data_unlearn['train']: {len(data_unlearn['train'])}, data_unlearn['valid']: {len(data_unlearn['valid'])}, data_unlearn['test']: {len(data_unlearn['test'])}")
    logger.info(f"data_retain: {len(data_retain)}")
    
    from unlearners import *
    from model_bases import DeepModels
    from schema import BasicUnlearnSchema
    from tool import complete_parameters
    from tool import PathGenerator as pather
    
    ## learn the model or conduct the unlearning
    extra_params = {'resume': args.resume}
    unlearn_schema:BasicUnlearnSchema = None
    _ext_name_ = ""
    schema_logname = args.logname + '_unlearning' # + unlearn_method
    if unlearn_method in ["ORG", "none", "original"]:
        extra_params['is_train'] = args.start_raw
        unlearn_schema = BuildLearn(logpath, schema_logname)
    elif unlearn_method in ["RT", "scratch", "retrain"]:
        unlearn_schema = UnlearnRetrain(logpath, schema_logname)
    elif unlearn_method in ["FT", "finetune"]:
        unlearn_schema = UnlearnFinetune(logpath, schema_logname)
    elif unlearn_method in ['L1FT', 'l1_sparse']:
        _ext_name_ = f"_l1_regular_{args.lamb}_dynamic_{args.dynamic_regular}"
        args.regularization = 'l1'
        logger.info(f" Set regularization to {args.regularization} with lambda: {args.lamb}")
        # args.dynamic_regular = True
        unlearn_schema = UnlearnSparseL1(logpath, schema_logname)
    elif unlearn_method in ["RL", "random_label"]:
        extra_params['num_classes'] = num_classes
        unlearn_schema = UnlearnRandomLabels(logpath, schema_logname)
    elif unlearn_method in ["SALUN", "salun"]:
        _ext_name_ = f"_maskth-{args.mask_threshold}"
        extra_params['mask_threshold'] = args.mask_threshold
        unlearn_schema = UnlearnSalunRL(logpath, schema_logname)
    elif unlearn_method in ["GA", "gradient_ascent", "negative_gradient"]:
        unlearn_schema = UnlearnGradientAscent(logpath, schema_logname)
    elif unlearn_method in ["WGA", "weighted_gradient_ascent"]:
        extra_params['beta'] = args.wga_beta
        _ext_name_ = f"_wga_beta-{str(args.wga_beta).replace('.', '_')}"
        unlearn_schema = UnlearnWeightedGradientAscent(logpath, schema_logname)
    elif unlearn_method in ["LKL", "lastklayer"]:
        _ext_name_ = f"_lastk-{args.last_k}_reinit-{args.re_init}"
        extra_params['last_k'] = args.last_k
        extra_params['re_init'] = args.re_init
        unlearn_schema = UnlearnLastKLayer(logpath, schema_logname)
    elif unlearn_method in ["FRA", "fisherapprox"]:
        _ext_name_ = f"_fishertype-{args.fisher_type}"
        extra_params['lamb'] = args.lamb
        extra_params['lossfn'] = args.lossfn
        extra_params['num_classes'] = num_classes
        extra_params['forget_classes'] = args.forget_classes
        extra_params['fisher_type'] = args.fisher_type
        unlearn_schema = UnlearnFisherApprox(logpath, schema_logname)
    elif unlearn_method in ["WF", "woodfisher"]:
        _ext_name_ = f"_alpha-{args.alpha}_Ns-{len(data['retain'])}"
        extra_params['alpha'] = args.alpha
        extra_params['lossfn'] = args.lossfn
        extra_params['Ns'] = len(data['retain'])
        unlearn_schema = UnlearnWoodFisher(logpath, schema_logname)
    elif unlearn_method in ["UI", "influence_unlearn"]:
        _ext_name_ = f"_recursion-{args.recursion_depth}_damp-{args.damp}_scale-{args.scale}_ravg-{args.r_averaging}_lastk-{args.last_k}"
        extra_params['recursion_depth'] = args.recursion_depth
        extra_params['damp'] = args.damp
        extra_params['scale'] = args.scale
        extra_params['r_averaging'] = args.r_averaging
        extra_params['last_k'] = args.last_k
        unlearn_schema = UnlearnInfluence(logpath, schema_logname)
    elif unlearn_method in ["CR", "certified_remove"]:
        extra_params['num_classes'] = num_classes
        extra_params['arch'] = args.arch
        extra_params['std'] = args.std
        extra_params['train_mode'] = args.train_mode  # 'ovr' or none 
        extra_params['train_seperate'] = args.train_seperate
        extra_params['sample_ratio'] = args.sample_rate
        extra_params['epochs'] = args.epochs
        extra_params['lamb'] = args.lamb
        extra_params['tol'] = args.tolerance
        unlearn_schema = UnlearnCertifiedRemoval(logpath, schema_logname)
        _ext_name_ = f"_trainmode-{args.train_mode}_trainsep-{args.train_seperate}_samp-{args.sample_ratio}_tol-{args.tolerance}"
    else:
        raise NotImplementedError
    
    logname_mod = args.logname + '_' + unlearn_schema.__get_name__()  # unlearn_method
    model_zero = DeepModels(args.arch, feature_dims, num_classes, logpath, logname_mod, args.pretrained)
    model_none = deepcopy(model_zero)
    
    ## EVALUATION: test the original model
    if unlearn_method not in ["ORG", "none", "original"]:
        model_zero.load(args.resume)
        model_zero._model.to(device)
        logger.info(" +++++++++++ Original Model Test +++++++++++ ")
        train_acc, train_auc = BasicUnlearnSchema._test_model_(model_zero, data['train'], args.batch_size, args.lossfn, device)
        train_retain_acc, train_retain_auc = BasicUnlearnSchema._test_model_(model_zero, data['retain'], args.batch_size, args.lossfn, device)
        valid_acc, valid_auc = BasicUnlearnSchema._test_model_(model_zero, data['valid'], args.batch_size, args.lossfn, device)
        test_acc, test_auc = BasicUnlearnSchema._test_model_(model_zero, data['test'], args.batch_size, args.lossfn, device)
        forget_acc, forget_auc = BasicUnlearnSchema._test_model_(model_zero, data['forget'], args.batch_size, args.lossfn, device)
        retain_acc, retain_auc = BasicUnlearnSchema._test_model_(model_zero, data_retain, args.batch_size, args.lossfn, device)
        
        logger.info(" ###### [Train, Valid, Test, Forget, Remain-Data] "
                    + f"  Acc: {train_acc:.4f}, {train_retain_acc:.4f}, {valid_acc:.4f}, {test_acc:.4f}, {forget_acc:.4f}, {retain_acc:.4f} ######")
                # + f"| Auc: {train_auc:.4f}, {train_retain_auc:.4f}, {valid_auc:.4f}, {test_auc:.4f}, {forget_auc:.4f}, {retain_auc:.4f} ###### ")

    ## the original model is used for unlearning
    model_ori = None
    if unlearn_method in ['none', 'original', 'scratch', 'retrain', "ORG", "RT"]:
        ## learning from scratch for training or re-training
        model_ori = model_none
    else:
        model_ori = model_zero
    
    ## perform unlearning
    ckpt_path = pather.get_checkpoint_path(f"{args.dataset}_{args.num_samples}_{args.arch}")
    model_params = complete_parameters(vars(args))
    model_params['num_classes'] = num_classes
    
    start_t = time.time()
    model_ori.parameter_config(**model_params)
    model_res = unlearn_schema.model_unlearn(model_ori, data, args.batch_size, device, ckpt_path, **extra_params)
    logger.info(f" +++++++++++ Unlearning Done [at {(time.time() - start_t):.2f} s]! +++++++++++ ")
    logger.info(f"+++++++++++++++ extra_params: {_ext_name_} ++++++++++++++++ ")
    logger.info(f"+++++++++++++++ model_params: {model_ori.params} ++++++++++++++++ ")
    
    ## save the unlearned model
    out_path = pather.get_target_model_path(f"{args.dataset}_{args.arch}/{unlearn_schema.__get_name__()}/")
    model_outpath = os.path.join(out_path, construct_name(args) + _ext_name_ + '.pth')
    save_model(model_res._model, model_outpath)
    logger.info(f" +++++++++++ Model saved at: {model_outpath} +++++++++++ ")
    
    ## EVALUATION: using the test data to evaluate the model
    if args.forget_classes is None:
        logger.info(" +++++++++++ Unlearned Model Test +++++++++++ ")
        train_acc, train_auc = BasicUnlearnSchema._test_model_(model_res, data['train'], args.batch_size, args.lossfn, device)
        train_retain_acc, train_retain_auc = BasicUnlearnSchema._test_model_(model_res, data['retain'], args.batch_size, args.lossfn, device)
        valid_acc, valid_auc = BasicUnlearnSchema._test_model_(model_res, data['valid'], args.batch_size, args.lossfn, device)
        test_acc, test_auc = BasicUnlearnSchema._test_model_(model_res, data['test'], args.batch_size, args.lossfn, device)
        forget_acc, forget_auc = BasicUnlearnSchema._test_model_(model_res, data['forget'], args.batch_size, args.lossfn, device)
        retain_acc, retain_auc = BasicUnlearnSchema._test_model_(model_res, data_retain, args.batch_size, args.lossfn, device)
        
        logger.info(" ###### [Train, Train-retain, Valid, Test, Forget, Remain-Data] "
                + f"  Acc: {train_acc:.4f}, {train_retain_acc:.4f}, {valid_acc:.4f}, {test_acc:.4f}, {forget_acc:.4f}, {retain_acc:.4f} ######")
                # + f"| Auc: {train_auc:.4f}, {train_retain_auc:.4f}, {valid_auc:.4f}, {test_auc:.4f}, {forget_auc:.4f}, {retain_auc:.4f} ###### ")
    else:
        logger.info(f" +++++++++++ Forget classes: {args.forget_classes} +++++++++++ ")
        train_retain_acc, _ = BasicUnlearnSchema._test_model_(model_res, data['retain'], args.batch_size, args.lossfn, device)
        train_forget_acc, _ = BasicUnlearnSchema._test_model_(model_res, data['forget'], args.batch_size, args.lossfn, device)
        test_retain_acc, _ = BasicUnlearnSchema._test_model_(model_res, data['test'], args.batch_size, args.lossfn, device)
        test_forget_acc, _ = BasicUnlearnSchema._test_model_(model_res, data_unlearn['test'], args.batch_size, args.lossfn, device)
        valid_retain_acc, _ = BasicUnlearnSchema._test_model_(model_res, data['valid'], args.batch_size, args.lossfn, device)
        valid_forget_acc, _ = BasicUnlearnSchema._test_model_(model_res, data_unlearn['valid'], args.batch_size, args.lossfn, device)
        
        logger.info(" ###### Remain [Train, Valid, Test]:"
                            + f"  Acc: {train_retain_acc:.4f}, {valid_retain_acc:.4f}, {test_retain_acc:.4f} ######")
        logger.info(" ###### Forget [Train, Valid, Test]:"
                            + f"  Acc: {train_forget_acc:.4f}, {valid_forget_acc:.4f}, {test_forget_acc:.4f} ######")
        
    logger.info("********************************* \n\n")
    print("Done!")
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
    parser = argparse.ArgumentParser(description='Robust Unlearn Model')
    parser.add_argument('--logname', default='trail_contrast', help='name of output log file')
    parser.add_argument('--log_dir', default='./logs', help='Folder where all logs are stored (default: .)')
    parser.add_argument('--out_dir', default='./outs', help='Folder where all outputs are stored (default: .)') 
    parser.add_argument('--exp_name', default='cont_logs', help='Subfolder where all logs are stored (default: logs)')
    parser.add_argument('--verbose', type=argparse2bool, default=True, help='whether to print the log')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=2, help='cuda device (default: 0)')
    
    ## dataset and model related parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100', 'stl10', 'imagenet', 'vggface2', 'svhn', 'fashionmnist'],
                        help='dataset for training (default: cifar10)')
    parser.add_argument('--select_classes', type=list_of_ints, default='3,8', help='selected classes for small dataset (default: 3,8)')
    parser.add_argument('--num_samples', type=int, default=-1, help='number of samples per class selected for small dataset')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers for data loading (default: 4)')
    parser.add_argument('--valid_size', type=int, default=5000, metavar='V', help='number of validation samples (default: 5000)')
    parser.add_argument('--top_k', type=int, default=-1, metavar='K', help='number of selected informative samples (default: 5000)')
    parser.add_argument('--num_to_forget', type=int, default=5000, metavar='Nf', help='number of samples to forget (default: 5000)')
    parser.add_argument('--forget_classes', type=list_of_ints, default=None, metavar='Cf', help='classes to forget (default: None)')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='sample ratio for the retain dataset (default: 0.5)')
    
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['mlps', 'logistic', 'simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'densenet', 'vit'],
                        help='model architecture (default: resnet18)')
    parser.add_argument('--model_path', type=str, default="", help='path of the model to unlearn')
    parser.add_argument('--dynamic_weight', type=int, default=0, choices=[-1, 0, 1]) #type=argparse2bool, default=False, help='dynamic_weight for feedback loss (default: False)')
    parser.add_argument('--queue_size', type=int, default=1024, help='size of the queue for the feedback loss (default: 1024)')
    
    # ##TODO: temporal parameters, will be removed later
    # parser.add_argument('--model_num', type=int, default=1, help="Number of original model")
    # parser.add_argument('--unlearn_num', type=int, default=1, help="Number of unlearned model")
    # parser.add_argument('--unlearn_size', type=int, default=1000, help="Size of the sample to be unlearned")
    
    ## unlearning related parameters
    parser.add_argument('--epochs', type=int, default=10, metavar='E', help='number of epochs for unlearning (default: 20)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='B', help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate (default: 0.001)')
    # parser.add_argument('--lr_feedback', type=float, default=0.001, help='initial learning rate for feedback (default: 0.001)')
    # parser.add_argument('--lam_reg', type=float, default=10.0, help='regularization parameter for adversarial part (default: 0.1)')    
    parser.add_argument('--strength', type=float, default=1.0, help='strength of the feedback (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay (default: 0.0005)')
    parser.add_argument('--class_wise', type=argparse2bool, default=True, help='whether to use class-wise feedback')
    # parser.add_argument('--generalize', type=argparse2bool, default=True, help='whether to contain generalization loss')
    parser.add_argument('--save_checkpoint', type=argparse2bool, default=False, help='whether to save the checkpoint')
    
    ## running parameter
    parser.add_argument('--protocal', type=str, default='moment', #'running',
                        choices=['running', 'moment'])  # ['FT', 'LASTK', 'RL', 'GA', 'DISTILL']
    parser.add_argument('--lossfn', type=str, default='ce', choices=['ce', 'bce', 'mse'])
    parser.add_argument('--optim', type=str, default="adam", choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping (default: 10)')
    parser.add_argument('--scheduler', default='CosineAnnealingWarmRestarts', #'CosineAnnealingLR', #'None', # 'LRScheduler', #
                        choices = ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR', 'LRScheduler', 'None'],
                        help='Pytorch Scheduler name: (default: The one used for train')

    ## protocal parameters
    parser.add_argument('--last_k', type=int, default=3, 
                        help="Number of last layers to be used for retraining for unlearning (default: 1)")
    parser.add_argument('--re_init', type=argparse2bool, default=True, 
                        help="Whether to re-initialize the weight of last layers for LastKlayer unlearning")   
    
    ## parameters for the different protocals
    # parser.add_argument('--finetune_over_retain', type=argparse2bool, default=True, 
    #                     help="Whether to finetune the model for retain data for the GA and RL protocal")
    # parser.add_argument('--correct_senstive', type=argparse2bool, default=True, 
    #                     help="Whether to correct the prediction over retain data for the DISTILL protocal")
    
    ### data augmentation related parameters
    parser.add_argument("--preproc_train_transform", type=str, default="normal",
                        choices=["none", "normal", "random", "randerase", "augafn", 
                                 "baseaug", "cutout", "cutmix", "autoaug", "augmix", "randaug", "test"], 
                        help="data augmentation for preprocessing the training data",)
    parser.add_argument("--preproc_test_transform", type=str, default="test",
                        choices=["none", "normal", "random", "randerase", "augafn", 
                                 "baseaug", "cutout", "cutmix", "autoaug", "augmix", "randaug", "test" ], 
                        help="data augmentation for preprocessing the testing data",)
    
    parser.add_argument('--augment_retain', type=argparse2bool, default=False, help='whether to augment the retain data')
    parser.add_argument('--weight_forget', type=float, default=1.0, help='weight for the forget data (default: 1.0)')
    parser.add_argument('--weight_retain', type=float, default=1.0, help='weight for the retain data (default: 1.0)')
    parser.add_argument('--projector_dimension', type=int, default=256, help='dimension of the projector (default: 256)')
    parser.add_argument('--outs_dimension', type=int, default=128, help='dimension of the output (default: 128)')
    
    
    args = parser.parse_args()
    return args


def data_model_test(model, data, batch_size, num_workers, device, transform=None):
    from learner import complete_test
    test_loader = DataLoader(data, batch_size, False, num_workers=num_workers)
    acc, auc = complete_test(model, test_loader, device, transform)
    return acc, auc


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    args.unlearn_method = 'contun'
    ## set output path
    outs = SingletonString()
    outs.content = args.out_dir
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # mod_fn = 'cifar10_default_resnet18_seed-1_Nf-0_split-0.2_lr-[0_001-0_001]_wd-5e-05_sched-None.pt'
    # args.model_path = './output/outs_10000_rob/cifar10/model_bases/ORG0/' + mod_fn

    # args.model_path = "./outs/cifar10_-1/model_bases/cifar10_resnet18/ORG/" + \
    #     "cifar10_resnet18_seed-3407_Nf-5000_ep-20_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-testregular_0.0_sched-None.pth"

    # args.model_path = "./outs/cifar10_-1_difval/model_bases/cifar10_resnet18/ORG/" + \
    #     "cifar10_resnet18_seed-0_Nf-1000_ep-25_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_1e-07_sched-None.pth"

    args.model_path = "./outs/cifar10_-1/model_bases/cifar10_resnet18/ORG/" + \
            "cifar10_resnet18_seed-1_Nf-0_ep-20_bs-256_lr-[0_0001-0_0002]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"
    
    from data_tool import DataLoaderTool, DataStore, construct_data
    DataStore.create_basic_folders()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    logpath = os.path.join(args.log_dir, args.exp_name)
    logger = create_logger(logpath, 'main_contun_' + args.logname)
    logger.info(args)

    logger.info("load dataset and transform")
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    if args.arch in ['resnet50', 'densenet', 'vit']:
        dt_size = 224 
        feature_dims = (3, dt_size, dt_size)
    processor = Transforms(dt_mean, dt_std, dt_size)
    train_transform = processor.get_transform(args.preproc_train_transform, num_classes)
    test_transform = processor.get_transform(args.preproc_test_transform, num_classes)
    transform = None

    num_samples = args.num_samples
    dataset_conf = { "num_samples": num_samples, "select_classes": args.select_classes}
    train_data, test_data = DataLoaderTool.load_dataset(args.dataset, train_transform, test_transform, **dataset_conf)
    num_train = len(train_data)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    data, data_unlearn, data_retain = construct_data(train_data, test_data, train_size=args.num_samples, valid_size=args.valid_size, 
                                                     forget_size=args.num_to_forget, forget_classes=args.forget_classes)

    ##  import after setting the out path
    from data_tool import flatten_subset
    from tool import PathGenerator as pather
    from contrast_model import ContrastMomentUnlearn, ContrastRunningUnlearn, ContrastModelWrapper

    ## load original model and test
    logger.info(f"Load original model from @ {args.model_path}")
    org_mod = load_model(args.model_path).to(device)
    
    proj_dims = args.projector_dimension #args._get_args('projector_dimension', 256)
    outs_dim = args.outs_dimension #args._get_args('outs_dimension', 128)
    
    logger.info("Init contrastive model")
    base_model = ContrastModelWrapper(org_mod, proj_dims, outs_dim)
    
    ## apply different unlearn models
    logger.info("Init unlearn model")
    unlearner = None
    if args.protocal == 'running':
        unlearner = ContrastRunningUnlearn
    elif args.protocal == 'moment':
        unlearner = ContrastMomentUnlearn
    else:
        raise ValueError("Invalid protocal")

    ## unlearn model initialization
    contun = unlearner(logpath, args.logname, args.out_dir, out_name='model_contun')
    contun.logger.info(args)
    # logger.info(f" +++++++  Selected top-K informative samples: {args.top_k}")
    
    # model setting
    logger.info("Set parameters and transform configurations")
    trans_params = '_'.join([args.preproc_train_transform, args.preproc_test_transform])
    contun.set_params(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, 
        weight_decay=args.weight_decay, momentum=args.momentum, save_checkpoints=args.save_checkpoint, 
        class_wise=args.class_wise, seed=args.seed, dynamic_weights=args.dynamic_weight
    )

    logger.info("EVALUATION: Original model over original dataset")
    train_acc, train_auc = data_model_test(org_mod, data['train'], args.batch_size, args.num_workers, device, transform)
    valid_acc, valid_auc = data_model_test(org_mod, data['valid'], args.batch_size, args.num_workers, device, transform)
    test_acc, test_auc = data_model_test(org_mod, data['test'], args.batch_size, args.num_workers, device, transform)
    forget_acc, forget_auc = data_model_test(org_mod, data['forget'], args.batch_size, args.num_workers, device, transform)
    retain_acc, retain_auc = data_model_test(org_mod, data_retain, args.batch_size, args.num_workers, device, transform)
    logger.info(" ###### [Train, Valid, Test, Forget, Remain-Data] "
                        + f"  Acc: {train_acc:.4f}, {valid_acc:.4f}, {test_acc:.4f}, {forget_acc:.4f}, {retain_acc:.4f} ######")
    #                   + f"| Auc: {train_auc:.4f}, {valid_auc:.4f}, {test_auc:.4f}, {forget_auc:.4f}, {retain_auc:.4f} ###### ")
    if args.forget_classes is not None:
        logger.info(f" +++++++++++ Forget Class: {args.forget_classes} +++++++++++ ")
        for type_, dt_ in data_unlearn.items():
            acc_dt, auc_dt = data_model_test(org_mod, dt_, args.batch_size, args.num_workers, device, transform)
            logger.info(f"##### [ {type_:6} ]: ACC: {acc_dt:.4f}, AUC: {auc_dt:.4f}")
        #                   + f"| Auc: {forget_auc:.4f}, {retain_auc:.4f} ###### ")

    if args.sample_ratio > 0 and args.sample_ratio < 1:
        from torch.utils.data import Subset
        num_sams = int(len(data['retain']) * args.sample_ratio)
        data['retain'] = flatten_subset(Subset(data['retain'], np.random.choice(len(data['retain']), num_sams, replace=False)))
        logger.info(f" +++++++++++ Remain data reduced to {num_sams} +++++++++++ ")
    
    ## set dataset for unlearning
    logger.info("Set data for unlearning")
    contun.set_data(data, num_classes, args.batch_size, args.num_workers)
    ## evaluation the original model
    # robunlearn.evaluation(org_mod, args.lossfn, device, 'none')
    acc_org, auc_org = contun.model_evaluate(org_mod, test_loader, device)
    
    # default_wt_forget = num_train / args.num_to_forget - 1.0
    # weight_forget = default_wt_forget
    weight_forget = args.weight_forget
    weight_retain = args.weight_retain
    
    ## perform forget-sample unlearn
    logger.info(" +++++++++++++++ Start Unlearning +++++++++++++++")
    ckpt_path = pather.get_checkpoint_path(f"{args.dataset}_{args.num_samples}_{args.arch}")
    unlearn_mod, ckpt_ = contun.unlearn(base_model, protocal='FT', loss=args.lossfn, patience=args.patience,
                                scheduler_option=args.scheduler, optimization=args.optim, augument_retain=args.augment_retain,
                                device=device, checkpoint_path=ckpt_path, model_type=args.arch, queue_size=args.queue_size,
                                weight_forget=weight_forget, weight_retain=weight_retain, output_dim=outs_dim)

    outfn = f"{args.arch}_{args.protocal}_{trans_params}_wtr-{weight_retain}_wtf-{weight_forget}_dywt-{args.dynamic_weight}_augr_{args.augment_retain}_{contun._suffix_}_best.pt"
    out_path = os.path.join(contun.path['model'], outfn)
    save_model(unlearn_mod, out_path)
    logger.info(f" ++++++++++++++++++ Model saved @ {out_path} +++++++++++++++++++")    

    logger.info("Evaluation: Unlearned model test")
    for type_ in ['train', 'retain', 'valid', 'test', 'forget']: #data.items():
        acc_dt, auc_dt = data_model_test(unlearn_mod, data[type_], args.batch_size, args.num_workers, device, transform)
        logger.info(f"##### [ {type_:6} ]: ACC: {acc_dt:.4f}")   # , AUC: {auc_dt:.4f}

    ## evaluation the unlearned model
    # robun.evaluation(unlearn_mod, args.lossfn, device, "none")
    acc_unlearn, auc_unlearn = contun.model_evaluate(unlearn_mod, test_loader, device)
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







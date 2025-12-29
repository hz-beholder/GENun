import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

# 保持你原有的引用不变
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
    
    # 随机遗忘时的数量，如果是类别遗忘，代码中会将其覆盖为 -1 (全部)
    parser.add_argument('--num_to_forget', type=int, default=5000, metavar='Nf', help='number of samples to forget (default: 5000)')
    
    # 核心控制参数：为空则为随机遗忘，为"5"则为类别遗忘
    parser.add_argument('--forget_classes', default=None, metavar='Cf', help='classes to forget (default: None)')
    
    # ================= [新增参数] =================
    # True: 冻结骨干+Mask最后一层 (默认) | False: 更新所有参数 (类似随机遗忘的更新方式)
    parser.add_argument('--mask_updates', type=argparse2bool, default=True, 
                        help='If True, freeze backbone and mask gradient for other classes. If False, update all parameters.')
    # ============================================

    parser.add_argument('--sample_ratio', type=float, default=1.0, help='sample ratio for the retain dataset (default: 0.5)')
    
    parser.add_argument('--arch', type=str, default='resnet18', help='model architecture (default: resnet18)',
                        choices=['mlps', 'logistic', 'simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'densenet', 'vit', 'vgg','ALLCNN'])
    parser.add_argument('--model_path', type=str, default="", help='path of the model to unlearn')
    
    ## unlearning related parameters
    parser.add_argument('--epochs', type=int, default=10, metavar='E', help='number of epochs for unlearning (default: 20)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='B', help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (default: 0.0005)')
    parser.add_argument('--class_wise', type=argparse2bool, default=True, help='whether to use class-wise feedback')
    parser.add_argument('--save_checkpoint', type=argparse2bool, default=False, help='whether to save the checkpoint')
    
    parser.add_argument('--over_forget', type=argparse2bool, default=True, help='whether using the forget-set for finetuning (weighted GA)')
    parser.add_argument('--regularizer', type=str, default='l1+l2', choices=['none', 'l1', 'l2', 'l1+l2', 'l1_diff', 'l2_diff'], help='weights regularization type')
    parser.add_argument('--gamma', type=float, default=5e-5, help='regularization parameter for weight regularization (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.97, help='trade-off between loss of retain and forget data (default: 0.5)')
    parser.add_argument('--no_reg_epochs', type=int, default=0, help='number of epochs without regularization (default: 0)')
    parser.add_argument('--dynamic_weight', type=argparse2bool, default=False, help='dynamic_weight for feedback loss (default: False)')

    # ## running parameter
    parser.add_argument('--protocal', type=str, default='GenUn', choices=['DU', 'LASTK', 'DISTILL', 'GenUn'])
    parser.add_argument('--lossfn', type=str, default='ce', choices=['ce', 'bce', 'mse'])
    parser.add_argument('--optim', type=str, default="adam", choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping (default: 10)')
    parser.add_argument('--scheduler', default='CosineAnnealingWarmRestarts', 
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
    parser.add_argument("--online_forget_aug", type=str, default="none", 
                        help="data augmentation for forgetting",
                        choices=["none", "normal", "random", "randerase", "augafn", "baseaug", 
                                 "cutout", "mixup", "cutmix", "autoaug", "augmix", "randaug", "test" ],)
    
    args = parser.parse_args()
    return args

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
            + f" Acc: {train_acc:.4f}, {train_retain_acc:.4f}, {valid_acc:.4f}, {test_acc:.4f}, {forget_acc:.4f}, {retain_acc:.4f} ######"
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
    if args.forget_classes is not None:
        try:
            # 移除空格并按逗号分割
            class_str_list = str(args.forget_classes).split(',')
            # 转换为整数列表
            args.forget_classes = [int(c.strip()) for c in class_str_list if c.strip()]
        except ValueError:
            print(f"Error: --forget_classes parameter '{args.forget_classes}' is invalid. Please use integers separated by commas (e.g., '5' or '3,8').")
            exit(1)
    args.unlearn_method = 'genun'
    
    ## set output path
    outs = SingletonString()
    outs.content = args.out_dir
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    from data_tool import DataLoaderTool, DataStore, construct_data
    from data_tool import flatten_subset
    from tool import PathGenerator
    from models_genun import GeneModUnlearn

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
    if args.arch in ['densenet', 'vit']: 
        dt_size = 224
        feature_dims = (3, dt_size, dt_size)
    tran_process = Transforms(dt_mean, dt_std, dt_size)
    
    ## pre-processing transform setup
    train_trans = tran_process.get_transform(args.preproc_train_transform, num_classes)
    test_trans = tran_process.get_transform(args.preproc_test_transform, num_classes)
    train_data, test_data = DataLoaderTool.load_dataset(args.dataset, train_trans, test_trans, **dataset_conf)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ================= [修改点 1] 处理类别遗忘的数据集构建 =================
    # 如果指定了 forget_classes，将 num_to_forget 设为 -1 (代表所有)，确保包含该类的所有样本
    current_forget_size = args.num_to_forget

    data, data_unlearn, data_retain = construct_data(train_data, test_data, train_size=args.num_samples, 
                valid_size=args.valid_size, forget_size=current_forget_size, forget_classes=args.forget_classes)
    # ====================================================================
    
    # 评估纯预训练模型
    logger.info(f" +++++++++++ Evaluation: Generic Pre-trained Model ({args.arch}) +++++++++++ ")
    base_model = get_model(args.arch, feature_dims, num_classes, pretrained=True).to(device)
    model_testing(base_model, data, data_retain, data_unlearn, args.batch_size, args.num_workers, 
                  device, transform, None, logger)
    del base_model
    torch.cuda.empty_cache()
    logger.info(" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n")

    ## load original model and test
    logger.info(f"Load original model from @ {args.model_path}")
    model_ = load_model(args.model_path).to(device)

    ## pre-evaluation of the original model
    logger.info(" +++++++++++ Evaluation: Original model  +++++++++++ ")
    model_testing(model_, data, data_retain, None, args.batch_size, args.num_workers, device, transform, None, logger)
    
    ## apply different unlearn models
    logger.info("Initialize unlearn model")
    mu_er = GeneModUnlearn(logpath, args.logname, args.out_dir, name=f'model_{args.protocal}')
    
    # model setting
    logger.info("Set parameters and basic configurations")
    
    lr_list = [ args.lr ] 
    for lr in lr_list:
        args.lr = lr
        logger.info(f"==== Running GenUn with lr={lr} ====")
        mu_er.set_params(**vars(args))
        mu_er.config_transform(dt_mean, dt_std, dt_size)
        
        if args.sample_ratio > 0 and args.sample_ratio < 1:
            from torch.utils.data import Subset
            num_sams = int(len(data['retain']) * args.sample_ratio)
            data['retain'] = flatten_subset(Subset(data['retain'], np.random.choice(len(data['retain']), num_sams, replace=False)))
            logger.info(f" +++++++++++ Remain data reduced to {num_sams} +++++++++++ ")

        ## set dataset for unlearning
        logger.info("Set unlearning data")
        mu_er.set_data(data, num_classes, args.batch_size, args.num_workers)
        
        acc_org, auc_org = mu_er.model_evaluate(model_, test_loader, device)

        # ================= [修改点 2] 类别遗忘的参数冻结与梯度掩码 =================
        # 逻辑：
        # 1. 必须是类别遗忘模式 (forget_classes is not None)
        # 2. 且 mask_updates 为 True (默认)
        
        grad_hook_handle_w = None
        grad_hook_handle_b = None
        
        if args.forget_classes is not None:
            # ==== 分支：类别遗忘 ====
            if args.mask_updates:
                # [模式 A]: 冻结骨干 + Mask分类层 (Partial Update)
                try:
                    target_cls_idx = args.forget_classes[0]
                    logger.info(f"Configuring Gradient Mask for Class-wise Unlearning (Class {target_cls_idx})...")

                    last_layer_name = 'fc' 
                    if args.arch.startswith('vgg') or 'densenet' in args.arch:
                        last_layer_name = 'classifier'
                    elif 'vit' in args.arch:
                        last_layer_name = 'head'
                    
                    last_module = None
                    for n, m in model_.named_modules():
                        if n == last_layer_name:
                            last_module = m
                            break
                    
                    if last_module is None:
                        if hasattr(model_, 'fc'): last_module, last_layer_name = model_.fc, 'fc'
                        elif hasattr(model_, 'classifier'): last_module, last_layer_name = model_.classifier, 'classifier'
                    
                    if last_module is None:
                        raise ValueError(f"Could not find last layer '{last_layer_name}' in model {args.arch}")

                    # 1. 冻结骨干
                    for name, param in model_.named_parameters():
                        if last_layer_name not in name:
                            param.requires_grad = False
                    logger.info(f"Backbone frozen. Only '{last_layer_name}' is trainable.")

                    # 2. 注册 Hook (Mask)
                    weight_mask = torch.zeros_like(last_module.weight)
                    weight_mask[target_cls_idx, :] = 1.0
                    
                    def get_mask_hook(mask):
                        def hook(grad):
                            return grad * mask.to(grad.device)
                        return hook

                    grad_hook_handle_w = last_module.weight.register_hook(get_mask_hook(weight_mask))
                    
                    if last_module.bias is not None:
                        bias_mask = torch.zeros_like(last_module.bias)
                        bias_mask[target_cls_idx] = 1.0
                        grad_hook_handle_b = last_module.bias.register_hook(get_mask_hook(bias_mask))
                    
                    logger.info(f"Gradient mask applied. Only weights for Class {target_cls_idx} will be updated.")

                except Exception as e:
                    logger.error(f"Error setting up class-wise unlearning: {e}")
                    exit(1)
            else:
                # [模式 B]: 类别遗忘，但更新所有参数 (Full Update)
                logger.info(f"Class-wise Unlearning (Class {args.forget_classes[0]}), but updating ALL parameters (End-to-End).")
        
        else:
            # ==== 分支：随机遗忘 (Random Unlearning) ====
            logger.info("Standard Random Unlearning Mode (All layers trainable).")
        # =========================================================================

        ## perform forget-sample unlearn
        logger.info(" +++++++++++++++ Start Unlearning (in process) +++++++++++++++")
        ckpt_path = PathGenerator.get_checkpoint_path(f"{args.dataset}_{args.num_samples}_{args.arch}")
        
        unlearn_mod, ckpt = mu_er.unlearn(
            model_, None, None, args.patience, args.scheduler, args.optim,
            args.online_train_aug, args.online_forget_aug, device, ckpt_path, 
            model_type=args.arch, over_forget=args.over_forget, suffix=trans_params,
        )
        adv_losser = 'none'
        feed_losser = 'none'
        modfn = f"{args.arch}_{args.protocal}_lr-{lr}_{trans_params}_adv-{adv_losser}_fdbk-{feed_losser}_overforget-{args.over_forget}_{mu_er._suffix_}_best.pt" 
        modout_path = os.path.join(mu_er.path['model'], modfn)
        save_model(unlearn_mod, modout_path)
        logger.info(f" ++++++++++++++++++ Model saved @ {modout_path} done! +++++++++++++++++++")    

        # 清理 hooks
        if grad_hook_handle_w is not None: grad_hook_handle_w.remove()
        if grad_hook_handle_b is not None: grad_hook_handle_b.remove()

        ## evaluation the unlearned model
        from learner import complete_test
        logger.info("\n" + "++++++++"*5)
        logger.info(" +++ Evaluation: Unlearned model test +++")
        for sub_dat in ['train', 'retain', 'valid', 'test', 'forget']: 
            if sub_dat in data:
                t_loader = DataLoader(data[sub_dat], args.batch_size, False, num_workers=args.num_workers)
                acc_dt, auc_dt = complete_test(unlearn_mod, t_loader, device, transform)
                logger.info(f"##### [ {sub_dat:6} ]: ACC: {acc_dt:.4f}")

        acc_unlearn, auc_unlearn = mu_er.model_evaluate(unlearn_mod, test_loader, device)
        logger.info(f"Accuracy: Original: {acc_org:.4f} --> Unlearned: {acc_unlearn:.4f}")

        logger.info(" +++++++++++ Unlearned Model Test +++++++++++ ")
        model_testing(unlearn_mod, data, data_retain, data_unlearn, args.batch_size, args.num_workers, 
                      device, transform, args.forget_classes, logger)
        logger.info("********************************* \n\n")
    print("Done!")
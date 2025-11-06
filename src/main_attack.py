import os
import argparse
import random
from typing import Iterable
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Subset

from logger import create_logger
from transforms import Transforms
from model_shallow import get_model as get_attack_model
from model_deep import get_model as get_source_model
from extractor import get_extractor
from attacker import AttackSource, Attackers
from const import SingletonString, CONSTRUCTIONSELF
from utils import argparse2bool, list_of_ints, load_model, seed_everything

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
float_formatter = "{:.4f}".format


def parse_args():
    parser = argparse.ArgumentParser(description='Membership Inference Attack For Unlearning')
    ######################### general parameters ################################
    parser.add_argument('--logname', default='ATK', help='name of output log file')
    parser.add_argument('--log_dir', default='.', help='Folder where all logs are stored (default: .)')
    parser.add_argument('--out_dir', default='./outs/svhn_-1/', help='Folder where all outputs are stored (default: .)')
    parser.add_argument('--exp_name', default='logs', help='Subfolder where all logs are stored (default: logs)')
    parser.add_argument('--verbose', type=argparse2bool, default=True, help='whether to print the log')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--cuda', type=str, default='6', help="Choose the GPU device")

    parser.add_argument('--dataset', type=str, default='cifar10', #'svhn',  #
                        choices=['mnist', 'cifar10', 'cifar100', 'stl10', 'imagenet', 'vggface2', 'svhn', 'fashionmnist',
                                #'small_mnist', 'small_mnist_2', 'mnist', 'small_cifar2', 'small_cifar5', 'small_cifar10', 
                                # 'tiny_imagenet5', 'tiny_imagenet_pretrain', 'tiny_image_finetune', 'adult', 'accident', 'location', 
                                ])
    parser.add_argument('--select_classes', type=list_of_ints, default='3,8', help='selected classes for small dataset (default: 3,8)')
    parser.add_argument('--num_samples', type=int, default=-1, help='number of samples per class selected for small dataset')
    parser.add_argument('--valid_size', type=int, default=7000, metavar='V', help='number of validation samples (default: 5000)')
    parser.add_argument('--num_to_forget', type=int, default=7000, metavar='Nf', help='number of samples to forget (default: 5000)')
    parser.add_argument('--forget_classes', type=list_of_ints, default=None, metavar='Cf', help='classes to forget (default: None)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers for data loading (default: 4)')

    ### data augmentation related parameters
    parser.add_argument("--preproc_train_transform", type=str, default="normal", help="data augmentation for preprocessing the training data",
                        choices=["normal", "mixup", "random", "augafn", "baseaug", "cutout", "cutmix", "autoaug", "augmix", "randaug", "test"],)
    parser.add_argument("--preproc_test_transform", type=str, default="normal", help="data augmentation for preprocessing the testing data",
                        choices=["normal", "mixup", "random", "augafn", "baseaug", "cutout", "cutmix", "autoaug", "augmix", "randaug", "test" ],)
    
    parser.add_argument('--arch', type=str, default='simple_cnn', #'resnet18', 
                            choices=['mlps', 'logistic', 'simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'densenet', 'vit'],
                            help="model architecture (default: resnet18)")
    parser.add_argument('--model_path', type=str, default="", help='path of the model to unlearn')
    parser.add_argument('--lossfn', type=str, default='ce', choices=['ce', 'bce', 'mse'])
    parser.add_argument('--unlearn_method', type=str, default='none', help="Name of the unlearn method")
    
    ######################### attack related parameters ################################
    parser.add_argument('--attack_model', type=str, default='lr', 
                       choices=['dt', 'mlp', 'lr', 'rf', 'svm', 'lira'],  # 添加lira
                       help="Attack model")
    
    # 添加lira特有的参数
    parser.add_argument('--lira_shadow_models', type=int, default=5, 
                       help="Number of shadow models for LiRA attack")
    parser.add_argument('--lira_augmentations', type=int, default=10, 
                       help="Number of augmentations per sample for LiRA")
    parser.add_argument('--lira_shift', type=int, default=4, 
                       help="Shift parameter for LiRA augmentations")
    
    parser.add_argument('--test_ratio', type=float, default=0.5, metavar='T_S', help='train-valid split ratio for attacker model training (default: 0.5)')
    # parser.add_argument('--extract_feature', type=argparse2bool, default=True, help='whether to extract feature online')
    parser.add_argument('--hidden_layer_sizes', type=list_of_ints, default='20', help="Hidden layer sizes for MLP")
    parser.add_argument('--attack_feature', type=str, default='loss', #'linear',  #'loss', #'gradient', #
                        choices=['entropy', 'loss', 'posterior', 'linear', 'nonlinear', 'mixlayer', 'gradient', 'gradientnorm','lira'], 
                        help="Type of sample features for membership inference attack")
    parser.add_argument('--last_k', type=int, default=1, help="Number of last layers to be used for attack")
    
    ## how does the attack model use the features
    parser.add_argument('--stacked', type=argparse2bool, default=False, help="Whether to use stacked features for attack")
    parser.add_argument('--include_posterior', type=argparse2bool, default=True, help="Whether to include posterior for attack")
    

    # ######################### defense related parameters ################################
    parser.add_argument('--is_dp_defence', type=argparse2bool, default=False)
    parser.add_argument('--top_k', type=int, default=0, choices=[0, 1, 2, 3, 4],  help=" 0 (label), 4 (no defense)")
    ###########################################################
    # parser.add_argument('--num_attacker_sample', type=int, default=1000, help="Number of samples that the attacker can access")
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training (default: 256)')
    parser.add_argument('--n_trails', type=int, default=5, help='number of trails for attack as average results')
    ######################### attack model related parameters ################################
    ### possible used parameters for deep attack model
    
    # parser.add_argument('--maxlr', type=float, default=0.001, help='the maximum learning rate (default: 0.001)')
    # parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 50)')
    # parser.add_argument('--attack_side', type=str, default='server', choices=['client', 'server'])
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    outs = SingletonString()
    outs.content = args.out_dir
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    
    # args.model_path = "./outs/cifar10_-1/model_bases/cifar10_resnet18/ORG/cifar10_resnet18_seed-3407_Nf-5000_ep-20_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-testregular_0.0_sched-None.pth"
    # args.model_path = "./outs/svhn_-1/model_bases/svhn_simple_cnn/ORG/svhn_simple_cnn_seed-0_Nf-7000_ep-40_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"    

    from data_tool import DataLoaderTool, DataStore, construct_data, flatten_subset
    logpath = os.path.join(args.log_dir, args.exp_name)
    logger = create_logger(logpath, 'main_proc_attack_' + args.logname)
    logger.info(args)
    
    # logger.info("load dataset and transform")
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    if args.arch in ['resnet50', 'densenet', 'vit']:
        dt_size = 224
        feature_dims = [3, dt_size, dt_size]
    preproc = Transforms(dt_mean, dt_std, dt_size)
    train_transform = preproc.get_transform(args.preproc_train_transform, num_classes)
    test_transform = preproc.get_transform(args.preproc_test_transform, num_classes)
    transform = None

    ## load data
    num_samples = args.num_samples
    dataset_conf = { "num_samples": num_samples, "select_classes": args.select_classes}
    train_data, test_data = DataLoaderTool.load_dataset(args.dataset, train_transform, test_transform, **dataset_conf)
    
    data, data_unlearn, data_remain = construct_data(train_data, test_data, train_size=args.num_samples, valid_size=args.valid_size, 
                                                    forget_size=args.num_to_forget, forget_classes=args.forget_classes)
    
    ## load original model and test
    logger.info(f"Load original model from @ {args.model_path}")
    # tar_model = get_source_model(args.arch, feature_dims, num_classes, pretrained=False).to(args.device)
    tar_model = load_model(args.model_path).to(device)
    
    from tool import PathGenerator as pather
    out_fn = f"attack_{args.unlearn_method}_{args.arch}_{args.attack_feature}_{args.attack_model}_{args.stacked}_{args.include_posterior}_{args.last_k}"
    attack_data_path = os.path.join(pather.get_attack_data_path(), out_fn + '.pkl')
    attack_model_path = os.path.join(pather.get_attack_model_path(), out_fn + '.pt')
    
    ## peform the attack
    ################################################################################################
    logger.info(f"Initialize the attack model {args.attack_model}")
    constructor = CONSTRUCTIONSELF
    
    ###TODO: can set different parameters for different attack models: lr, svm, mlp, rf
    ext_param = {'hidden_layer_sizes': args.hidden_layer_sizes, 'max_iter': 600, 'random_state': args.seed, 'n_jobs': 1}
    # 如果是lira攻击，添加额外参数
    if args.attack_model == 'lira':
        ext_param.update({
            'shadow_models': args.lira_shadow_models,
            'augmentations': args.lira_augmentations, 
            'shift': args.lira_shift
        })
    attack_model = get_attack_model(args.attack_model, **ext_param)
    
    logger.info(f"Initialize the feature extractor {args.attack_feature}")
    attack_source = AttackSource()
    extractor = get_extractor(args.attack_feature, batch_size=args.batch_size, lossfn=args.lossfn, 
                        last_k=args.last_k, stacked=args.stacked, include_posterior=args.include_posterior)
    attack_source.set_extractor(extractor)

    logger.info("Launching MI attack .....")
    
    if args.forget_classes is None:
        logger.info(" ***** Forget random sampled data!")
        pos_data, neg_data = data['forget'], data['test']
    else:
        logger.info(" ***** Forget specific classes data!")
        pos_data, neg_data = data['forget'], data_unlearn['test']
    
    logger.info("Setting up transforms for LiRA attack...")

    # 保存原始transform
    def get_dataset_transform(dataset):
        """安全地获取数据集的transform"""
        if hasattr(dataset, 'transform'):
            return dataset.transform
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'transform'):
            return dataset.dataset.transform
        else:
            return None

    def set_dataset_transform(dataset, transform):
        """安全地设置数据集的transform"""
        if hasattr(dataset, 'transform'):
            dataset.transform = transform
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'transform'):
            dataset.dataset.transform = transform
        # 如果都没有transform属性，我们无法设置，但至少不会报错

    original_pos_transform = get_dataset_transform(pos_data)
    original_neg_transform = get_dataset_transform(neg_data)

    # 创建一个兼容的transform，确保不包含需要标签的增强（如Mixup）
    compatible_transform = preproc.get_transform("normal", num_classes)

    # 应用兼容transform
    set_dataset_transform(pos_data, compatible_transform)
    set_dataset_transform(neg_data, compatible_transform)

    ## generate the attack data
    ## get the data for training and testing the attack models
    try:
        if not os.path.exists(attack_data_path):
            logger.info(f"#### Generate attack data and Save to {attack_data_path}")
            attack_data = attack_source.build_source_data(pos_data, neg_data, tar_model, device)
            attack_data_reform = attack_source.data_reform(attack_data, constructor, is_defence=args.is_dp_defence, top_k=args.top_k)
            attack_source.save_features(attack_data_reform, attack_data_path)
        
        logger.info(f"#### Load attack data {attack_data_path}")
        attack_data = attack_source.load_features(attack_data_path)
        
    finally:
        # 恢复原始transform
        set_dataset_transform(pos_data, original_pos_transform)
        set_dataset_transform(neg_data, original_neg_transform)
        logger.info("Restored original data transforms")

    ## construct the balanced dataset for attack
    min_size = min(len(pos_data), len(neg_data))
    pos_indices = random.sample(range(len(pos_data)), min_size)
    neg_indices = random.sample(range(len(neg_data)), min_size)
    pos_data = flatten_subset(Subset(pos_data, pos_indices))
    neg_data = flatten_subset(Subset(neg_data, neg_indices))
    
    ## generate the attack data
    ## get the data for training and testing the attack models
    if not os.path.exists(attack_data_path):
        logger.info(f"#### Generate attack data and Save to {attack_data_path}")
        attack_data = attack_source.build_source_data(pos_data, neg_data, tar_model, device)
        attack_data_reform = attack_source.data_reform(attack_data, constructor, is_defence=args.is_dp_defence, top_k=args.top_k)
        attack_source.save_features(attack_data_reform, attack_data_path)
    
    logger.info(f"#### Load attack data {attack_data_path}")
    attack_data = attack_source.load_features(attack_data_path)
    
    ## the average results for multiple runs
    train_res_avg, test_res_avg = defaultdict(list), defaultdict(list)
    
    # if args.attack_feature == 'nonlinear':
    #     args.n_trails = 1
        
    for trial in range(args.n_trails):
        attack_train, attack_test = train_test_split(attack_data, test_size=args.test_ratio, random_state=trial) #shuffle=True
        # logger.info(f"Train size: {len(attack_train)}, Test size: {len(attack_test)}")
        attaker = Attackers(logpath, args.logname + '_' + args.attack_model)
        ## based on the extracted features, train the attack models
        attaker.train(attack_model, attack_train, constructor, attack_model_path)
        _, train_res = attaker.evaluate(attack_train, constructor, "Training")
        _, test_res = attaker.evaluate(attack_test, constructor, "Testing")

        for k_ in train_res.keys():
            train_res_avg[k_].append(np.array(train_res[k_]))
            test_res_avg[k_].append(np.array(test_res[k_]))
    
    res_tr_avg, res_ts_avg = dict(), dict()
    for k_ in train_res_avg.keys():
        res_tr_avg[k_] = np.mean(train_res_avg[k_], axis=0)
        res_ts_avg[k_] = np.mean(test_res_avg[k_], axis=0)
    attack_res = [res_tr_avg, res_ts_avg]
    
    logger.info(f"Attack result (Average over {args.n_trails} trails): {attack_res}")
    logger.info(f"Attack Done!")
    
    ## reformat the output
    ################################################################################################    
    ## output attacker performance result to file
    logger.info("Performance result output ....")    
    
    mlp_hidden_layer = args.hidden_layer_sizes if args.attack_model == 'mlp' else -1
    last_k = args.last_k if args.attack_feature in ['linear', 'nonlinear', 'mixlayer', 'gradient'] else 0
    out_content = f"{args.attack_model}, {args.unlearn_method}, {args.attack_feature}, " + \
                    f"{args.include_posterior}, {args.stacked}, {args.last_k}, {mlp_hidden_layer}, "
    
    attack_res_path = os.path.join(pather.get_attack_result_path(), f"attack_server_{args.unlearn_method}.csv")
    if not os.path.exists(attack_res_path):
        with open(attack_res_path, 'w') as fp:
            fp.writelines("Method, Unlearn, Feature, Posterior, stack, last_K, mlp_layer, construct, auc_train, acc_train, auc_test, acc_test\n")
            fp.close()
    
    with open(attack_res_path, 'a') as fp:
        for apoc_ in constructor:
            key = apoc_.name
            temp = list()
            for term in attack_res:
                if isinstance(term[key], Iterable):
                    temp.extend(term[key])
                else:
                    temp.append(term[key])
            temp = np.array(temp).flatten()
            # acc_test 即为 ASR（Attack Success Rate）
            fp.writelines(out_content + f"{key}, {', '.join([float_formatter(x) for x in temp])}\n")
        fp.close()
    
    
    logger.info("Done!")

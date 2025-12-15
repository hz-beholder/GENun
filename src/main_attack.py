import os
import argparse
import random
from typing import Iterable
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Subset
# 新增
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
import hashlib
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
    parser.add_argument('--lira_shadow_models', type=int, default=8, 
                       help="Number of shadow models for LiRA attack")
    parser.add_argument('--lira_augmentations', type=int, default=8, 
                       help="Number of augmentations per sample for LiRA")
    parser.add_argument('--lira_shift', type=int, default=4, 
                       help="Shift parameter for LiRA augmentations")
    # 可控 shadow 训练超参
    parser.add_argument('--lira_epochs', type=int, default=20, help='epochs for shadow models')
    parser.add_argument('--lira_lr', type=float, default=1e-3, help='lr for shadow models')
    parser.add_argument('--lira_weight_decay', type=float, default=5e-5, help='weight decay for shadow models')
    parser.add_argument('--lira_shadow_train_size', type=int, default=50000, help='bootstrap train size for each shadow')
    parser.add_argument('--lira_inject_repeats', type=int, default=64, help='how many repeats to inject target sample into IN-shadow training')
    parser.add_argument('--lira_use_remain_pool', type=argparse2bool, default=True, help='prefer data_remain as base pool')
    parser.add_argument('--test_ratio', type=float, default=0.5, metavar='T_S', help='train-valid split ratio for attacker model training (default: 0.5)')
    # parser.add_argument('--extract_feature', type=argparse2bool, default=True, help='whether to extract feature online')
    parser.add_argument('--hidden_layer_sizes', type=list_of_ints, default='20', help="Hidden layer sizes for MLP")
    parser.add_argument('--attack_feature', type=str, default='loss', #'linear',  #'loss', #'gradient', #
                        choices=['entropy', 'loss', 'posterior', 'linear', 'nonlinear', 'mixlayer', 'gradient', 'gradientnorm','lira',], 
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

# ============ LiRA 工具：张量增强与数据集封装 ============

def _tensor_random_augment(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return x
    
    # 原有的增强逻辑
    if shift and shift > 0:
        dx = random.randint(-shift, shift)
        dy = random.randint(-shift, shift)
        if dx != 0 or dy != 0:
            x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
    if random.random() < 0.5:
        x = torch.flip(x, dims=(2,))  # 水平翻转
    return x
    if shift and shift > 0:
        dx = random.randint(-shift, shift)
        dy = random.randint(-shift, shift)
        if dx != 0 or dy != 0:
            x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
    if random.random() < 0.5:
        x = torch.flip(x, dims=(2,))  # 水平翻转
    return x

class TensorAugmentDataset(Dataset):
    def __init__(self, base_ds: Dataset, shift: int):
        self.base = base_ds
        self.shift = shift
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = _tensor_random_augment(x, self.shift)
        return x, y

class SingleSampleTensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: int, repeats: int, shift: int):
        self.x = x.detach().clone()
        self.y = int(y)
        self.repeats = int(repeats)
        self.shift = int(shift)
    def __len__(self):
        return self.repeats
    def __getitem__(self, idx):
        x = _tensor_random_augment(self.x, self.shift)
        return x, self.y

class AugmentN(Dataset):
    # 用于对单一样本产生 N 次增强，评估目标/阴影损失
    def __init__(self, x: torch.Tensor, y: int, n: int, shift: int):
        self.x = x.detach().clone()
        self.y = int(y)
        self.n = int(n)
        self.shift = int(shift)
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        x = _tensor_random_augment(self.x, self.shift)
        return x, self.y

@torch.no_grad()
def _ce_losses_on_augmented(model: torch.nn.Module, x: torch.Tensor, y: int, n: int, shift: int, device) -> np.ndarray:
    ds = AugmentN(x, y, n, shift)
    dl = DataLoader(ds, batch_size=min(64, n), shuffle=False)
    model.eval()
    losses = []
    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction='none')
        losses.append(loss.detach().cpu().numpy())
    return np.concatenate(losses, axis=0)

def _train_one_shadow(model_fn, feature_dims, num_classes, device, train_ds: Dataset, epochs: int, lr: float, wd: float, batch_size: int, seed: int):
    torch.cuda.empty_cache()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = model_fn.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
    return model

def _log_norm_pdf(x, mu, sigma, eps=1e-6):
    sigma = max(sigma, eps)
    return -0.5*np.log(2*np.pi) - np.log(sigma) - 0.5*((x - mu)/sigma)**2

def run_lira_online(pos_ds: Dataset, neg_ds: Dataset, base_pool_ds: Dataset, tar_model: torch.nn.Module, args, feature_dims, num_classes, device, logger):
    # 平衡正负样本
    min_size = min(len(pos_ds), len(neg_ds))
    pos_indices = random.sample(range(len(pos_ds)), min_size)
    neg_indices = random.sample(range(len(neg_ds)), min_size)
    
    logger.info(f"LiRA attack on {min_size*2} samples (balanced).")

    # 合并所有目标样本
    all_target_samples = []
    for idx in pos_indices:
        x, y = pos_ds[idx]
        all_target_samples.append((x, int(y), 1))  # (tensor, label, membership_label)
    for idx in neg_indices:
        x, y = neg_ds[idx]
        all_target_samples.append((x, int(y), 0))
    
    random.shuffle(all_target_samples)
    
    # 基础池大小
    pool_len = len(base_pool_ds)
    bs_train_size = min(args.lira_shadow_train_size, pool_len)
    batch_size = min(args.batch_size, 128)
    
    # 影子模型数量
    num_shadow_models = args.lira_shadow_models
    
    # 步骤1: 准备影子模型的训练数据集
    logger.info("Preparing shadow model datasets...")
    
    # 创建N个影子模型的数据集，每个包含约一半的目标样本
    shadow_model_datasets = []
    target_sample_membership = []  # 记录每个目标样本在哪些影子模型中
    
    # 初始化成员关系记录
    for _ in range(len(all_target_samples)):
        target_sample_membership.append([])
    
    # 为每个影子模型创建训练数据集
    for shadow_idx in range(num_shadow_models):
        # 从基础池中采样
        pool_indices = random.sample(range(pool_len), bs_train_size)
        
        # 选择约一半的目标样本加入该影子模型
        shadow_target_samples = []
        for target_idx, (x, y, _) in enumerate(all_target_samples):
            if random.random() < 0.5:  # 约50%的概率包含该目标样本
                shadow_target_samples.append((x, y))
                target_sample_membership[target_idx].append(shadow_idx)
        
        # 合并基础池样本和目标样本
        if shadow_target_samples:
            # 修复：直接使用shadow_target_samples，避免张量比较
            target_tensors = [x for x, y in shadow_target_samples]
            target_labels = [y for x, y in shadow_target_samples]
            
            # 创建目标样本的数据集
            class TensorListDataset(Dataset):
                def __init__(self, tensors, labels):
                    self.tensors = tensors
                    self.labels = labels
                
                def __len__(self):
                    return len(self.tensors)
                
                def __getitem__(self, idx):
                    return self.tensors[idx], self.labels[idx]
            
            target_ds = TensorListDataset(target_tensors, target_labels)
            final_shadow_ds = ConcatDataset([
                TensorAugmentDataset(Subset(base_pool_ds, pool_indices), args.lira_shift),
                TensorAugmentDataset(target_ds, args.lira_shift)
            ])
        else:
            final_shadow_ds = TensorAugmentDataset(Subset(base_pool_ds, pool_indices), args.lira_shift)
        
        shadow_model_datasets.append(final_shadow_ds)
    
    # 步骤2: 并行训练所有影子模型
    logger.info(f"Training {num_shadow_models} shadow models...")
    shadow_models = []
    
    for shadow_idx in range(num_shadow_models):
        model = get_source_model(args.arch, feature_dims, num_classes, pretrained=False)
        shadow_model = _train_one_shadow(
            model, feature_dims, num_classes, device, 
            shadow_model_datasets[shadow_idx],
            epochs=args.lira_epochs, lr=args.lira_lr, wd=args.lira_weight_decay,
            batch_size=batch_size, seed=args.seed + shadow_idx
        )
        shadow_models.append(shadow_model)
        if (shadow_idx + 1) % 5 == 0:
            logger.info(f"Trained {shadow_idx + 1}/{num_shadow_models} shadow models")
    
    # 步骤3: 为所有目标样本计算损失
    logger.info("Computing losses for all target samples...")
    
    # 预计算所有影子模型对所有目标样本的损失
    shadow_losses = np.zeros((num_shadow_models, len(all_target_samples), args.lira_augmentations))
    
    for shadow_idx, shadow_model in enumerate(shadow_models):
        for target_idx, (x, y, _) in enumerate(all_target_samples):
            losses = _ce_losses_on_augmented(shadow_model, x, y, args.lira_augmentations, args.lira_shift, device)
            shadow_losses[shadow_idx, target_idx, :] = losses
        
        if (shadow_idx + 1) % 5 == 0:
            logger.info(f"Computed losses for {shadow_idx + 1}/{num_shadow_models} shadow models")
    
    # 步骤4: 计算每个目标样本的似然比得分
    logger.info("Computing likelihood ratios...")
    
    scores = []
    labels = []
    
    for target_idx, (x, y, mlabel) in enumerate(all_target_samples):
        # 获取该目标样本在哪些影子模型中
        in_shadow_indices = target_sample_membership[target_idx]
        out_shadow_indices = [i for i in range(num_shadow_models) if i not in in_shadow_indices]
        
        if not in_shadow_indices or not out_shadow_indices:
            # 如果目标样本出现在所有影子模型或没有影子模型中，跳过
            continue
        
        # 获取目标模型在该样本上的观测损失
        obs_losses = _ce_losses_on_augmented(tar_model, x, y, args.lira_augmentations, args.lira_shift, device)
        
        # 计算似然比
        llr = 0.0
        for aug_idx in range(args.lira_augmentations):
            # IN 分布的参数估计（包含该样本的影子模型）
            in_losses = shadow_losses[in_shadow_indices, target_idx, aug_idx]
            mu_in, std_in = float(np.mean(in_losses)), float(np.std(in_losses) + 1e-6)
            
            # OUT 分布的参数估计（不包含该样本的影子模型）
            out_losses = shadow_losses[out_shadow_indices, target_idx, aug_idx]
            mu_out, std_out = float(np.mean(out_losses)), float(np.std(out_losses) + 1e-6)
            
            # 当前观测值
            x_obs = float(obs_losses[aug_idx])
            
            # 累积对数似然比
            llr += _log_norm_pdf(x_obs, mu_in, std_in) - _log_norm_pdf(x_obs, mu_out, std_out)
        
        scores.append(llr)
        labels.append(mlabel)
        
        if (target_idx + 1) % 50 == 0:
            logger.info(f"Processed {target_idx + 1}/{len(all_target_samples)} target samples")
    
    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.int64)
    
    # 清理影子模型以释放GPU内存
    for model in shadow_models:
        del model
    torch.cuda.empty_cache()
    
    # 计算指标
    if len(scores) == 0:
        logger.warning("No valid samples for evaluation!")
        # 返回默认结果
        auc_tr, acc_tr = 0.5, 0.5
        auc_ts, acc_ts = 0.5, 0.5
    else:
        idx_train, idx_test = train_test_split(np.arange(len(scores)), test_size=args.test_ratio, random_state=args.seed)
        
        def _metrics(idxs):
            y = labels[idxs]
            s = scores[idxs]
            if len(np.unique(y)) > 1:
                auc = roc_auc_score(y, s)
            else:
                auc = 0.5
            acc = float(np.mean((s > 0).astype(int) == y))
            return auc, acc
        
        auc_tr, acc_tr = _metrics(idx_train)
        auc_ts, acc_ts = _metrics(idx_test)
    
    # 构造兼容的输出格式
    res_tr = {}
    res_ts = {}
    for apoc_ in CONSTRUCTIONSELF:
        key = apoc_.name
        res_tr[key] = np.array([auc_tr, acc_tr])
        res_ts[key] = np.array([auc_ts, acc_ts])
    
    logger.info(f"LiRA attack completed: Train AUC={auc_tr:.4f}, Test AUC={auc_ts:.4f}")
    
    return [res_tr, res_ts]

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

    # ============ LiRA 专用分支 ============
    if args.attack_model == 'lira':
        logger.info("Running Online LiRA attack with on-the-fly shadow training...")
        # 选择 shadow 训练池：优先 data_remain['train']，否则使用原始 train_data
        base_pool = None
        try:
            if isinstance(data_remain, dict) and ('train' in data_remain):
                base_pool = data_remain['train']
        except Exception:
            base_pool = None
        if base_pool is None:
            base_pool = train_data

        # 将 base_pool 的 transform 设为“normal”，避免额外随机增强（张量域增强替代）
        # 安全保存/恢复
        def get_dataset_transform(dataset):
            if hasattr(dataset, 'transform'):
                return dataset.transform
            elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'transform'):
                return dataset.dataset.transform
            else:
                return None
        def set_dataset_transform(dataset, transform):
            if hasattr(dataset, 'transform'):
                dataset.transform = transform
            elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'transform'):
                dataset.dataset.transform = transform

        orig_base_tf = get_dataset_transform(base_pool)
        normal_tf = Transforms(*DataStore.get_normalizer(args.dataset), 
                               ).get_transform("normal", DataStore.get_dataset_info(args.dataset)[1])
        # 注意：Transforms.get_transform 需要 mean/std/size；上面调用保持兼容
        set_dataset_transform(base_pool, normal_tf)

        try:
            attack_res = run_lira_online(
                pos_ds=pos_data,
                neg_ds=neg_data,
                base_pool_ds=base_pool,
                tar_model=tar_model,
                args=args,
                feature_dims=feature_dims,
                num_classes=num_classes,
                device=device,
                logger=logger
            )
        finally:
            set_dataset_transform(base_pool, orig_base_tf)

        # 下方沿用原有结果写出逻辑
        train_res_avg, test_res_avg = defaultdict(list), defaultdict(list)
        for k_ in attack_res[0].keys():
            train_res_avg[k_].append(np.array(attack_res[0][k_]))
            test_res_avg[k_].append(np.array(attack_res[1][k_]))
        res_tr_avg, res_ts_avg = dict(), dict()
        for k_ in train_res_avg.keys():
            res_tr_avg[k_] = np.mean(train_res_avg[k_], axis=0)
            res_ts_avg[k_] = np.mean(test_res_avg[k_], axis=0)
        attack_res = [res_tr_avg, res_ts_avg]
        logger.info(f"LiRA Attack result: {attack_res}")
    else:
        # ======== 原有通用攻击流程保留 ========
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
        compatible_transform = preproc.get_transform("randaug", num_classes)

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
        
        for trial in range(args.n_trails):
            attack_train, attack_test = train_test_split(attack_data, test_size=args.test_ratio, random_state=trial) #shuffle=True
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

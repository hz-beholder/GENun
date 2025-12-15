#! /usr/bin/env python

import os, json, time, argparse, numpy as np, torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.utils.data import Subset, ConcatDataset, DataLoader
from copy import deepcopy
from scipy.stats import norm, rankdata

# 假设这些本地模块存在于您的环境中，保持原样导入
from schema import BasicUnlearnSchema
from logger import create_logger
from transforms import Transforms
from utils import argparse2bool, list_of_ints, seed_everything, mkdir
from data_tool import DataStore, DataLoaderTool
from model_bases import DeepModels
from tool import complete_parameters, PathGenerator as pather
from unlearners import *
from models_genun import GeneModUnlearn

# -------------------------- 参数 --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Shadow+LIRA for CIFAR-10")
    parser.add_argument('--logname', default='shadow_lira')
    parser.add_argument('--log_dir', default='.')
    parser.add_argument('--out_dir', default='./shadow_outs/')
    parser.add_argument('--exp_name', default='shadow_logs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--select_classes', type=list_of_ints, default=None)
    parser.add_argument('--num_shadows', type=int, default=64)
    parser.add_argument('--shadow_seed_start', type=int, default=1000)
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--pretrained', type=argparse2bool, default=True)

    # 遗忘方法
    parser.add_argument('--unlearn_method', type=str, default='ORG',
                        choices=['ORG', 'RT', 'FT', 'RL', 'GA', 'WGA', 'LKL', 'FRA', 'UI', 'CR',
                                 'L1FT', 'SALUN', 'GENM'],
                        help='unlearning method for target model (default: ORG)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--optim', default='Adam')
    parser.add_argument('--maxlr', type=float, default=0.001)
    parser.add_argument('--minlr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=5e-05)
    parser.add_argument('--train_transform', default='normal')
    parser.add_argument('--test_transform', default='test')
    parser.add_argument('--lossfn', default='ce')

    # 特定参数
    parser.add_argument('--lamb', type=float, default=1e-6, help='regularization parameter')
    parser.add_argument('--last_k', type=int, default=2, help='last k layers for LKL')
    parser.add_argument('--re_init', type=argparse2bool, default=True, help='reinitialize for LKL')
    parser.add_argument('--mask_threshold', type=float, default=0.5, help='mask threshold for SALUN')
    parser.add_argument('--evaluate_mia', type=argparse2bool, default=True)
    return parser.parse_args()

# -------------------------- 数据划分 --------------------------
def make_cifar10_splits(args, logger):
    from torch.utils.data import random_split
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    processor = Transforms(dt_mean, dt_std, dt_size)
    train_tf = processor.get_transform(args.train_transform, num_classes)
    test_tf  = processor.get_transform(args.test_transform, num_classes)

    train_raw, test_raw = DataLoaderTool.load_dataset(
        args.dataset, train_tf, test_tf,
        num_samples=args.num_samples, select_classes=args.select_classes)

    rng = torch.Generator().manual_seed(0)
    train_set, valid_set, _= random_split(train_raw, [45000, 5000, 0], generator=rng)
    retain_set, forget_set = random_split(train_set, [40000, 5000], generator=rng)
    test_set = test_raw

    logger.info(f"Data splits -> retain:{len(retain_set)} forget:{len(forget_set)} "
                f"valid:{len(valid_set)} test:{len(test_set)}")
    return {'retain': retain_set, 'forget': forget_set,
            'valid': valid_set, 'test': test_set, 'train': train_set}

# -------------------------- shadow 数据 --------------------------
def shadow_data_split(args, shadow_id, retain, forget, valid, test_front, logger):
    rng = np.random.RandomState(args.shadow_seed_start + shadow_id)
    pool = ConcatDataset([retain, valid])
    idx_all = np.arange(len(pool))
    rng.shuffle(idx_all)

    train_idx = idx_all[:30000]
    valid_idx = idx_all[40000:45000]

    shadow_train = Subset(pool, train_idx)
    shadow_valid = Subset(pool, valid_idx)
    shadow_test  = shadow_valid

    retain_in_train = sum(1 for i in train_idx if i < 40000)
    valid_in_train  = len(train_idx) - retain_in_train
    logger.info(f"Shadow {shadow_id} | Train={len(shadow_train)} "
                f"(retain≈{retain_in_train}, valid≈{valid_in_train}) "
                f"| Valid={len(shadow_valid)} | Test={len(shadow_test)}")

    if shadow_id < args.num_shadows // 2:
        shadow_train = ConcatDataset([shadow_train, forget])
        logger.info(f"  -> with forget (total {len(shadow_train)})")
    else:
        logger.info(f"  -> no forget")

    return {'train': shadow_train, 'valid': shadow_valid, 'test': shadow_test}

# -------------------------- 训练原始模型（ORG） --------------------------
def train_original_model(args, data_splits, device, logger):
    seed_everything(args.seed)
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    if args.arch in ['resnet50', 'densenet', 'vit']:
        dt_size = 224
        feature_dims = [3, dt_size, dt_size]

    model_params = complete_parameters(vars(args))
    model_params['num_classes'] = num_classes

    model = DeepModels(args.arch, feature_dims, num_classes,
                       args.log_dir, "target_original", args.pretrained)
    model.parameter_config(**model_params)
    model._model.to(device)

    ckpt_path = pather.get_checkpoint_path(f"{args.dataset}_target_original")
    builder = BuildLearn(args.log_dir, "target_original_train")

    logger.info(f"Training original model | Train: {len(data_splits['train'])} "
                f"| Valid: {len(data_splits['valid'])} | Test: {len(data_splits['test'])}")

    trained = builder.model_unlearn(model, data_splits, args.batch_size,
                                    device, ckpt_path, is_train=True)
    return trained

# -------------------------- 应用遗忘方法 --------------------------
def apply_unlearn_method(args, original_model, data_splits, device, logger):
    unlearn_method = args.unlearn_method
    forget_data = data_splits['forget']
    retain_data = data_splits['retain']

    data_dict = {
        'train': data_splits['train'],
        'valid': data_splits['valid'],
        'test':  data_splits['test'],
        'forget': forget_data,
        'retain': retain_data
    }

    extra_params = {}
    _ext_name_ = ""

    if unlearn_method == 'GENM':
        logger.info("Using GenUn (GENM) unlearning method")
        mu_er = GeneModUnlearn(args.log_dir, args.logname, args.out_dir, name=f"target_genm")
        dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
        feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
        if args.arch in ['resnet50', 'densenet', 'vit']:
            dt_size = 224
            feature_dims = [3, dt_size, dt_size]

        # 把 args 转成 GenUn 需要的格式
        genun_args = argparse.Namespace(
            dataset=args.dataset,
            arch=args.arch,
            num_samples=args.num_samples,
            valid_size=len(data_splits['valid']),
            num_to_forget=len(forget_data),
            forget_classes=None,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.maxlr,
            weight_decay=args.weight_decay,
            optim=args.optim.lower(),
            patience=args.patience,
            scheduler='CosineAnnealingWarmRestarts',
            lossfn=args.lossfn,
            preproc_train_transform=args.train_transform,
            preproc_test_transform=args.test_transform,
            online_train_aug='none',
            online_forget_aug='none',
            over_forget=True,
            regularizer='l1',
            gamma=1e-5,
            alpha=1.0,
            no_reg_epochs=0,
            dynamic_weight=False,
            class_wise=True,
            sample_ratio=1.0,
            save_checkpoint=False,
            num_workers=4,
            cuda=int(args.cuda) if args.cuda.isdigit() else 0,
            seed=args.seed,
            model_path="/mnt/wanghan/outs/cifar10_-1/checkpoints/cifar10_-1_resnet18/ORG_GENE_M_0.pt"
        )

        # 构造原始模型
        from model_bases import DeepModels
        from tool import complete_parameters
        model_params = complete_parameters(vars(args))
        model_params['num_classes'] = num_classes
        original_model_cp = DeepModels(args.arch, feature_dims, num_classes,
                                       args.log_dir, "target_genm_orig", True)
        original_model_cp.parameter_config(**model_params)
        original_model_cp._model.load_state_dict(original_model._model.state_dict())
        original_model_cp._model.to(device)

        # 配置 GenUn
        mu_er.set_params(**vars(genun_args))
        mu_er.args = genun_args
        mu_er.config_transform(dt_mean, dt_std, dt_size)
        mu_er.set_data(data_dict, num_classes, args.batch_size, 4)

        # 执行遗忘
        ckpt_path = pather.get_checkpoint_path(f"{args.dataset}_target_genm")
        unlearned_model, _ = mu_er.unlearn(
            original_model_cp._model, None, None,
            args.patience, genun_args.scheduler, genun_args.optim,
            genun_args.online_train_aug, genun_args.online_forget_aug,
            device, ckpt_path, model_type=args.arch, over_forget=True,
            suffix=''
        )
        # 包装成 DeepModels 返回
        unlearn_wrap = DeepModels(args.arch, feature_dims, num_classes,
                                  args.log_dir, "target_genm", False)
        unlearn_wrap.parameter_config(**model_params)
        unlearn_wrap._model.load_state_dict(unlearned_model.state_dict())
        unlearn_wrap._model.to(device)
        return unlearn_wrap

    if unlearn_method in ["ORG", "none", "original"]:
        logger.info("Using original model without unlearning")
        return original_model

    elif unlearn_method in ["RT", "scratch", "retrain"]:
        unlearn_schema = UnlearnRetrain(args.log_dir, "target_unlearn")
        logger.info("Using Retrain unlearning method")

    elif unlearn_method in ["FT", "finetune"]:
        unlearn_schema = UnlearnFinetune(args.log_dir, "target_unlearn")
        logger.info("Using Finetune unlearning method")

    elif unlearn_method in ['L1FT', 'l1_sparse']:
        _ext_name_ = f"_l1_regular_{args.lamb}"
        args.regularization = 'l1'
        logger.info(f"Using L1 Sparse unlearning with lambda: {args.lamb}")
        unlearn_schema = UnlearnSparseL1(args.log_dir, "target_unlearn")

    elif unlearn_method in ["RL", "random_label"]:
        feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
        extra_params['num_classes'] = num_classes
        unlearn_schema = UnlearnRandomLabels(args.log_dir, "target_unlearn")
        logger.info("Using Random Labels unlearning method")

    elif unlearn_method in ["SALUN", "salun"]:
        _ext_name_ = f"_maskth-{args.mask_threshold}"
        extra_params['mask_threshold'] = args.mask_threshold
        unlearn_schema = UnlearnSalunRL(args.log_dir, "target_unlearn")
        logger.info("Using SALUN unlearning method")

    elif unlearn_method in ["GA", "gradient_ascent", "negative_gradient"]:
        unlearn_schema = UnlearnGradientAscent(args.log_dir, "target_unlearn")
        logger.info("Using Gradient Ascent unlearning method")

    elif unlearn_method in ["LKL", "lastklayer"]:
        _ext_name_ = f"_lastk-{args.last_k}_reinit-{args.re_init}"
        extra_params['last_k'] = args.last_k
        extra_params['re_init'] = args.re_init
        unlearn_schema = UnlearnLastKLayer(args.log_dir, "target_unlearn")
        logger.info("Using Last K Layer unlearning method")

    else:
        raise NotImplementedError(f"Unlearn method {unlearn_method} not implemented")

    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    model_params = complete_parameters(vars(args))
    model_params['num_classes'] = num_classes

    if unlearn_method in ["RT", "scratch", "retrain"]:
        unlearn_model = DeepModels(args.arch, feature_dims, num_classes,
                                   args.log_dir, "target_unlearn", args.pretrained)
        unlearn_model.parameter_config(**model_params)
        unlearn_model._model.to(device)
    else:
        unlearn_model = deepcopy(original_model)
    unlearn_model.parameter_config(**model_params)

    ckpt_path = pather.get_checkpoint_path(f"{args.dataset}_target_unlearn")
    model_res = unlearn_schema.model_unlearn(unlearn_model, data_dict, args.batch_size,
                                             device, ckpt_path, **extra_params)
    logger.info(f"Unlearning method {unlearn_method} completed")
    return model_res

def get_target_model_filename(unlearn_method, **kwargs):
    base_name = f"target_model_{unlearn_method}"
    if unlearn_method == 'L1FT':
        base_name += f"_lamb{kwargs.get('lamb', '1e-6')}"
    elif unlearn_method == 'SALUN':
        base_name += f"_maskth{kwargs.get('mask_threshold', '0.5')}"
    elif unlearn_method == 'LKL':
        base_name += f"_lastk{kwargs.get('last_k', '2')}_reinit{kwargs.get('re_init', 'True')}"
    return base_name + ".pth"

# -------------------------- 训练目标模型 --------------------------
def train_target_model(args, data_splits, device, logger):
    original_model_path = os.path.join(args.out_dir, "target_model_ORG.pth")
    if not os.path.exists(original_model_path):
        logger.info("Training original model (ORG method)...")
        original_model = train_original_model(args, data_splits, device, logger)
        torch.save(original_model._model.state_dict(), original_model_path)
        logger.info("Original model trained & saved.")
    else:
        logger.info("Loading existing original model...")
        dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
        feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
        if args.arch in ['resnet50', 'densenet', 'vit']:
            dt_size = 224
            feature_dims = [3, dt_size, dt_size]
        model_params = complete_parameters(vars(args))
        model_params['num_classes'] = num_classes
        original_model = DeepModels(args.arch, feature_dims, num_classes,
                                    args.log_dir, "target_original", args.pretrained)
        original_model.parameter_config(**model_params)
        original_model._model.load_state_dict(torch.load(original_model_path, map_location=device, weights_only=True))
        original_model._model.to(device)

    if args.unlearn_method in ['ORG', 'none', 'original']:
        logger.info("Using original model as target model")
        return original_model
    logger.info(f"Applying unlearn method: {args.unlearn_method}")
    target_model = apply_unlearn_method(args, original_model, data_splits, device, logger)
    return target_model

# -------------------------- 评估 --------------------------
@torch.no_grad()
def eval_model(model, data_dict, batch_size, device, lossfn='ce'):
    names = ['train', 'retain', 'valid', 'test', 'forget', 'remain']
    test_ds = data_dict['test']
    print('[shadow] eval_model  test_ds len =', len(test_ds))
    if hasattr(test_ds, 'indices'):
        print('[shadow] first 5 indices =', test_ds.indices[:5])

    remain_data = ConcatDataset([data_dict['retain'], data_dict['valid'], data_dict['test']])
    loaders = {k: DataLoader(data_dict[k] if k != 'remain' else remain_data,
                             batch_size=batch_size, shuffle=True, num_workers=4) for k in names}
    model._model.eval()
    accs = {}
    for k in names:
        correct = total = 0
        for x, y in loaders[k]:
            x, y = x.to(device), y.to(device)
            out = model._model(x)
            correct += (out.argmax(1) == y).sum().item()
            total   += y.size(0)
        accs[k] = correct / total
    return accs

# -------------------------- 影子模型 --------------------------
def train_shadow_model(args, model_id, data_splits, device, logger):
    seed_everything(args.shadow_seed_start + model_id)
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    if args.arch in ['resnet50', 'densenet', 'vit']:
        dt_size = 224
        feature_dims = [3, dt_size, dt_size]
    model_params = complete_parameters(vars(args))
    model_params['num_classes'] = num_classes
    model = DeepModels(args.arch, feature_dims, num_classes,
                       args.log_dir, f"shadow_{model_id}", args.pretrained)
    model.parameter_config(**model_params)
    model._model.to(device)
    ckpt_path = pather.get_checkpoint_path(f"{args.dataset}_shadow_{model_id}")
    builder = BuildLearn(args.log_dir, f"shadow_train_{model_id}")
    logger.info(f"Shadow {model_id} | Train: {len(data_splits['train'])} "
                f"| Valid: {len(data_splits['valid'])} | Test: {len(data_splits['test'])}")
    trained = builder.model_unlearn(model, data_splits, args.batch_size,
                                    device, ckpt_path, is_train=True)
    return trained

def save_model(model, path):
    torch.save(model._model.state_dict(), path)

def load_shadow_model(args, shadow_id, device):
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    if args.arch in ['resnet50', 'densenet', 'vit']:
        dt_size = 224
        feature_dims = [3, dt_size, dt_size]
    model_params = complete_parameters(vars(args))
    model_params['num_classes'] = num_classes
    model = DeepModels(args.arch, feature_dims, num_classes,
                       args.log_dir, f"shadow_{shadow_id}", args.pretrained)
    model.parameter_config(**model_params)
    path = os.path.join(args.out_dir, f"shadow_{shadow_id}.pth")
    model._model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model._model.to(device)
    return model

def load_target_model(args, device):
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    if args.arch in ['resnet50', 'densenet', 'vit']:
        dt_size = 224
        feature_dims = [3, dt_size, dt_size]
    model_params = complete_parameters(vars(args))
    model_params['num_classes'] = num_classes
    target_model_filename = get_target_model_filename(
        args.unlearn_method,
        lamb=args.lamb,
        mask_threshold=args.mask_threshold,
        last_k=args.last_k,
        re_init=args.re_init
    )
    model = DeepModels(args.arch, feature_dims, num_classes,
                       args.log_dir, "target", args.pretrained)
    model.parameter_config(**model_params)
    path = os.path.join(args.out_dir, target_model_filename)
    model._model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model._model.to(device)
    return model

# -------------------------- logits --------------------------
@torch.no_grad()
def get_logits(model, dataset, batch_size, device, num_workers=0):
    """
    修改：同时返回 Logits 和 Labels (Ground Truth)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model._model.eval()
    logits = []
    labels = []
    for x, y in loader:
        logits.append(model._model(x.to(device)).cpu())
        labels.append(y.cpu())
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()

def softmax_logit_score(logits_np, labels, epsilon=1e-10, return_probs=False):
    """
    修改：根据 Labels 提取真实类别的概率，而非最大概率。
    logits_np: (N_models, N_samples, N_classes) or (N_samples, N_classes)
    labels: (N_samples,)
    """
    # 1. 计算 Softmax
    max_logits = np.max(logits_np, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_np - max_logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # 2. 提取真实标签对应的概率 (Gather操作)
    if logits_np.ndim == 3: # Case: multiple shadow models
        n_models, n_samples, n_classes = logits_np.shape
        # 扩展 labels: (1, n_samples) -> (n_models, n_samples)
        labels_expanded = np.tile(labels[None, :], (n_models, 1))

        model_indices = np.arange(n_models)[:, None]
        sample_indices = np.arange(n_samples)[None, :]
        true_class_probs = softmax_probs[model_indices, sample_indices, labels_expanded]
    else: # Case: single target model (N_samples, N_classes)
        n_samples, n_classes = logits_np.shape
        sample_indices = np.arange(n_samples)
        true_class_probs = softmax_probs[sample_indices, labels]

    # 3. 计算 Logit Score: log(p / (1-p))
    logit_scores = np.log((true_class_probs + epsilon) / (1 - true_class_probs + epsilon))

    if return_probs:
        return logit_scores, true_class_probs
    return logit_scores

# -------------------------- LIRA 高斯参数 --------------------------
def estimate_lira_params(shadow_models, forget_plus_test, batch_size, device, out_dir, logger):
    n_models = len(shadow_models)
    n_samples = len(forget_plus_test)
    logits = np.zeros((n_models, n_samples, 10), dtype=np.float32)
    labels_all = None

    logger.info("Collecting shadow logits on forget+test(10k)...")
    for i, m in enumerate(shadow_models):
        l, targets = get_logits(m, forget_plus_test, batch_size, device, num_workers=0)
        logits[i] = l
        if labels_all is None:
            labels_all = targets
        logger.info(f"  shadow {i} done")

    # 修改：传入 labels 计算 Ground Truth 的 Logit Score
    conf, probs = softmax_logit_score(logits, labels_all, return_probs=True)

    labels_mask = np.zeros((n_models, n_samples), bool)
    labels_mask[:n_models // 2] = True # 前一半是 IN

    mean_in   = np.zeros(n_samples)
    std_in    = np.zeros(n_samples)
    mean_out  = np.zeros(n_samples)
    std_out   = np.zeros(n_samples)

    for s in range(n_samples):
        in_conf  = conf[labels_mask[:, s], s]
        out_conf = conf[~labels_mask[:, s], s]
        mean_in[s]   = np.mean(in_conf)
        std_in[s]    = np.std(in_conf)  + 1e-30
        mean_out[s]  = np.mean(out_conf)
        std_out[s]   = np.std(out_conf) + 1e-30

    params = {'mean_in': mean_in.tolist(), 'std_in': std_in.tolist(),
              'mean_out': mean_out.tolist(), 'std_out': std_out.tolist()}
    with open(os.path.join(out_dir, 'lira_params.json'), 'w') as f:
        json.dump(params, f)
    logger.info("Saved lira_params.json")
    return params, probs

def plot_lira_scatter(params, out_dir, logger):
    import matplotlib.pyplot as plt

    # 1. 提取数据
    n_forget = 5000
    mean_in = np.array(params['mean_in'])
    mean_out = np.array(params['mean_out'])

    # 拆分数据
    f_mean_in = mean_in[:n_forget]
    f_mean_out = mean_out[:n_forget]
    t_mean_in = mean_in[n_forget:]
    t_mean_out = mean_out[n_forget:]

    # 2. 绘图
    plt.figure(figsize=(8, 8))
    lims = [0, max(np.max(mean_out), np.max(mean_in))]
    plt.plot(lims, lims, '--', color='gray', alpha=0.5, zorder=0)

    plt.scatter(t_mean_out, t_mean_in, c='tab:blue', alpha=0.2, s=10,
                label='test (n=5000)', zorder=1, edgecolors='none')
    plt.scatter(f_mean_out, f_mean_in, c='tab:red', alpha=0.6, s=10,
                label='forget (n=5000)', zorder=2, edgecolors='none')

    plt.xlabel(r'$\mu_{out}$')
    plt.ylabel(r'$\mu_{in}$')
    plt.title(r'$\mu_{in}$ vs $\mu_{out}$ per sample (Optimized)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(out_dir, 'lira_scatter_optimized.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved optimized scatter plot to {save_path}")

def plot_lira_scatter_separated(params, out_dir, logger):
    import matplotlib.pyplot as plt

    n_forget = 5000
    mean_in = np.array(params['mean_in'])
    mean_out = np.array(params['mean_out'])

    max_val = max(np.max(mean_in), np.max(mean_out))
    min_val = min(np.min(mean_in), np.min(mean_out))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    axes[0].plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.5)
    axes[0].scatter(mean_out[:n_forget], mean_in[:n_forget],
                    c='tab:red', alpha=0.5, s=10, label='forget')
    axes[0].set_title('Forget Set Only (Members)')
    axes[0].set_xlabel(r'$\mu_{out}$')
    axes[0].set_ylabel(r'$\mu_{in}$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.5)
    axes[1].scatter(mean_out[n_forget:], mean_in[n_forget:],
                    c='tab:blue', alpha=0.5, s=10, label='test')
    axes[1].set_title('Test Set Only (Non-members)')
    axes[1].set_xlabel(r'$\mu_{out}$')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'lira_scatter_separated.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved separated scatter plot to {save_path}")

# ================= 新增功能：Fig 3 复现 =================
def plot_fig3_replication(params, raw_probs, num_shadows, out_dir, logger):
    """
    仿照论文 Fig 3 复现 (Optimized for Visual Consistency)。
    包含隐私约束筛选和双图绘制。
    """
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata

    # 1. 准备数据 (仅分析 Forget set)
    n_forget = 5000
    mean_in = np.array(params['mean_in'])[:n_forget]
    mean_out = np.array(params['mean_out'])[:n_forget]

    # === 关键修改：定义有效样本池 (valid_mask) ===
    score_gap = mean_in - mean_out
    valid_mask = score_gap > 0  # 仅保留 Member表现优于Non-member 的样本

    if np.sum(valid_mask) < 10:
        logger.warning("Few samples satisfy mean_in > mean_out. Relaxing constraint.")
        valid_mask = np.ones_like(valid_mask, dtype=bool)

    # 转换为排名
    rank_in = rankdata(mean_in)       # Rank高 = Easy Fit
    rank_out = rankdata(mean_out)     # Rank高 = Inlier

    def get_masked_idx(metric, maximize=True):
        valid_indices = np.where(valid_mask)[0]
        metric_valid = metric[valid_indices]
        if maximize:
            best_local_pos = np.argmax(metric_valid)
        else:
            best_local_pos = np.argmin(metric_valid)
        return valid_indices[best_local_pos]

    # 2. 筛选四个典型样本
    idx_easy_inlier = get_masked_idx(rank_in + rank_out, maximize=True)
    idx_easy_outlier = get_masked_idx(rank_in - rank_out, maximize=True)
    idx_hard_inlier = get_masked_idx(rank_out - rank_in, maximize=True)
    idx_hard_outlier = get_masked_idx(rank_in + rank_out, maximize=False)

    selected_indices = {
        'Easy to fit / Inlier': idx_easy_inlier,
        'Easy to fit / Outlier': idx_easy_outlier,
        'Hard to fit / Inlier': idx_hard_inlier,
        'Hard to fit / Outlier': idx_hard_outlier
    }

    logger.info("Fig 3 Replication (Constrained) - Selected Indices:")
    for k, v in selected_indices.items():
        logger.info(f"  {k}: Index {v} | Score_In={mean_in[v]:.2f}, Score_Out={mean_out[v]:.2f}")

    split_idx = num_shadows // 2
    # ==========================================
    # 绘图组 1: 原始 Loss (Log Scale)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 定义固定的 X 轴范围
    x_min_fixed = 1e-10
    x_max_fixed = 50

    for i, (title, idx) in enumerate(selected_indices.items()):
        ax = axes[i]
        sample_probs = raw_probs[:, idx]
        probs_in = sample_probs[:split_idx]
        probs_out = sample_probs[split_idx:]

        losses_in = -np.log(probs_in + 1e-30)
        losses_out = -np.log(probs_out + 1e-30)

        bins = np.logspace(np.log10(x_min_fixed), np.log10(x_max_fixed), 40)

        ax.hist(losses_in, bins=bins, alpha=0.7, color='tab:red',
                label='member', density=False)
        ax.hist(losses_out, bins=bins, alpha=0.7, color='tab:blue',
                label='non-member', density=False)

        ax.set_ylim(0, 20)
        ax.set_xlim(x_min_fixed, x_max_fixed)
        ax.set_xscale('log')

        ax.set_title(f"{title}\n(Index: {idx})")
        ax.set_xlabel('Cross Entropy Loss (Log Scale)')
        ax.set_ylabel('Number of Models')
        if i == 0: ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path_loss = os.path.join(out_dir, 'fig3_replication_loss_log.png')
    plt.savefig(save_path_loss, dpi=150)
    plt.close()
    logger.info(f"Saved Loss Plot to {save_path_loss}")
    # ==========================================
    # 绘图组 2: Logit Scaling (Linear Scale)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    x_min_linear = -15
    x_max_linear = 20

    for i, (title, idx) in enumerate(selected_indices.items()):
        ax = axes[i]
        sample_probs = raw_probs[:, idx]
        probs_in = sample_probs[:split_idx]
        probs_out = sample_probs[split_idx:]

        epsilon = 1e-30
        logits_in = np.log(probs_in + epsilon) - np.log(1 - probs_in + epsilon)
        logits_out = np.log(probs_out + epsilon) - np.log(1 - probs_out + epsilon)

        bins = np.linspace(x_min_linear, x_max_linear, 40)

        ax.hist(logits_in, bins=bins, alpha=0.7, color='tab:red',
                label='member', density=False)
        ax.hist(logits_out, bins=bins, alpha=0.7, color='tab:blue',
                label='non-member', density=False)

        ax.set_ylim(0, 10)
        ax.set_xlim(x_min_linear, x_max_linear)

        ax.set_title(f"{title}\n(Index: {idx})")
        ax.set_xlabel('Logit Scaled Score (Linear Scale)')
        ax.set_ylabel('Number of Models')
        if i == 0: ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path_logit = os.path.join(out_dir, 'fig3_replication_logit_linear.png')
    plt.savefig(save_path_logit, dpi=150)
    plt.close()
    logger.info(f"Saved Logit Plot to {save_path_logit}")

# ================= 新增功能：混合直方图 =================
def plot_mixed_logit_hist(raw_probs, num_shadows, out_dir, logger):
    """
    绘制 Logit Score 混合直方图 (Mixed Histogram)。
    分别对 Forget Set 和 Test Set 绘制。
    """
    import matplotlib.pyplot as plt

    n_split = 5000
    if raw_probs.shape[1] < n_split * 2:
        logger.warning(f"Not enough samples for test mixed hist. Need {n_split*2}, got {raw_probs.shape[1]}")
        data_pairs = [('Forget', raw_probs[:, :n_split])]
    else:
        data_pairs = [
            ('Forget', raw_probs[:, :n_split]),
            ('Test', raw_probs[:, n_split:])
        ]

    split_idx = num_shadows // 2
    epsilon = 1e-30

    for name, probs in data_pairs:
        probs_in = probs[:split_idx, :].flatten()
        probs_out = probs[split_idx:, :].flatten()

        logits_in = np.log(probs_in + epsilon) - np.log(1 - probs_in + epsilon)
        logits_out = np.log(probs_out + epsilon) - np.log(1 - probs_out + epsilon)

        plt.figure(figsize=(8, 6))
        all_logits = np.concatenate([logits_in, logits_out])
        min_val = np.min(all_logits)
        max_val = np.max(all_logits)
        bins = np.linspace(min_val, max_val, 60)

        plt.hist(logits_in, bins=bins, alpha=0.5, label='IN (Shadow 0-N/2)', color='tab:blue')
        plt.hist(logits_out, bins=bins, alpha=0.5, label='OUT (Shadow N/2-N)', color='tab:orange')

        plt.title(f'{name} samples - mixed hist')
        plt.xlabel('Logit Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(out_dir, f'{name.lower()}_mixed_hist.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Saved {name} mixed hist to {save_path}")

# -------------------------- 攻击目标模型 --------------------------
def lira_target(target_model, forget_plus_test, params, batch_size, device):
    logits, labels = get_logits(target_model, forget_plus_test, batch_size, device, num_workers=0)
    conf = softmax_logit_score(logits, labels) # 单模型模式

    mean_in  = np.array(params['mean_in'])
    std_in   = np.array(params['std_in'])
    mean_out = np.array(params['mean_out'])
    std_out  = np.array(params['std_out'])
    std_in  += 1e-2
    std_out += 1e-2
    from scipy.stats import norm
    pr_in  = -norm.logpdf(conf, mean_in, std_in)
    pr_out = -norm.logpdf(conf, mean_out, std_out)
    scores = pr_out - pr_in
    y_true = np.zeros(len(scores))
    y_true[:5000] = 1
    return scores, y_true

# -------------------------- ROC --------------------------
def plot_roc(scores, y_true, out_dir, logger, unlearn_method):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    bal_accs = 1 - (fpr + (1 - tpr)) / 2
    best_idx = np.argmax(bal_accs)
    best_threshold = thresholds[best_idx]
    y_pred = (scores >= best_threshold).astype(int)
    member_mask = y_true == 1
    member_acc = np.mean(y_pred[member_mask] == y_true[member_mask]) if np.sum(member_mask) > 0 else 0
    non_member_mask = y_true == 0
    non_member_acc = np.mean(y_pred[non_member_mask] == y_true[non_member_mask]) if np.sum(non_member_mask) > 0 else 0
    low = tpr[np.where(fpr < 0.001)[0][-1]] if np.any(fpr < 0.001) else 0.0

    logger.info(f"LIRA AUC={roc_auc:.4f}  BalAcc={bal_accs[best_idx]:.4f}")
    logger.info(f"best_threshold: {best_threshold:.4f}")
    logger.info(f"Member Accuracy (TPR): {member_acc:.4f}")
    logger.info(f"Non-member Accuracy (TNR): {non_member_acc:.4f}")
    logger.info(f"TPR@0.1%FPR={low:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}', linewidth=2)
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'LIRA ROC - {unlearn_method} (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'lira_roc_{unlearn_method}_linear.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}', linewidth=2)
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.8)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel('False Positive Rate (log scale)')
    plt.ylabel('True Positive Rate (log scale)')
    plt.title(f'LIRA ROC - {unlearn_method} (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'lira_roc_{unlearn_method}_log.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'auc': roc_auc,
        'bal_acc': bal_accs[best_idx],
        'member_acc': member_acc,
        'non_member_acc': non_member_acc,
        'tpr@0.1%fpr': low
    }

# -------------------------- 工具 --------------------------
def find_missing_shadows(out_dir, num_shadows):
    missing = []
    for i in range(num_shadows):
        if not os.path.exists(os.path.join(out_dir, f'shadow_{i}.pth')):
            missing.append(i)
    return missing

def target_model_exists(args, out_dir):
    target_model_filename = get_target_model_filename(
        args.unlearn_method,
        lamb=args.lamb,
        mask_threshold=args.mask_threshold,
        last_k=args.last_k,
        re_init=args.re_init
    )
    return os.path.exists(os.path.join(out_dir, target_model_filename))

# -------------------------- main --------------------------
def main():
    args = parse_args()
    mkdir(args.out_dir)
    logger = create_logger(args.log_dir, args.logname)
    logger.info(args)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    data = make_cifar10_splits(args, logger)
    retain, forget, valid, test, train_set = data['retain'], data['forget'], data['valid'], data['test'], data['train']

    test_front = Subset(test, range(5000))
    test_back  = Subset(test, range(5000, 10000))
    forget_plus_test = ConcatDataset([forget, test_back])
    # =====    =====
    print('[shadow] GenUn will receive test_front len =', len(test_front))
    print('[shadow] test_front first 5 indices =', test_front.indices[:5])

    # ----- 训练或加载目标模型 -----
    if not target_model_exists(args, args.out_dir):
        logger.info(f"Training target model with unlearn method: {args.unlearn_method}")
        target_data_splits = {
            'train': train_set,
            'valid': valid,
            'test': test_front,
            'retain': retain,
            'forget': forget
        }
        target_model = train_target_model(args, target_data_splits, device, logger)
        target_model_filename = get_target_model_filename(
            args.unlearn_method,
            lamb=args.lamb,
            mask_threshold=args.mask_threshold,
            last_k=args.last_k,
            re_init=args.re_init
        )
        save_path = os.path.join(args.out_dir, target_model_filename)
        save_model(target_model, save_path)
        logger.info(f"Target model trained & saved at: {save_path}")
    else:
        logger.info("Loading existing target model...")
        target_model = load_target_model(args, device)

    # ======== 遗忘后模型准确率打印 =========
    logger.info("++++++++++ Post-Unlearn Accuracy ++++++++++")
    accs = eval_model(target_model, {
        'train': train_set, 'retain': retain, 'valid': valid,
        'test': test, 'forget': forget
    }, args.batch_size, device, args.lossfn)
    logger.info(f"[Train, Train-retain, Valid, Test, Forget, Remain-Data]  "
                f"Acc: {accs['train']:.4f}, {accs['retain']:.4f}, {accs['valid']:.4f}, "
                f"{accs['test']:.4f}, {accs['forget']:.4f}, {accs['remain']:.4f}")
    # ============================================

    missing_shadows = find_missing_shadows(args.out_dir, args.num_shadows)
    shadow_models = []

    if missing_shadows:
        logger.info(f"Missing shadow models: {missing_shadows}")
        for i in missing_shadows:
            splits = shadow_data_split(args, i, retain, forget, valid, test_front, logger)
            model = train_shadow_model(args, i, splits, device, logger)
            save_model(model, os.path.join(args.out_dir, f"shadow_{i}.pth"))
            logger.info(f"Shadow {i} trained & saved.")

    shadow_models = [load_shadow_model(args, i, device) for i in range(args.num_shadows)]

    params_path = os.path.join(args.out_dir, 'lira_params.json')
    if not os.path.exists(params_path):
        params, probs = estimate_lira_params(shadow_models, forget_plus_test,
                                             args.batch_size, device, args.out_dir, logger)
    else:
        logger.info("Loading params from json...")
        params = json.load(open(params_path))
        # 重新计算 logits 以获取原始概率用于 Fig 3 绘图
        logger.info("Recalculating logits for Fig 3 plot...")
        _, probs = estimate_lira_params(shadow_models, forget_plus_test,
                                        args.batch_size, device, args.out_dir, logger)

    # === Fig 3 复现调用 ===
    plot_fig3_replication(params, probs, args.num_shadows, args.out_dir, logger)

    # === 新增：绘制 Logit 混合分布直方图 ===
    plot_mixed_logit_hist(probs, args.num_shadows, args.out_dir, logger)

    plot_lira_scatter(params, args.out_dir, logger)
    plot_lira_scatter_separated(params, args.out_dir, logger)
    scores, y_true = lira_target(target_model, forget_plus_test, params,
                                 args.batch_size, device)
    results = plot_roc(scores, y_true, args.out_dir, logger, args.unlearn_method)

    results['unlearn_method'] = args.unlearn_method
    results.update({
        'lamb': args.lamb, 'last_k': args.last_k, 're_init': args.re_init, 'mask_threshold': args.mask_threshold
    })
    result_filename = get_target_model_filename(
        args.unlearn_method,
        lamb=args.lamb,
        mask_threshold=args.mask_threshold,
        last_k=args.last_k,
        re_init=args.re_init
    ).replace('.pth', '_results.json')

    with open(os.path.join(args.out_dir, result_filename), 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"All done! Unlearn method: {args.unlearn_method}")

if __name__ == '__main__':
    main()
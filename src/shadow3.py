#! /usr/bin/env python
import os, json, argparse, numpy as np, torch
import collections
from torch.utils.data import Subset, DataLoader
from copy import deepcopy
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc

# === 绘图依赖设置 ===
import matplotlib
matplotlib.use('Agg') # 强制使用非交互后端，避免服务器报错
import matplotlib.pyplot as plt

# ============================================================
# 假设这些本地依赖依然存在
# ============================================================
from logger import create_logger
from transforms import Transforms
from utils import mkdir
from data_tool import DataStore, DataLoaderTool
from model_bases import DeepModels
from tool import complete_parameters
from unlearners import * 
from models_genun import GeneModUnlearn

# ============================================================
# 辅助类
# ============================================================
class CustomSubset(Subset):
    """解决访问 targets 属性问题的 Subset"""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        if hasattr(dataset, 'targets'):
            all_targets = np.array(dataset.targets)
            self.targets = all_targets[indices].tolist()
        elif hasattr(dataset, 'labels'):
            all_labels = np.array(dataset.labels)
            self.targets = all_labels[indices].tolist()
        else:
            self.targets = None

    def __getattr__(self, name):
        # [修复关键点]：防止访问 'dataset' 属性本身时触发无限递归
        if name == 'dataset':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute 'dataset'")
        
        # 正常转发其他属性（如 transform, classes 等）给内部 dataset
        return getattr(self.dataset, name)

def compute_scaled_logit(model, loader, device, labels):
    model._model.eval()
    logits_list = []
    with torch.no_grad():
        for x, _ in loader:
            out = model._model(x.to(device))
            logits_list.append(out.cpu().numpy())
    
    logits = np.concatenate(logits_list)
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    
    rows = np.arange(len(logits))
    p_true = probs[rows, labels]
    p_true = np.clip(p_true, 1e-7, 1 - 1e-7)
    return np.log(p_true / (1 - p_true))

# === [新增] 通用准确率计算函数 ===
def compute_accuracy(model, loader, device):
    """计算模型在给定Loader上的准确率"""
    model._model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model._model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100.0 * correct / total

def get_model(args, model_name, num_classes, feature_dims, device):
    model_params = complete_parameters(vars(args))
    model_params['num_classes'] = num_classes
    model = DeepModels(args.arch, feature_dims, num_classes, args.log_dir, model_name, False)
    model.parameter_config(**model_params)
    model._model.to(device)
    return model

# ============================================================
# 核心调度器
# ============================================================
class GlobalScheduler:
    def __init__(self, full_dataset, seed=42, target_class=5, pool_size=1000):
        self.full_dataset = full_dataset
        self.rng = np.random.RandomState(seed)
        self.total_len = len(full_dataset)
        
        # 1. 锁定 Pool 1k
        if hasattr(full_dataset, 'targets'):
            labels = np.array(full_dataset.targets)
        else:
            labels = np.array([y for _, y in full_dataset])
        class_5_indices = np.where(labels == target_class)[0]
        self.rng.shuffle(class_5_indices)
        self.pool_1k = class_5_indices[:pool_size]
        self.pool_1k_set = set(self.pool_1k)
        
        self.schedule_map = {} 

    def precompute_schedule(self, num_bases, variants_per_base, start_seed, shadow_start_seed):
        print(">> Pre-computing global schedule to ensure balance across GPUs...")
        for mode, seed_offset in [('target', start_seed), ('shadow', shadow_start_seed)]:
            counter = collections.defaultdict(int)
            for i in range(num_bases):
                base_seed = seed_offset + i
                local_rng = np.random.RandomState(base_seed)
                all_indices = np.arange(self.total_len)
                train_indices = local_rng.choice(all_indices, size=25000, replace=False)
                train_set = set(train_indices)
                
                candidates = list(train_set.intersection(self.pool_1k_set))
                
                for v in range(variants_per_base):
                    np.random.shuffle(candidates)
                    candidates.sort(key=lambda x: counter[x])
                    forget_size = 200
                    selected = candidates[:forget_size]
                    
                    for idx in selected: counter[idx] += 1
                    
                    if len(selected) < forget_size:
                        needed = forget_size - len(selected)
                        others = list(train_set - set(selected))
                        fillers = np.random.choice(others, size=needed, replace=False)
                        selected.extend(fillers)
                    
                    key = f"{mode}_{i}_{v}"
                    self.schedule_map[key] = np.array(selected)
        print(">> Schedule computed. Ready for parallel execution.")

    def get_plan(self, mode, base_id, var_id):
        return self.schedule_map[f"{mode}_{base_id}_{var_id}"]

    def get_base_train_indices(self, mode, base_id, start_seed, shadow_start_seed):
        seed = start_seed if mode == 'target' else shadow_start_seed
        real_seed = seed + base_id
        local_rng = np.random.RandomState(real_seed)
        return local_rng.choice(np.arange(self.total_len), size=25000, replace=False)

# ============================================================
# 结果分析与绘图
# ============================================================
def analyze_and_plot_results(y_true, y_scores, save_dir, method_name, logger):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    total = P + N
    accuracy_list = (tpr * P + (1 - fpr) * N) / total
    best_acc = np.max(accuracy_list)
    best_thresh_idx = np.argmax(accuracy_list)
    best_thresh = thresholds[best_thresh_idx]
    
    logger.info(f"\n====== EXTENDED RESULTS ({method_name}) ======")
    logger.info(f"Total Inference Points: {total}")
    logger.info(f"U-LiRA AUC: {roc_auc:.4f}")
    logger.info(f"Best Attack Accuracy: {best_acc:.4f} (at threshold {best_thresh:.4f})")
    logger.info(f"============================================\n")
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {method_name}\nBest Acc = {best_acc:.4f}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path_linear = os.path.join(save_dir, f"roc_linear_{method_name}.png")
    plt.savefig(save_path_linear, dpi=300)
    plt.close()
    logger.info(f"Saved Linear ROC curve to: {save_path_linear}")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='crimson', lw=2, label=f'{method_name} (AUC={roc_auc:.4f})')
    plt.plot([1e-5, 1], [1e-5, 1], color='navy', lw=2, linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1e-5, 1.0])
    plt.ylim([1e-5, 1.0])
    plt.xlabel('False Positive Rate (Log Scale)')
    plt.ylabel('True Positive Rate (Log Scale)')
    plt.title(f'Log-Scale ROC - {method_name}')
    plt.legend(loc="lower right")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    save_path_log = os.path.join(save_dir, f"roc_log_{method_name}.png")
    plt.savefig(save_path_log, dpi=300)
    plt.close()
    logger.info(f"Saved Log-Scale ROC curve to: {save_path_log}")

# ============================================================
# 训练流水线 (包含 GENM 修复版 和 准确率计算)
# ============================================================
def run_worker(gpu_id, total_gpus, args, scheduler, full_dataset, test_set, device, logger):
    all_base_ids = np.arange(args.num_base_models)
    my_base_ids = np.array_split(all_base_ids, total_gpus)[gpu_id]
    logger.info(f"GPU {gpu_id} assigned tasks: Base Models {my_base_ids[0]} to {my_base_ids[-1]}")
    
    pool_subset = CustomSubset(full_dataset, scheduler.pool_1k)
    pool_loader = DataLoader(pool_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    pool_labels = []
    if hasattr(full_dataset, 'targets'):
        all_labels = np.array(full_dataset.targets)
        pool_labels = all_labels[scheduler.pool_1k]
    else:
        for _, y in pool_subset: pool_labels.append(y)
        pool_labels = np.array(pool_labels)

    for mode in ['target', 'shadow']:
        start_seed = args.seed if mode == 'target' else args.shadow_seed_start
        for i in my_base_ids:
            base_seed = start_seed + i
            base_name = f"{mode}_base_{i}_seed{base_seed}"
            
            # 确保 Base Model 目录存在
            if not os.path.exists(args.base_model_dir):
                try: os.makedirs(args.base_model_dir, exist_ok=True)
                except: pass
                
            base_path = os.path.join(args.base_model_dir, f"{base_name}.pth")
            
            # === Base Model 准备 ===
            feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
            if args.arch == 'resnet18' and args.dataset == 'cifar10': feature_dims = [3, 32, 32]
            base_model = get_model(args, base_name, num_classes, feature_dims, device)
            train_indices = scheduler.get_base_train_indices(mode, i, args.seed, args.shadow_seed_start)
            
            if not os.path.exists(base_path):
                base_model.params['epochs'] = args.train_epochs
                base_model.params['lr'] = args.train_lr
                train_sub = CustomSubset(full_dataset, train_indices)
                builder = BuildLearn(args.log_dir, f"{base_name}_log")
                data_dict = {'train': train_sub, 'valid': train_sub, 'test': test_set}
                base_model = builder.model_unlearn(base_model, data_dict, args.batch_size, device, base_path, is_train=True)
                torch.save(base_model._model.state_dict(), base_path)
            else:
                base_model._model.load_state_dict(torch.load(base_path, map_location=device))
                
            # === Variants (Unlearn: ORG, GENM, SCRUB, etc.) ===
            for v in range(args.variants_per_model):
                var_name = f"{mode}_base_{i}_var_{v}_{args.unlearn_method}"
                
                # 确保 Variants 目录存在
                if not os.path.exists(args.method_model_dir):
                    try: os.makedirs(args.method_model_dir, exist_ok=True)
                    except: pass
                    
                var_path = os.path.join(args.method_model_dir, f"{var_name}.pth")
                res_path = os.path.join(args.method_model_dir, f"{var_name}_res.json")
                
                forget_indices = scheduler.get_plan(mode, i, v)
                retain_indices = np.setdiff1d(train_indices, forget_indices)
                
                if not os.path.exists(res_path):
                    variant_model = deepcopy(base_model)
                    variant_model.params['epochs'] = args.epochs
                    variant_model.params['maxlr'] = args.lr
                    
                    if not os.path.exists(var_path):
                        # 1. ORG 方法: 不做操作
                        if args.unlearn_method == 'ORG':
                            pass 

                        # 2. GENM 方法 (增强健壮性修复版)
                        elif args.unlearn_method == 'GENM':
                            valid_len = 2500
                            retain_train = retain_indices[:-valid_len]
                            retain_valid = retain_indices[-valid_len:]
                            
                            splits = {
                                'train': CustomSubset(full_dataset, train_indices),
                                'forget': CustomSubset(full_dataset, forget_indices),
                                'retain': CustomSubset(full_dataset, retain_train),
                                'valid': CustomSubset(full_dataset, retain_valid),
                                'test': test_set
                            }
                            
                            # 定义初始化权重路径
                            genm_init_path = f"{base_path}ORG_GENE_M_0.pt"

                            # [关键修改]：开启大范围 try-catch，捕获所有初始化错误
                            try:
                                logger.info(f"GENM: Preparing to setup GeneModUnlearn...")

                                # 1. 尝试加载权重 (即使这里失败也不要让进程崩溃，而是记录错误)
                                if os.path.exists(genm_init_path):
                                    logger.info(f"GENM: Loading initialization weights from {genm_init_path}")
                                    # weights_only=False 允许加载全模型对象
                                    checkpoint = torch.load(genm_init_path, map_location=device, weights_only=False)
                                    
                                    # 兼容 dict 和 full object
                                    if isinstance(checkpoint, dict):
                                        if 'state_dict' in checkpoint:
                                            state_dict = checkpoint['state_dict']
                                        elif 'model' in checkpoint:
                                            state_dict = checkpoint['model']
                                        else:
                                            state_dict = checkpoint
                                    elif hasattr(checkpoint, 'state_dict'):
                                        state_dict = checkpoint.state_dict()
                                    else:
                                        state_dict = checkpoint
                                        
                                    # 处理 'module.' 前缀
                                    new_state_dict = {}
                                    for k, val in state_dict.items():
                                        name = k.replace("module.", "")
                                        new_state_dict[name] = val
                                    
                                    # strict=False 加载
                                    missing_keys, unexpected_keys = variant_model._model.load_state_dict(new_state_dict, strict=False)
                                    logger.info(f"GENM: Initialization weights loaded. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                                else:
                                    logger.warning(f"GENM: Init path {genm_init_path} NOT FOUND! Fallback to Base Model.")

                                # 2. 配置 GENM 参数
                                # [修正] model_path 应该指向 GENM 的 init 文件，而不是 base model，以防内部逻辑依赖
                                genun_args = argparse.Namespace(
                                    dataset=args.dataset,
                                    arch=args.arch,
                                    num_samples=len(full_dataset),
                                    valid_size=len(retain_valid),
                                    num_to_forget=len(forget_indices),
                                    forget_classes=None,
                                    batch_size=args.batch_size,
                                    epochs=args.epochs,
                                    lr=args.lr,
                                    weight_decay=5e-5,
                                    optim='adam',
                                    patience=10,
                                    scheduler='CosineAnnealingWarmRestarts',
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
                                    cuda=args.gpu_id,
                                    seed=args.seed,
                                    model_path=genm_init_path if os.path.exists(genm_init_path) else base_path
                                )
                                
                                logger.info("GENM: Initializing GeneModUnlearn class...")
                                # 确保目录存在
                                if not os.path.exists(args.method_model_dir):
                                    os.makedirs(args.method_model_dir, exist_ok=True)

                                mu_er = GeneModUnlearn(args.log_dir, f"{var_name}_log", args.method_model_dir, name=f"{var_name}_genm")
                                dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
                                mu_er.config_transform(dt_mean, dt_std, dt_size)
                                mu_er.set_params(**vars(genun_args))
                                mu_er.args = genun_args
                                
                                logger.info("GENM: Setting data...")
                                mu_er.set_data(splits, num_classes, args.batch_size, 4)
                                
                                # 确保保存路径目录存在
                                os.makedirs(os.path.dirname(var_path), exist_ok=True)
                                
                                logger.info(f"GENM: Starting unlearn loop... Output: {var_path}")
                                ckpt_dir = os.path.dirname(var_path) 

                                unlearned_model_net, _ = mu_er.unlearn(
                                    variant_model._model, None, None,
                                    args.epochs, genun_args.scheduler, genun_args.optim,
                                    genun_args.online_train_aug, genun_args.online_forget_aug,
                                    device, 
                                    ckpt_dir,  # <--- [修复] 这里传入目录 ckpt_dir，而不是文件路径 var_path
                                    model_type=args.arch, over_forget=True,
                                    suffix=''
                                )
                                logger.info("GENM: Unlearn loop finished.")
                                variant_model._model.load_state_dict(unlearned_model_net.state_dict())

                            except Exception as e:
                                # [关键] 打印完整堆栈，以便知道具体哪一行报错
                                logger.error(f"GENM CRITICAL FAILURE in Worker {gpu_id}:")
                                logger.exception(e) 
                                # 即使 GENM 失败，也尝试保存当前的 variant_model (至少不会导致文件缺失，虽然结果可能不对)
                                logger.info("GENM: Saving current model state as fallback to prevent pipeline crash.")
                            
                        # 3. 其他常规遗忘方法
                        else:
                            valid_len = 2500
                            retain_train = retain_indices[:-valid_len]
                            retain_valid = retain_indices[-valid_len:]
                            splits = {
                                'train': CustomSubset(full_dataset, train_indices),
                                'forget': CustomSubset(full_dataset, forget_indices),
                                'retain': CustomSubset(full_dataset, retain_train),
                                'valid': CustomSubset(full_dataset, retain_valid),
                                'test': test_set
                            }
                            
                            unlearner = None
                            kwargs = {}
                            if args.unlearn_method == 'SCRUB': 
                                unlearner = UnlearnScrub(args.log_dir, "unlearn_log")
                            elif args.unlearn_method == 'SALUN': 
                                unlearner = UnlearnSalunRL(args.log_dir, "unlearn_log")
                                kwargs['mask_threshold'] = 0.5
                            elif args.unlearn_method == 'FT': unlearner = UnlearnFinetune(args.log_dir, "unlearn_log")
                            elif args.unlearn_method == 'RT': 
                                unlearner = UnlearnRetrain(args.log_dir, "unlearn_log")
                                kwargs['reinit'] = True
                            elif args.unlearn_method == 'GA': unlearner = UnlearnGradientAscent(args.log_dir, "unlearn_log")
                            elif args.unlearn_method == 'RL': 
                                unlearner = UnlearnRandomLabels(args.log_dir, "unlearn_log")
                                kwargs['num_classes'] = 10
                            elif args.unlearn_method == 'LKL':
                                unlearner = UnlearnLastKLayer(args.log_dir, "unlearn_log")
                                kwargs['last_k'] = 1
                                kwargs['re_init'] = True
                            elif args.unlearn_method == 'UI':
                                unlearner = UnlearnInfluence(args.log_dir, "unlearn_log")
                                kwargs['recursion_depth'] = 1000
                                kwargs['damp'] = 0.01
                                kwargs['scale'] = 25
                                kwargs['r_averaging'] = 3
                                kwargs['last_k'] = 20
                            elif args.unlearn_method == 'GENF':
                                # =======================================================
                                # [修改开始] 针对 GENF：强制验证集仅包含 Class 5
                                # =======================================================
                                
                                # 1. 获取标签 (为了筛选 Class 5)
                                if hasattr(full_dataset, 'targets'):
                                    all_targets_ref = np.array(full_dataset.targets)
                                else:
                                    # 如果没有直接的 targets 属性，需要手动提取（注意性能）
                                    all_targets_ref = np.array([y for _, y in full_dataset])
                                
                                # 2. 找出 retain_indices 中属于 Class 5 的样本
                                # 注意：retain_indices 是全局索引
                                retain_labels = all_targets_ref[retain_indices]
                                is_class_5 = (retain_labels == 5)
                                
                                indices_c5 = retain_indices[is_class_5]      # Retain中的 Class 5
                                indices_others = retain_indices[~is_class_5] # Retain中的 其他类
                                
                                # 3. 随机打乱 Class 5 索引以确保随机性
                                # (如果不打乱，可能取到的总是特定的样本)
                                rng_splitter = np.random.RandomState(args.seed + i + v) # 确定的随机性
                                rng_splitter.shuffle(indices_c5)
                                
                                # 4. 划分验证集 (仅取 Class 5，最多 2500 个)
                                target_valid_size = 2500
                                valid_size = min(len(indices_c5), target_valid_size)
                                
                                retain_valid = indices_c5[:valid_size]        # 验证集 (纯 Class 5)
                                remaining_c5 = indices_c5[valid_size:]        # 剩下的 Class 5 回到训练集
                                
                                # 5. 重组训练用保留集 (Retain Train = 其他类 + 剩下的 Class 5)
                                retain_train = np.concatenate([indices_others, remaining_c5])
                                rng_splitter.shuffle(retain_train) # 再次打乱
                                
                                logger.info(f"GENF Data Split: Valid (Class 5 only): {len(retain_valid)}, Retain Train: {len(retain_train)}")

                                # 6. 构建 Splits 字典
                                splits = {
                                    'train': CustomSubset(full_dataset, train_indices),
                                    'forget': CustomSubset(full_dataset, forget_indices),
                                    'retain': CustomSubset(full_dataset, retain_train),
                                    'valid': CustomSubset(full_dataset, retain_valid),
                                    'test': test_set
                                }
                                unlearner = UnlearnGenF(args.log_dir, "unlearn_log")
                                kwargs['out_dir'] = os.path.abspath(args.out_dir)
                                kwargs['lr'] = args.lr
                            elif args.unlearn_method == 'L1FT':  
                                unlearner = UnlearnSparseL1(args.log_dir, "unlearn_log")
                                kwargs['lamb'] = args.lamb
                            
                            if unlearner:
                                os.makedirs(os.path.dirname(var_path), exist_ok=True)
                                variant_model = unlearner.model_unlearn(
                                    variant_model, splits, args.batch_size, device, var_path, **kwargs
                                )
                        
                        # 最后保存防线
                        if not os.path.exists(os.path.dirname(var_path)):
                            os.makedirs(os.path.dirname(var_path), exist_ok=True)
                        torch.save(variant_model._model.state_dict(), var_path)
                    else:
                        variant_model._model.load_state_dict(torch.load(var_path, map_location=device))
                    
                    scores = compute_scaled_logit(variant_model, pool_loader, device, pool_labels)
                    info = {
                        'base_id': int(i), 'var_id': int(v), 'mode': mode, 'scores': scores.tolist(),
                        'train_set_mask': np.isin(scheduler.pool_1k, train_indices).tolist(),
                        'forget_set_mask': np.isin(scheduler.pool_1k, forget_indices).tolist()
                    }
                    with open(res_path, 'w') as f: json.dump(info, f)

                    # ============================================================
                    # === [定制评估] 针对 v=0 变体的详细准确率分析 ===
                    # ============================================================
                    if v == 0:
                        logger.info(f"[{mode.upper()} Base {i}] Calculating specialized metrics for Variant 0...")
                        
                        # --- 0. 准备基础随机数生成器 ---
                        rng_eval = np.random.RandomState(args.seed + i + 9999) 
                        _loader_args = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 4}

                        # --- 1. 获取全量数据的标签 (用于筛选 Class 5) ---
                        if hasattr(full_dataset, 'targets'):
                            all_targets = np.array(full_dataset.targets)
                        else:
                            # 如果没有 targets 属性，手动遍历一遍（慢但稳）
                            all_targets = np.array([y for _, y in full_dataset])
                        
                        # --- 2. 筛选出“未见过的(Unseen) 第5类样本” ---
                        # A. 找到所有属于第5类的索引
                        all_class_5_indices = np.where(all_targets == 5)[0]
                        
                        # B. 找到所有未参与该模型训练的索引 (Full - TrainIndices)
                        # train_indices 是当前模型用过的，我们要排除它
                        all_indices = np.arange(len(full_dataset))
                        unseen_indices = np.setdiff1d(all_indices, train_indices)
                        
                        # C. 取交集：既是第5类，又没训练过
                        candidate_unseen_c5 = np.intersect1d(all_class_5_indices, unseen_indices)
                        
                        # D. 随机采样 200 个
                        if len(candidate_unseen_c5) >= 200:
                            indices_unseen_c5 = rng_eval.choice(candidate_unseen_c5, size=200, replace=False)
                        else:
                            logger.warning(f"Base {i}: Not enough unseen class 5 samples ({len(candidate_unseen_c5)}<200). Using all available.")
                            indices_unseen_c5 = candidate_unseen_c5

                        # --- 3. (可选) 对比组：采样 200 个遗忘样本 (也是 Class 5) ---
                        # 你的遗忘集本身就是从 Class 5 选出来的，直接用 forget_indices 即可
                        indices_forget = forget_indices 

                        # --- 4. 构建 Dataset 和 Loader ---
                        subset_unseen_c5 = CustomSubset(full_dataset, indices_unseen_c5)
                        subset_forget    = CustomSubset(full_dataset, indices_forget)
                        
                        loader_unseen_c5 = DataLoader(subset_unseen_c5, **_loader_args)
                        loader_forget    = DataLoader(subset_forget, **_loader_args)
                        
                        # --- 5. 计算准确率 ---
                        acc_unseen_c5 = compute_accuracy(variant_model, loader_unseen_c5, device)
                        acc_forget    = compute_accuracy(variant_model, loader_forget, device)
                        
                        # --- 6. 打印结果 ---
                        log_msg = (
                            f"\n>>> Class 5 Analysis [Mode: {mode} | Base: {i}]\n"
                            f"    Forget Set Acc (Class 5, Seen then Forgotten): {acc_forget:.2f}%\n"
                            f"    Unseen Set Acc (Class 5, Never Seen)         : {acc_unseen_c5:.2f}%\n"
                            f"------------------------------------------------------------"
                        )
                        logger.info(log_msg)
                        
                        # (可选) 保存到 JSON
                        # extra_res_path = os.path.join(args.method_model_dir, f"{var_name}_c5_acc.json")
                        # with open(extra_res_path, 'w') as f:
                        #     json.dump({'acc_forget': acc_forget, 'acc_unseen_c5': acc_unseen_c5}, f)

                    del variant_model
            del base_model
            torch.cuda.empty_cache()
            if (i - my_base_ids[0] + 1) % 5 == 0:
                logger.info(f"[GPU {gpu_id}] Progress: {i}/{my_base_ids[-1]}")

# ============================================================
# 评估逻辑 (Evaluation Aggregator)
# ============================================================
# ============================================================
# 评估逻辑 (Evaluation Aggregator)
# ============================================================
def run_evaluation(args, scheduler, logger):
    logger.info("Gathering results from all workers...")
    
    # === Step 1: 收集 Shadow Data (这部分保持不变，我们需要所有数据来计算分布) ===
    in_dists = [[] for _ in range(1000)]
    out_dists = [[] for _ in range(1000)]
    
    for i in range(args.num_base_models):
        for v in range(args.variants_per_model):
            fname = f"shadow_base_{i}_var_{v}_{args.unlearn_method}_res.json"
            fpath = os.path.join(args.method_model_dir, fname)
            
            if not os.path.exists(fpath):
                # logger.warning(f"Missing file: {fpath}") # 可选：减少日志噪音
                continue
                
            with open(fpath, 'r') as f: res = json.load(f)
            scores = res['scores']
            is_train = res['train_set_mask']
            is_forget = res['forget_set_mask']
            
            for j in range(1000):
                if is_forget[j]: in_dists[j].append(scores[j])
                elif not is_train[j]: out_dists[j].append(scores[j])

    # === Step 2: 评估 Target Models (此处修改核心逻辑) ===
    y_true_all, y_scores_all = [], []
    nan_points_count = 0
    
    for i in range(args.num_base_models):
        for v in range(args.variants_per_model):
            fname = f"target_base_{i}_var_{v}_{args.unlearn_method}_res.json"
            fpath = os.path.join(args.method_model_dir, fname)
            
            if not os.path.exists(fpath): continue
            
            with open(fpath, 'r') as f: res = json.load(f)
            scores = res['scores']
            is_train = res['train_set_mask']
            is_forget = res['forget_set_mask']
            
            # --- [修改开始]：先分类索引，再下采样负样本 ---
            
            # 1. 找出所有的正样本(Forget)和负样本(Non-Member)的索引
            pos_indices = [] # Label = 1
            neg_indices = [] # Label = 0
            
            for j in range(1000):
                if is_forget[j]:
                    pos_indices.append(j)
                elif not is_train[j]:
                    neg_indices.append(j)
            
            # 2. 下采样负样本，使其数量等于正样本 (1:1 平衡)
            # 正样本通常是 200 个，负样本通常是 ~500 个
            target_count = len(pos_indices) 
            
            if len(neg_indices) > target_count:
                # 使用与模型ID绑定的随机种子，保证可复现性，但每个模型抽样不同
                rng_local = np.random.RandomState(args.seed + i + 10086)
                neg_indices = rng_local.choice(neg_indices, size=target_count, replace=False).tolist()
            
            # 3. 合并需要计算的索引列表
            indices_to_evaluate = pos_indices + neg_indices
            
            # 4. 只遍历选中的索引
            for j in indices_to_evaluate:
                # 确定标签
                if j in pos_indices:
                    label = 1
                else:
                    label = 0
                
                # --- 以下计算逻辑保持不变 ---
                in_vals, out_vals = in_dists[j], out_dists[j]
                
                in_vals_arr = np.array(in_vals)
                out_vals_arr = np.array(out_vals)
                in_vals_clean = in_vals_arr[np.isfinite(in_vals_arr)]
                out_vals_clean = out_vals_arr[np.isfinite(out_vals_arr)]
                
                if len(in_vals_clean) < 2 or len(out_vals_clean) < 2: continue

                mu_in, std_in = np.mean(in_vals_clean), np.std(in_vals_clean) + 1e-10
                mu_out, std_out = np.mean(out_vals_clean), np.std(out_vals_clean) + 1e-10
                
                if not np.isfinite(scores[j]): continue
                
                l_in = norm.logpdf(scores[j], mu_in, std_in)
                l_out = norm.logpdf(scores[j], mu_out, std_out)
                final_score = l_in - l_out
                
                if np.isnan(final_score) or np.isinf(final_score):
                    nan_points_count += 1
                    continue
                
                y_true_all.append(label)
                y_scores_all.append(final_score)
            
            # --- [修改结束] ---

    logger.info(f"Skipped {nan_points_count} NaN/Inf points due to numerical instability.")
    if len(y_true_all) == 0:
        logger.error("No valid data points. Check model training!")
        return

    # 现在的 y_true_all 中，0和1的比例应该是 1:1，Accuracy 约为 0.5 才是真正的随机猜测
    analyze_and_plot_results(y_true_all, y_scores_all, args.out_dir, args.unlearn_method, logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--total_gpus', type=int, default=1)
    parser.add_argument('--log_dir', default='.')
    parser.add_argument('--out_dir', default='./ulira_dist_outs/')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--num_base_models', type=int, default=128)
    parser.add_argument('--variants_per_model', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shadow_seed_start', type=int, default=2000)
    
    # unlearn_method 可以是 FT, SCRUB, ORG, GENM 等
    parser.add_argument('--unlearn_method', default='FT')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--train_lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lamb', type=float, default=1e-4) 
    
    args = parser.parse_args()
    
    mkdir(args.out_dir)
    args.base_model_dir = os.path.join(args.out_dir, "common_bases")
    mkdir(args.base_model_dir)
    args.method_model_dir = os.path.join(args.out_dir, args.unlearn_method)
    mkdir(args.method_model_dir)
    
    logger = create_logger(args.log_dir, f'worker_{args.gpu_id}')
    
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    processor = Transforms(dt_mean, dt_std, dt_size)
    full_train_dataset, official_test_dataset = DataLoaderTool.load_dataset(
        args.dataset, 
        processor.get_transform('normal', 10), 
        processor.get_transform('test', 10), 
        num_samples=-1
    )
    
    scheduler = GlobalScheduler(full_train_dataset, seed=args.seed)
    scheduler.precompute_schedule(args.num_base_models, args.variants_per_model, args.seed, args.shadow_seed_start)
    
    if args.mode == 'train':
        device = torch.device(f'cuda:{args.gpu_id}')
        run_worker(args.gpu_id, args.total_gpus, args, scheduler, full_train_dataset, official_test_dataset, device, logger)
    elif args.mode == 'eval':
        run_evaluation(args, scheduler, logger)

if __name__ == '__main__':
    main()
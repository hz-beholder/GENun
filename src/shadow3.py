#! /usr/bin/env python
import os, json, argparse, numpy as np, torch
import collections
from torch.utils.data import Subset, DataLoader
from copy import deepcopy
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc

# ============================================================
# 假设这些本地依赖依然存在
# ============================================================
from logger import create_logger
from transforms import Transforms
from utils import mkdir
from data_tool import DataStore, DataLoaderTool
from model_bases import DeepModels
from tool import complete_parameters
from unlearners import * # 包含 BuildLearn, UnlearnSCRUB 等

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
    p_true = np.clip(p_true, 1e-10, 1 - 1e-10)
    return np.log(p_true / (1 - p_true))

def get_model(args, model_name, num_classes, feature_dims, device):
    model_params = complete_parameters(vars(args))
    model_params['num_classes'] = num_classes
    model = DeepModels(args.arch, feature_dims, num_classes, args.log_dir, model_name, False)
    model.parameter_config(**model_params)
    model._model.to(device)
    return model

# ============================================================
# 核心调度器：支持预计算
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
        
        # 调度表：key=f"{mode}_{base_id}_{var_id}", value=forget_indices
        self.schedule_map = {} 

    def precompute_schedule(self, num_bases, variants_per_base, start_seed, shadow_start_seed):
        """
        在开始训练前，预先计算所有 GPU 都会用到的全局调度表。
        这保证了即使在不同 GPU 上运行，大家对'哪个模型该遗忘哪些样本'达成共识。
        """
        print(">> Pre-computing global schedule to ensure balance across GPUs...")
        
        for mode, seed_offset in [('target', start_seed), ('shadow', shadow_start_seed)]:
            # 为每个 mode 维护独立的计数器
            counter = collections.defaultdict(int)
            
            for i in range(num_bases):
                base_seed = seed_offset + i
                # 复现 Base Model 的训练集划分
                local_rng = np.random.RandomState(base_seed)
                all_indices = np.arange(self.total_len)
                train_indices = local_rng.choice(all_indices, size=25000, replace=False)
                train_set = set(train_indices)
                
                # 找出候选者
                candidates = list(train_set.intersection(self.pool_1k_set))
                
                for v in range(variants_per_base):
                    # --- 计数器均衡核心逻辑 ---
                    # 1. 打乱候选者（避免顺序偏差）
                    np.random.shuffle(candidates)
                    # 2. 按当前计数排序（优先选出现次数少的）
                    candidates.sort(key=lambda x: counter[x])
                    
                    # 3. 选取前 200 个
                    forget_size = 200
                    selected = candidates[:forget_size]
                    
                    # 4. 更新计数器
                    for idx in selected:
                        counter[idx] += 1
                    
                    # 5. 如果不够，随机填充 (罕见情况)
                    if len(selected) < forget_size:
                        needed = forget_size - len(selected)
                        others = list(train_set - set(selected))
                        fillers = np.random.choice(others, size=needed, replace=False)
                        selected.extend(fillers)
                    
                    # 6. 存入调度表
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
# 训练流水线
# ============================================================
def run_worker(gpu_id, total_gpus, args, scheduler, full_dataset, test_set, device, logger):
    """
    单个 GPU Worker 的工作流
    """
    # 1. 确定当前 GPU 负责的任务范围
    # 将 128 个 Base Model 均匀分配给 8 个 GPU
    all_base_ids = np.arange(args.num_base_models)
    my_base_ids = np.array_split(all_base_ids, total_gpus)[gpu_id]
    
    logger.info(f"GPU {gpu_id} assigned tasks: Base Models {my_base_ids[0]} to {my_base_ids[-1]}")
    
    # 准备 Pool 1k 数据加载器 (用于计算 score)
    pool_subset = CustomSubset(full_dataset, scheduler.pool_1k)
    pool_loader = DataLoader(pool_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 获取标签
    pool_labels = []
    if hasattr(full_dataset, 'targets'):
        all_labels = np.array(full_dataset.targets)
        pool_labels = all_labels[scheduler.pool_1k]
    else:
        for _, y in pool_subset: pool_labels.append(y)
        pool_labels = np.array(pool_labels)

    # 遍历 Target 和 Shadow 两种模式
    for mode in ['target', 'shadow']:
        start_seed = args.seed if mode == 'target' else args.shadow_seed_start
        
        for i in my_base_ids:
            base_seed = start_seed + i
            base_name = f"{mode}_base_{i}_seed{base_seed}"
            base_path = os.path.join(args.base_model_dir, f"{base_name}.pth")
            
            # === 1. Base Model (Train or Load) ===
            feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
            if args.arch == 'resnet18' and args.dataset == 'cifar10': feature_dims = [3, 32, 32]
            base_model = get_model(args, base_name, num_classes, feature_dims, device)
            
            # 获取训练集索引
            train_indices = scheduler.get_base_train_indices(mode, i, args.seed, args.shadow_seed_start)
            
            # 文件锁机制 (简单实现)：如果文件不存在，且没有 .lock 文件，则训练
            # 在多GPU环境下，Base Model 不会冲突，因为我们按 ID 切分了任务
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
                
            # === 2. Variants (Unlearn) ===
            for v in range(args.variants_per_model):
                var_name = f"{mode}_base_{i}_var_{v}_{args.unlearn_method}"
                var_path = os.path.join(args.method_model_dir, f"{var_name}.pth")
                res_path = os.path.join(args.method_model_dir, f"{var_name}_res.json")
                
                # 获取预计算的 Forget Indices
                forget_indices = scheduler.get_plan(mode, i, v)
                retain_indices = np.setdiff1d(train_indices, forget_indices)
                
                if not os.path.exists(res_path): # 只有结果不存在才跑
                    variant_model = deepcopy(base_model)
                    variant_model.params['epochs'] = args.epochs
                    variant_model.params['maxlr'] = args.lr
                    
                    if not os.path.exists(var_path):
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
                        
                        # 初始化 Unlearner
                        if args.unlearn_method == 'SCRUB': unlearner = UnlearnSCRUB(args.log_dir, "unlearn_log")
                        elif args.unlearn_method == 'FT': unlearner = UnlearnFinetune(args.log_dir, "unlearn_log")
                        elif args.unlearn_method == 'NegGrad+': unlearner = UnlearnNegGradPlus(args.log_dir, "unlearn_log")
                        elif args.unlearn_method == 'L1FT': unlearner = UnlearnSparseL1(args.log_dir, "unlearn_log")
                        
                        kwargs = {}
                        if args.unlearn_method == 'L1FT': kwargs['lamb'] = args.lamb
                        
                        variant_model = unlearner.model_unlearn(
                            variant_model, splits, args.batch_size, device, var_path, **kwargs
                        )
                        torch.save(variant_model._model.state_dict(), var_path)
                    else:
                        variant_model._model.load_state_dict(torch.load(var_path, map_location=device))
                    
                    # 计算并保存分数
                    scores = compute_scaled_logit(variant_model, pool_loader, device, pool_labels)
                    
                    # 构造结果字典
                    info = {
                        'base_id': int(i),
                        'var_id': int(v),
                        'mode': mode,
                        'scores': scores.tolist(),
                        'train_set_mask': np.isin(scheduler.pool_1k, train_indices).tolist(),
                        'forget_set_mask': np.isin(scheduler.pool_1k, forget_indices).tolist()
                    }
                    
                    with open(res_path, 'w') as f:
                        json.dump(info, f)
                    
                    del variant_model
            
            del base_model
            torch.cuda.empty_cache()
            
            if (i - my_base_ids[0] + 1) % 5 == 0:
                logger.info(f"[GPU {gpu_id}] Progress: {i}/{my_base_ids[-1]}")

# ============================================================
# 评估逻辑 (Evaluation Aggregator)
# ============================================================
def run_evaluation(args, scheduler, logger):
    logger.info("Gathering results from all workers...")
    
    in_dists = [[] for _ in range(1000)]
    out_dists = [[] for _ in range(1000)]
    
    # 1. 收集 Shadow Data
    for i in range(args.num_base_models):
        for v in range(args.variants_per_model):
            fname = f"shadow_base_{i}_var_{v}_{args.unlearn_method}_res.json"
            fpath = os.path.join(args.method_model_dir, fname)
            
            if not os.path.exists(fpath):
                logger.warning(f"Missing file: {fpath}")
                continue
                
            with open(fpath, 'r') as f:
                res = json.load(f)
                
            scores = res['scores']
            is_train = res['train_set_mask']
            is_forget = res['forget_set_mask']
            
            for j in range(1000):
                if is_forget[j]:
                    in_dists[j].append(scores[j])
                elif not is_train[j]:
                    out_dists[j].append(scores[j])

    # 2. 评估 Target Models
    y_true_all, y_scores_all = [], []
    
    for i in range(args.num_base_models):
        for v in range(args.variants_per_model):
            fname = f"target_base_{i}_var_{v}_{args.unlearn_method}_res.json"
            fpath = os.path.join(args.method_model_dir, fname)
            
            if not os.path.exists(fpath): continue
            
            with open(fpath, 'r') as f:
                res = json.load(f)
            
            scores = res['scores']
            is_train = res['train_set_mask']
            is_forget = res['forget_set_mask']
            
            for j in range(1000):
                label = -1
                if is_forget[j]: label = 1
                elif not is_train[j]: label = 0
                
                if label != -1:
                    in_vals, out_vals = in_dists[j], out_dists[j]
                    if len(in_vals) < 2 or len(out_vals) < 2: continue
                    
                    mu_in, std_in = np.mean(in_vals), np.std(in_vals) + 1e-10
                    mu_out, std_out = np.mean(out_vals), np.std(out_vals) + 1e-10
                    
                    l_in = norm.logpdf(scores[j], mu_in, std_in)
                    l_out = norm.logpdf(scores[j], mu_out, std_out)
                    
                    y_true_all.append(label)
                    y_scores_all.append(l_in - l_out)

    # 3. 计算 AUC
    if len(y_true_all) == 0:
        logger.error("No valid data points for evaluation.")
        return

    fpr, tpr, _ = roc_curve(y_true_all, y_scores_all)
    roc_auc = auc(fpr, tpr)
    logger.info(f"\n====== FINAL RESULTS ({args.unlearn_method}) ======")
    logger.info(f"Total Inference Points: {len(y_true_all)}")
    logger.info(f"U-LiRA AUC: {roc_auc:.4f}")
    logger.info(f"============================================\n")


def main():
    parser = argparse.ArgumentParser()
    # 基础配置
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--gpu_id', type=int, default=0, help="Current GPU ID (0-7)")
    parser.add_argument('--total_gpus', type=int, default=1, help="Total number of GPUs")
    
    parser.add_argument('--log_dir', default='.')
    parser.add_argument('--out_dir', default='./ulira_dist_outs/')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--num_base_models', type=int, default=128)
    parser.add_argument('--variants_per_model', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shadow_seed_start', type=int, default=2000)
    
    parser.add_argument('--unlearn_method', default='FT')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--train_lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--lamb', type=float, default=1e-6) 
    
    args = parser.parse_args()
    
    # 目录初始化
    mkdir(args.out_dir)
    args.base_model_dir = os.path.join(args.out_dir, "common_bases")
    mkdir(args.base_model_dir)
    args.method_model_dir = os.path.join(args.out_dir, args.unlearn_method)
    mkdir(args.method_model_dir)
    
    logger = create_logger(args.log_dir, f'worker_{args.gpu_id}')
    
    # 加载数据
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    processor = Transforms(dt_mean, dt_std, dt_size)
    full_train_dataset, official_test_dataset = DataLoaderTool.load_dataset(
        args.dataset, 
        processor.get_transform('normal', 10), 
        processor.get_transform('test', 10), 
        num_samples=-1
    )
    
    # 初始化调度器并预计算
    scheduler = GlobalScheduler(full_train_dataset, seed=args.seed)
    # 即使是 Evaluate 模式，我们也需要 scheduler 来确定 pool_1k
    scheduler.precompute_schedule(args.num_base_models, args.variants_per_model, args.seed, args.shadow_seed_start)
    
    if args.mode == 'train':
        device = torch.device(f'cuda:{args.gpu_id}')
        run_worker(args.gpu_id, args.total_gpus, args, scheduler, full_train_dataset, official_test_dataset, device, logger)
    elif args.mode == 'eval':
        run_evaluation(args, scheduler, logger)

if __name__ == '__main__':
    main()
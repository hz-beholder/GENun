#! /usr/bin/env python
import os, json, argparse, numpy as np, torch
import matplotlib.pyplot as plt  # [新增] 用于绘图
from torch.utils.data import Subset, ConcatDataset, DataLoader
from copy import deepcopy
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc

from logger import create_logger
from transforms import Transforms
from utils import argparse2bool, seed_everything, mkdir
from data_tool import DataStore, DataLoaderTool
from model_bases import DeepModels
from tool import complete_parameters, PathGenerator as pather
from unlearners import *

class CustomSubset(Subset):
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

def parse_args():
    parser = argparse.ArgumentParser(description="Strict U-LiRA with Statistic Transfer")
    parser.add_argument('--logname', default='ulira_strict')
    parser.add_argument('--log_dir', default='.')
    parser.add_argument('--out_dir', default='./ulira_strict_outs/')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--num_shadows', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shadow_seed_start', type=int, default=1000)
    parser.add_argument('--unlearn_method', default='FT', choices=['SCRUB', 'FT', 'NegGrad+', 'ORG', 'L1FT'])
    parser.add_argument('--cuda', default='0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=2) 
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--train_transform', default='normal') 
    parser.add_argument('--test_transform', default='test')
    parser.add_argument('--lamb', type=float, default=1e-6) 
    parser.add_argument('--mask_threshold', type=float, default=0.5) 
    return parser.parse_args()

class ModelDataSplitter:
    def __init__(self, full_dataset, seed, fixed_forget_indices=None):
        self.full_dataset = full_dataset
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.fixed_forget_indices = fixed_forget_indices 
        if hasattr(full_dataset, 'targets'):
            self.labels = np.array(full_dataset.targets)
        else:
            self.labels = np.array([y for _, y in full_dataset])
        self._split()

    def _split(self):
        total_len = len(self.full_dataset) 
        all_indices = np.arange(total_len)
        self.train_pool_indices = self.rng.choice(all_indices, size=25000, replace=False)
        self.unseen_pool_indices = np.setdiff1d(all_indices, self.train_pool_indices)
        pool_labels = self.labels[self.train_pool_indices]
        class_5_local = np.where(pool_labels == 5)[0]
        class_5_global = self.train_pool_indices[class_5_local]
        forget_size = 200
        
        if self.fixed_forget_indices is not None:
            overlap = np.intersect1d(class_5_global, self.fixed_forget_indices)
            current_forget_indices = list(overlap)
            if len(current_forget_indices) < forget_size:
                remainder = np.setdiff1d(class_5_global, overlap)
                self.rng.shuffle(remainder)
                needed = forget_size - len(current_forget_indices)
                current_forget_indices.extend(remainder[:needed])
            self.forget_indices = np.array(current_forget_indices)
        else:
            self.rng.shuffle(class_5_global)
            self.forget_indices = class_5_global[:forget_size]

        self.retain_indices = np.setdiff1d(self.train_pool_indices, self.forget_indices)
        unseen_labels = self.labels[self.unseen_pool_indices]
        unseen_cls5_local = np.where(unseen_labels == 5)[0]
        unseen_cls5_global = self.unseen_pool_indices[unseen_cls5_local]
        self.rng.shuffle(unseen_cls5_global)
        self.test_indices = unseen_cls5_global[:forget_size] 

    def get_splits(self):
        valid_len = 2500
        retain_train_indices = self.retain_indices[:-valid_len]
        valid_indices = self.retain_indices[-valid_len:]
        train_indices = np.concatenate([retain_train_indices, valid_indices, self.forget_indices])
        return {
            'train': CustomSubset(self.full_dataset, train_indices), 
            'forget': CustomSubset(self.full_dataset, self.forget_indices), 
            'retain': CustomSubset(self.full_dataset, retain_train_indices), 
            'valid': CustomSubset(self.full_dataset, valid_indices), 
            'lira_non_member': CustomSubset(self.full_dataset, self.test_indices), 
            'indices': {
                'train_pool': self.train_pool_indices, 
                'forget': self.forget_indices 
            }
        }

def run_model_lifecycle(args, model_name, splitter, official_test_set, device, logger):
    splits = splitter.get_splits()
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    if args.arch == 'resnet18' and args.dataset == 'cifar10': 
        feature_dims = [3, 32, 32]
    model_params = complete_parameters(vars(args))
    model_params['num_classes'] = num_classes
    model = DeepModels(args.arch, feature_dims, num_classes, args.log_dir, model_name, False)
    model.parameter_config(**model_params)
    model._model.to(device)
    
    org_path = os.path.join(args.out_dir, f"{model_name}_org.pth")
    if not os.path.exists(org_path):
        model.params['epochs'] = args.train_epochs
        model.params['lr'] = args.train_lr 
        model.params['maxlr'] = args.train_lr
        builder = BuildLearn(args.log_dir, f"{model_name}_log")
        data_dict = {'train': splits['train'], 'valid': splits['valid'], 'test': official_test_set}
        model = builder.model_unlearn(model, data_dict, args.batch_size, device, org_path, is_train=True)
        torch.save(model._model.state_dict(), org_path)
    else:
        model._model.load_state_dict(torch.load(org_path, map_location=device))

    eval_sets = {'Forget': splits['forget'], 'Retain': splits['retain'], 'Test': official_test_set}
    acc_org = eval_accuracy(model, eval_sets, args.batch_size, device)
    logger.info(f"[{model_name}] [ORIGINAL] Acc: Forget={acc_org['Forget']:.4f}, Retain={acc_org['Retain']:.4f}, Test={acc_org['Test']:.4f}")

    if args.unlearn_method == 'ORG': return model, splits
        
    unlearn_path = os.path.join(args.out_dir, f"{model_name}_{args.unlearn_method}.pth")
    unlearn_model = deepcopy(model)
    unlearn_model.params['epochs'] = args.epochs
    unlearn_model.params['lr'] = args.lr 
    unlearn_model.params['maxlr'] = args.lr 
    
    if not os.path.exists(unlearn_path):
        if args.unlearn_method == 'SCRUB': unlearner = UnlearnSCRUB(args.log_dir, "unlearn_log")
        elif args.unlearn_method == 'FT': unlearner = UnlearnFinetune(args.log_dir, "unlearn_log")
        elif args.unlearn_method == 'NegGrad+': unlearner = UnlearnNegGradPlus(args.log_dir, "unlearn_log")
        elif args.unlearn_method == 'L1FT': unlearner = UnlearnSparseL1(args.log_dir, "unlearn_log")
        else: raise ValueError(f"Method {args.unlearn_method} not hooked up.")
        data_dict = {'train': splits['train'], 'forget': splits['forget'], 'retain': splits['retain'], 'valid': splits['valid'], 'test': official_test_set}
        kwargs = {}
        if args.unlearn_method == 'L1FT': kwargs['lamb'] = args.lamb
        kwargs['lr'] = args.lr
        unlearn_model = unlearner.model_unlearn(unlearn_model, data_dict, args.batch_size, device, unlearn_path, **kwargs)
        torch.save(unlearn_model._model.state_dict(), unlearn_path)
    else:
        unlearn_model._model.load_state_dict(torch.load(unlearn_path, map_location=device))
        
    acc_un = eval_accuracy(unlearn_model, eval_sets, args.batch_size, device)
    logger.info(f"[{model_name}] [POST-{args.unlearn_method}] Acc: Forget={acc_un['Forget']:.4f}, Retain={acc_un['Retain']:.4f}, Test={acc_un['Test']:.4f}")
    return unlearn_model, splits

@torch.no_grad()
def eval_accuracy(model, data_dict, batch_size, device):
    model._model.eval()
    results = {}
    for name, dataset in data_dict.items():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model._model(x)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        results[name] = correct / total if total > 0 else 0.0
    return results

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

def main():
    args = parse_args()
    mkdir(args.out_dir)
    logger = create_logger(args.log_dir, 'ulira_log')
    logger.info(f"Strict U-LiRA | Train: {args.train_epochs}ep@{args.train_lr} | Unlearn: {args.epochs}ep@{args.lr}")
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    os.environ['DO_NOT_USE_HEAVY_WORKERS'] = '1'
    torch.set_num_threads(4) 
    
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    processor = Transforms(dt_mean, dt_std, dt_size)
    full_train_dataset, official_test_dataset = DataLoaderTool.load_dataset(
        args.dataset, processor.get_transform(args.train_transform, 10), 
        processor.get_transform(args.test_transform, 10), num_samples=-1)
    
    logger.info("=== 1. Target Model Lifecycle ===")
    target_splitter = ModelDataSplitter(full_train_dataset, seed=args.seed)
    target_model, target_splits = run_model_lifecycle(args, "target", target_splitter, official_test_dataset, device, logger)
    target_forget_indices = target_splits['indices']['forget']
    
    logger.info(f"=== 2. Shadow Models Lifecycle (N={args.num_shadows}) ===")
    eval_dataset = ConcatDataset([target_splits['forget'], target_splits['lira_non_member']])
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    eval_labels = np.array([y for _, y in eval_dataset])
    
    # 评估集索引：前200是正样本(Forget)，后200是负样本(Test)
    eval_global_indices = np.concatenate([target_splits['indices']['forget'], target_splits['lira_non_member'].indices])
    # 标记正负样本：True=正样本, False=负样本
    is_positive_sample = np.concatenate([np.ones(len(target_splits['forget']), dtype=bool), 
                                         np.zeros(len(target_splits['lira_non_member']), dtype=bool)])
    
    shadow_scores_matrix = []
    shadow_metadata = []
    
    for i in range(args.num_shadows):
        s_splitter = ModelDataSplitter(full_train_dataset, seed=args.shadow_seed_start + i, fixed_forget_indices=target_forget_indices)
        s_name = f"shadow_{i}"
        score_cache_path = os.path.join(args.out_dir, f"{s_name}_{args.unlearn_method}_scores.npy")
        idx_cache_path = os.path.join(args.out_dir, f"{s_name}_idx.json")
        
        if os.path.exists(score_cache_path) and os.path.exists(idx_cache_path):
            scores = np.load(score_cache_path)
            with open(idx_cache_path, 'r') as f: indices_info = json.load(f)
            if (i+1) % 50 == 0: logger.info(f"Shadow {i} loaded from cache.")
        else:
            s_model, s_splits = run_model_lifecycle(args, s_name, s_splitter, official_test_dataset, device, logger)
            scores = compute_scaled_logit(s_model, eval_loader, device, eval_labels)
            indices_info = {'train_pool': s_splits['indices']['train_pool'].tolist(), 'forget': s_splits['indices']['forget'].tolist()}
            np.save(score_cache_path, scores)
            with open(idx_cache_path, 'w') as f: json.dump(indices_info, f)
            del s_model
            torch.cuda.empty_cache()
            
        shadow_scores_matrix.append(scores)
        shadow_metadata.append(indices_info)
        if (i+1) % 50 == 0: logger.info(f"Processed Shadow {i+1}/{args.num_shadows}")

    shadow_scores_matrix = np.array(shadow_scores_matrix) 

    logger.info("=== 3. Computing U-LiRA Scores (with Statistic Transfer) ===")
    target_scores = compute_scaled_logit(target_model, eval_loader, device, eval_labels)
    lira_final_scores = []
    
    # --- 阶段1: 从正样本中学习 IN 分布的统计特征 ---
    diffs = [] # 记录 mu_in - mu_out
    std_ins = [] # 记录 sigma_in
    
    for j in range(len(eval_global_indices)):
        if not is_positive_sample[j]: continue # 只看正样本
        
        global_idx = eval_global_indices[j]
        in_scores_dist = []
        out_scores_dist = []
        
        for k in range(args.num_shadows):
            score_kj = shadow_scores_matrix[k, j]
            meta = shadow_metadata[k]
            if global_idx in meta['forget']: in_scores_dist.append(score_kj)
            elif global_idx not in meta['train_pool']: out_scores_dist.append(score_kj)
            
        if len(in_scores_dist) >= 2 and len(out_scores_dist) >= 2:
            mu_in, std_in = np.mean(in_scores_dist), np.std(in_scores_dist) + 1e-10
            mu_out = np.mean(out_scores_dist)
            diffs.append(mu_in - mu_out)
            std_ins.append(std_in)
    
    global_diff_mean = np.mean(diffs) if diffs else 0.0
    global_std_in_mean = np.mean(std_ins) if std_ins else 1.0
    logger.info(f"Statistic Transfer: Avg Gap (IN-OUT)={global_diff_mean:.4f}, Avg Std IN={global_std_in_mean:.4f}")

    # --- 阶段2: 计算所有样本的 LiRA Score (对负样本应用 Transfer) ---
    for j in range(len(eval_global_indices)):
        global_idx = eval_global_indices[j]
        in_scores_dist = []
        out_scores_dist = []
        
        # 收集数据
        for k in range(args.num_shadows):
            score_kj = shadow_scores_matrix[k, j]
            meta = shadow_metadata[k]
            if global_idx in meta['forget']: in_scores_dist.append(score_kj)
            elif global_idx not in meta['train_pool']: out_scores_dist.append(score_kj)
        
        # 确定参数
        if is_positive_sample[j]:
            # 正样本：有足够的 IN 模型，直接计算
            if len(in_scores_dist) < 2 or len(out_scores_dist) < 2:
                lira_final_scores.append(0)
                continue
            mu_in, std_in = np.mean(in_scores_dist), np.std(in_scores_dist) + 1e-10
            mu_out, std_out = np.mean(out_scores_dist), np.std(out_scores_dist) + 1e-10
        else:
            # 负样本：IN 模型不足，使用 Transfer Statistics
            # 我们有充足的 OUT 模型，计算 mu_out
            if len(out_scores_dist) < 2:
                lira_final_scores.append(0)
                continue
            mu_out, std_out = np.mean(out_scores_dist), np.std(out_scores_dist) + 1e-10
            
            # 构造虚拟 IN 分布
            mu_in = mu_out + global_diff_mean
            std_in = global_std_in_mean # 或者混合策略：(std_out + global_std_in_mean) / 2
            
        # 计算 Score
        l_in = norm.logpdf(target_scores[j], mu_in, std_in)
        l_out = norm.logpdf(target_scores[j], mu_out, std_out)
        l_final = l_in - l_out
        # 处理可能的无效值
        if np.isnan(l_final) or np.isinf(l_final):
             l_final = 0
        lira_final_scores.append(l_final)

    y_true = np.concatenate([np.ones(len(target_splits['forget'])), np.zeros(len(target_splits['lira_non_member']))])
    valid_mask = [i for i, s in enumerate(lira_final_scores) if s != 0]
    
    if len(valid_mask) == 0:
        logger.warning("No valid scores found for ROC calculation.")
        return

    y_true_valid = y_true[valid_mask]
    y_scores_valid = np.array(lira_final_scores)[valid_mask]
    
    # [新增] 获取阈值以计算最佳Acc
    fpr, tpr, thresholds = roc_curve(y_true_valid, y_scores_valid)
    roc_auc = auc(fpr, tpr)
    
    # [新增] 计算 Best Accuracy (基于平衡样本的假设，或者使用加权计算)
    # Acc = (TPR * P + (1-FPR) * N) / (P + N)
    P_count = np.sum(y_true_valid == 1)
    N_count = np.sum(y_true_valid == 0)
    
    if P_count + N_count > 0:
        accuracies = (tpr * P_count + (1 - fpr) * N_count) / (P_count + N_count)
        best_acc = np.max(accuracies)
    else:
        best_acc = 0.0
    
    logger.info(f"\n================ RESULTS ================")
    logger.info(f"Method: {args.unlearn_method} | U-LiRA AUC: {roc_auc:.4f} | Best Acc: {best_acc:.4f}")
    logger.info(f"=========================================\n")
    
    results = {'method': args.unlearn_method, 'auc': roc_auc, 'best_acc': best_acc, 'scores': y_scores_valid.tolist(), 'labels': y_true_valid.tolist()}
    with open(os.path.join(args.out_dir, f"res_{args.unlearn_method}.json"), 'w') as f: json.dump(results, f)

    # [新增] 绘制并保存 ROC 曲线
    plt.switch_backend('Agg') # 确保在无显示器的服务器上也能运行
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {args.unlearn_method} (Best Acc={best_acc:.4f})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    roc_path = os.path.join(args.out_dir, f"roc_{args.unlearn_method}.png")
    plt.savefig(roc_path, dpi=100)
    plt.close()
    logger.info(f"ROC curve saved to {roc_path}")
    # [新增] 绘制 Log-Scale ROC 曲线 (用于观察低 FPR 下的攻击效果)
    plt.figure(figsize=(8, 6))
    
    # 绘制曲线
    plt.plot(fpr, tpr, color='crimson', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    
    # 绘制对角线 (随机猜测线)，注意起点不能是0，因为log(0)无定义，设为一个很小的数比如 1e-5
    plt.plot([1e-6, 1], [1e-6, 1], color='navy', lw=2, linestyle='--')

    # --- 关键设置开始 ---
    plt.xscale('log') # X轴设为对数
    plt.yscale('log') # Y轴设为对数
    
    # 设置显示范围
    # 通常 LiRA 攻击关注 10^-5 到 1 的范围
    plt.xlim([1e-5, 1.0]) 
    plt.ylim([1e-5, 1.0])
    
    # 设置网格：which='both' 会同时显示主刻度(0.1)和次刻度(0.02, 0.03...)
    plt.grid(True, which='both', ls='-', alpha=0.2)
    # --- 关键设置结束 ---

    plt.xlabel('False Positive Rate (Log Scale)')
    plt.ylabel('True Positive Rate (Log Scale)')
    plt.title(f'Log-Scale ROC - {args.unlearn_method} (Best Acc={best_acc:.4f})')
    plt.legend(loc="lower right")
    
    # 保存对数版本的图片
    log_roc_path = os.path.join(args.out_dir, f"roc_log_{args.unlearn_method}.png")
    plt.savefig(log_roc_path, dpi=100)
    plt.close()
    logger.info(f"Log-Scale ROC curve saved to {log_roc_path}")

if __name__ == '__main__':
    main()
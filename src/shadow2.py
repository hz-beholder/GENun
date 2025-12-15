#! /usr/bin/env python
import os, json, argparse, numpy as np, torch
from torch.utils.data import Subset, ConcatDataset, DataLoader
from copy import deepcopy
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc

# 导入本地依赖
from logger import create_logger
from transforms import Transforms
from utils import argparse2bool, seed_everything, mkdir
from data_tool import DataStore, DataLoaderTool
from model_bases import DeepModels
from tool import complete_parameters, PathGenerator as pather
from unlearners import *

# ================= 修复 AttributeError 的自定义类 =================
class CustomSubset(Subset):
    """
    一个带属性透传的 Subset 类。
    解决了 model_bases.py 访问 dataset.targets 报错的问题。
    """
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
# =============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Strict U-LiRA with Full Evaluation")
    parser.add_argument('--logname', default='ulira_strict')
    parser.add_argument('--log_dir', default='.')
    parser.add_argument('--out_dir', default='./ulira_strict_outs/')
    
    # 论文实验核心设置
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--num_shadows', type=int, default=256, help="Paper uses 256 shadows")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shadow_seed_start', type=int, default=1000)
    
    # 遗忘方法配置
    parser.add_argument('--unlearn_method', default='FT', choices=['SCRUB', 'FT', 'NegGrad+', 'ORG', 'L1FT'])
    parser.add_argument('--cuda', default='0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10) 
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    
    parser.add_argument('--train_transform', default='normal') 
    parser.add_argument('--test_transform', default='test')
    
    parser.add_argument('--lamb', type=float, default=1e-6) 
    parser.add_argument('--mask_threshold', type=float, default=0.5) 
    
    return parser.parse_args()

# ==============================================================================
# 1. 数据划分逻辑
# ==============================================================================
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
        valid_len = 1000
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

# ==============================================================================
# 2. 模型生命周期
# ==============================================================================
def run_model_lifecycle(args, model_name, splitter, official_test_set, device, logger):
    splits = splitter.get_splits()
    
    # A. 训练/加载
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
        builder = BuildLearn(args.log_dir, f"{model_name}_log")
        data_dict = {
            'train': splits['train'], 
            'valid': splits['valid'], 
            'test': official_test_set 
        }
        model = builder.model_unlearn(model, data_dict, args.batch_size, device, org_path, is_train=True)
        torch.save(model._model.state_dict(), org_path)
    else:
        model._model.load_state_dict(torch.load(org_path, map_location=device))

    # B. 遗忘
    if args.unlearn_method == 'ORG':
        return model, splits
        
    unlearn_path = os.path.join(args.out_dir, f"{model_name}_{args.unlearn_method}.pth")
    unlearn_model = deepcopy(model)
    
    if not os.path.exists(unlearn_path):
        if args.unlearn_method == 'SCRUB':
            unlearner = UnlearnSCRUB(args.log_dir, "unlearn_log")
        elif args.unlearn_method == 'FT':
            unlearner = UnlearnFinetune(args.log_dir, "unlearn_log")
        elif args.unlearn_method == 'NegGrad+':
            unlearner = UnlearnNegGradPlus(args.log_dir, "unlearn_log")
        elif args.unlearn_method == 'L1FT':
            unlearner = UnlearnSparseL1(args.log_dir, "unlearn_log")
        else:
            raise ValueError(f"Method {args.unlearn_method} not hooked up.")
            
        data_dict = {
            'train': splits['train'],
            'forget': splits['forget'],
            'retain': splits['retain'],
            'valid': splits['valid'],
            'test': official_test_set 
        }
        
        kwargs = {}
        if args.unlearn_method == 'L1FT': kwargs['lamb'] = args.lamb
        
        unlearn_model = unlearner.model_unlearn(
            unlearn_model, data_dict, args.batch_size, device, unlearn_path, **kwargs
        )
        torch.save(unlearn_model._model.state_dict(), unlearn_path)
    else:
        unlearn_model._model.load_state_dict(torch.load(unlearn_path, map_location=device))
        
    return unlearn_model, splits

# ==============================================================================
# 3. 工具函数
# ==============================================================================
@torch.no_grad()
def eval_accuracy(model, data_dict, batch_size, device):
    """
    计算并在控制台打印各个数据集上的准确率。
    """
    model._model.eval()
    results = {}
    
    for name, dataset in data_dict.items():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        correct = 0
        total = 0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model._model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        acc = correct / total if total > 0 else 0.0
        results[name] = acc
        
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

# ==============================================================================
# 4. 主程序
# ==============================================================================
def main():
    args = parse_args()
    mkdir(args.out_dir)
    logger = create_logger(args.log_dir, 'ulira_log')
    logger.info(f"Running Strict U-LiRA Reproduction with {args.num_shadows} shadows.")
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    os.environ['DO_NOT_USE_HEAVY_WORKERS'] = '1'
    torch.set_num_threads(4) 
    
    # 1. 加载数据
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    processor = Transforms(dt_mean, dt_std, dt_size)
    
    full_train_dataset, official_test_dataset = DataLoaderTool.load_dataset(
        args.dataset, 
        processor.get_transform(args.train_transform, 10), 
        processor.get_transform(args.test_transform, 10), 
        num_samples=-1
    )
    
    logger.info(f"Loaded: Train Universe ({len(full_train_dataset)}), Official Test ({len(official_test_dataset)})")
    
    # --------------------------------------------------------------------------
    # 2. 目标模型 (Target Model)
    # --------------------------------------------------------------------------
    logger.info("=== 1. Target Model Lifecycle ===")
    target_splitter = ModelDataSplitter(full_train_dataset, seed=args.seed)
    
    target_model, target_splits = run_model_lifecycle(
        args, "target", target_splitter, official_test_dataset, device, logger
    )
    
    # === 目标模型评估 ===
    logger.info(f"Evaluating Target Model (Unlearn Method: {args.unlearn_method})...")
    eval_sets = {
        'Forget Set': target_splits['forget'],
        'Retain Set': target_splits['retain'],
        'Official Test': official_test_dataset
    }
    accs = eval_accuracy(target_model, eval_sets, args.batch_size, device)
    
    logger.info(f"Target Model Stats: Forget Acc={accs['Forget Set']:.4f}, "
                f"Retain Acc={accs['Retain Set']:.4f}, Test Acc={accs['Official Test']:.4f}")
    
    target_forget_indices = target_splits['indices']['forget']
    
    # --------------------------------------------------------------------------
    # 3. 影子模型 (Shadow Models)
    # --------------------------------------------------------------------------
    logger.info(f"=== 2. Shadow Models Lifecycle (N={args.num_shadows}) ===")
    
    eval_dataset = ConcatDataset([target_splits['forget'], target_splits['lira_non_member']])
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    eval_labels = []
    for _, y in eval_dataset: eval_labels.append(y)
    eval_labels = np.array(eval_labels)
    
    eval_global_indices = np.concatenate([
        target_splits['indices']['forget'],
        target_splits['lira_non_member'].indices 
    ])
    
    shadow_scores_matrix = []
    shadow_metadata = []
    
    for i in range(args.num_shadows):
        s_splitter = ModelDataSplitter(
            full_train_dataset, 
            seed=args.shadow_seed_start + i,
            fixed_forget_indices=target_forget_indices 
        )
        
        s_name = f"shadow_{i}"
        score_cache_path = os.path.join(args.out_dir, f"{s_name}_scores.npy")
        idx_cache_path = os.path.join(args.out_dir, f"{s_name}_idx.json")
        
        if os.path.exists(score_cache_path) and os.path.exists(idx_cache_path):
            scores = np.load(score_cache_path)
            with open(idx_cache_path, 'r') as f:
                indices_info = json.load(f)
        else:
            s_model, s_splits = run_model_lifecycle(
                args, s_name, s_splitter, official_test_dataset, device, logger
            )
            
            # === 影子模型评估 (抽样评估，避免刷屏) ===
            # 只对第 0 个和每 50 个影子模型输出评估信息，确保影子模型的行为正常
            if i == 0 or (i + 1) % 50 == 0:
                s_eval_sets = {
                    'Forget Set': s_splits['forget'],
                    'Retain Set': s_splits['retain'],
                    'Official Test': official_test_dataset
                }
                s_accs = eval_accuracy(s_model, s_eval_sets, args.batch_size, device)
                logger.info(f"[Shadow {i}] Stats: Forget Acc={s_accs['Forget Set']:.4f}, "
                            f"Retain Acc={s_accs['Retain Set']:.4f}, Test Acc={s_accs['Official Test']:.4f}")
            
            scores = compute_scaled_logit(s_model, eval_loader, device, eval_labels)
            
            indices_info = {
                'train_pool': s_splits['indices']['train_pool'].tolist(),
                'forget': s_splits['indices']['forget'].tolist()
            }
            
            np.save(score_cache_path, scores)
            with open(idx_cache_path, 'w') as f:
                json.dump(indices_info, f)
            del s_model
            torch.cuda.empty_cache()
            
        shadow_scores_matrix.append(scores)
        shadow_metadata.append(indices_info)
        
        if (i+1) % 10 == 0: logger.info(f"Processed Shadow {i+1}/{args.num_shadows}")

    shadow_scores_matrix = np.array(shadow_scores_matrix) 

    # --------------------------------------------------------------------------
    # 4. U-LiRA 攻击计算
    # --------------------------------------------------------------------------
    logger.info("=== 3. Computing U-LiRA Scores ===")
    
    target_scores = compute_scaled_logit(target_model, eval_loader, device, eval_labels)
    lira_final_scores = []
    
    for j in range(len(eval_global_indices)):
        global_idx = eval_global_indices[j]
        in_scores_dist = []
        out_scores_dist = []
        
        for k in range(args.num_shadows):
            score_kj = shadow_scores_matrix[k, j]
            meta = shadow_metadata[k]
            
            if global_idx in meta['forget']:
                in_scores_dist.append(score_kj)
            elif global_idx not in meta['train_pool']:
                out_scores_dist.append(score_kj)
        
        if len(in_scores_dist) < 2 or len(out_scores_dist) < 2:
            lira_final_scores.append(0)
            continue
            
        mu_in, std_in = np.mean(in_scores_dist), np.std(in_scores_dist) + 1e-10
        mu_out, std_out = np.mean(out_scores_dist), np.std(out_scores_dist) + 1e-10
        
        l_in = norm.logpdf(target_scores[j], mu_in, std_in)
        l_out = norm.logpdf(target_scores[j], mu_out, std_out)
        lira_final_scores.append(l_in - l_out)

    # --------------------------------------------------------------------------
    # 5. 结果评估 (AUC)
    # --------------------------------------------------------------------------
    y_true = np.concatenate([
        np.ones(len(target_splits['forget'])), 
        np.zeros(len(target_splits['lira_non_member']))
    ])
    
    valid_mask = [i for i, s in enumerate(lira_final_scores) if s != 0]
    y_true = y_true[valid_mask]
    y_scores = np.array(lira_final_scores)[valid_mask]
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    logger.info(f"\n================ RESULTS ================")
    logger.info(f"Method: {args.unlearn_method}")
    logger.info(f"U-LiRA AUC: {roc_auc:.4f}")
    logger.info(f"=========================================\n")
    
    results = {
        'method': args.unlearn_method,
        'auc': roc_auc,
        'scores': y_scores.tolist(),
        'labels': y_true.tolist()
    }
    with open(os.path.join(args.out_dir, f"res_{args.unlearn_method}.json"), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
#! /usr/bin/env python
import os, json, argparse, numpy as np, torch
import collections
import math
from torch.utils.data import Subset, DataLoader
from copy import deepcopy
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc

# === 绘图依赖设置 ===
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ============================================================
# 假设本地依赖 (保持你的文件结构)
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
# 辅助工具类
# ============================================================
class CustomSubset(Subset):
    """带 targets 属性的 Subset，修复 PyTorch Subset 丢失 labels 的问题"""
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
    """计算 LiRA 需要的 Scaled Logit"""
    model._model.eval()
    logits_list = []
    with torch.no_grad():
        for x, _ in loader:
            out = model._model(x.to(device))
            logits_list.append(out.cpu().numpy())
    
    if len(logits_list) == 0:
        return np.array([])

    logits = np.concatenate(logits_list)
    # Logit Scaling Trick
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    
    rows = np.arange(len(logits))
    p_true = probs[rows, labels]
    p_true = np.clip(p_true, 1e-10, 1 - 1e-10) # 避免 log(0)
    return np.log(p_true / (1 - p_true))

def get_model(args, model_name, num_classes, feature_dims, device):
    """获取模型实例"""
    model_params = complete_parameters(vars(args))
    model_params['num_classes'] = num_classes
    model = DeepModels(args.arch, feature_dims, num_classes, args.log_dir, model_name, False)
    model.parameter_config(**model_params)
    model._model.to(device)
    return model

# ============================================================
# 全局调度器：负责分配谁该遗忘什么
# ============================================================
class GlobalScheduler:
    def __init__(self, full_dataset, seed=42, target_class=5):
        self.full_dataset = full_dataset
        self.total_len = len(full_dataset)
        self.seed = seed
        
        # 1. 锁定所有第 5 类样本
        if hasattr(full_dataset, 'targets'):
            labels = np.array(full_dataset.targets)
        else:
            labels = np.array([y for _, y in full_dataset])
        self.class_5_indices = np.where(labels == target_class)[0]
        print(f">> [Scheduler] Found {len(self.class_5_indices)} samples in Class {target_class}.")
        
        self.schedule_map = {} 

    def get_base_train_indices(self, mode, base_id, start_seed, shadow_start_seed):
        """
        [关键] 复现 Base 模型的训练集索引
        """
        seed_offset = start_seed if mode == 'target' else shadow_start_seed
        real_seed = seed_offset + base_id
        local_rng = np.random.RandomState(real_seed)
        # 假设 Base 训练集大小为 25000 (CIFAR-10 的一半)
        return local_rng.choice(np.arange(self.total_len), size=25000, replace=False)

    def precompute_schedule(self, num_bases, variants_per_base, start_seed, shadow_start_seed, forget_size=200):
        print(">> Pre-computing global schedule for FULL Class 5 coverage...")
        
        for mode, seed_offset in [('target', start_seed), ('shadow', shadow_start_seed)]:
            for i in range(num_bases):
                # 1. 获取该 Base 的训练集
                train_indices = self.get_base_train_indices(mode, i, start_seed, shadow_start_seed)
                
                # 2. 找出该模型训练集中包含的 Class 5 样本 (Candidates for unlearning)
                candidates = np.intersect1d(train_indices, self.class_5_indices)
                
                # 3. 为每个变体分配 200 个样本
                sched_rng = np.random.RandomState(seed_offset + i + 9999) # 独立的随机流
                
                for v in range(variants_per_base):
                    # 随机抽取 200 个
                    if len(candidates) >= forget_size:
                        chosen = sched_rng.choice(candidates, size=forget_size, replace=False)
                    else:
                        chosen = candidates # 样本不足全选
                    
                    key = f"{mode}_{i}_{v}"
                    self.schedule_map[key] = chosen

        print(f">> Schedule computed. ({len(self.schedule_map)} plans)")

    def get_plan(self, mode, base_id, var_id):
        return self.schedule_map[f"{mode}_{base_id}_{var_id}"]

# ============================================================
# Worker：训练与遗忘执行者
# ============================================================
def run_worker(gpu_id, total_gpus, args, scheduler, full_dataset, test_set, device, logger):
    # 分配任务
    all_base_ids = np.arange(args.num_base_models)
    my_base_ids = np.array_split(all_base_ids, total_gpus)[gpu_id]
    logger.info(f"GPU {gpu_id} assigned Base Models: {my_base_ids[0]} to {my_base_ids[-1]}")

    for mode in ['target', 'shadow']:
        start_seed = args.seed if mode == 'target' else args.shadow_seed_start
        
        for i in my_base_ids:
            # === 1. Base Model 准备 ===
            base_seed = start_seed + i
            base_name = f"{mode}_base_{i}_seed{base_seed}"
            base_path = os.path.join(args.base_model_dir, f"{base_name}.pth")
            
            feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
            if args.arch == 'resnet18' and args.dataset == 'cifar10': feature_dims = [3, 32, 32]
            base_model = get_model(args, base_name, num_classes, feature_dims, device)
            
            # Base Model 训练 (如果不存在)
            train_indices = scheduler.get_base_train_indices(mode, i, args.seed, args.shadow_seed_start)
            
            if not os.path.exists(base_path):
                # logger.info(f"Training Base: {base_name}")
                base_model.params['epochs'] = args.train_epochs
                base_model.params['lr'] = args.train_lr
                train_sub = CustomSubset(full_dataset, train_indices)
                builder = BuildLearn(args.log_dir, f"{base_name}_log")
                data_dict = {'train': train_sub, 'valid': train_sub, 'test': test_set}
                base_model = builder.model_unlearn(base_model, data_dict, args.batch_size, device, base_path, is_train=True)
                torch.save(base_model._model.state_dict(), base_path)
            else:
                base_model._model.load_state_dict(torch.load(base_path, map_location=device))
            
            # === 2. 变体生成 (40 个) ===
            for v in range(args.variants_per_model):
                var_name = f"{mode}_base_{i}_var_{v}_{args.unlearn_method}"
                var_path = os.path.join(args.method_model_dir, f"{var_name}.pth")
                res_path = os.path.join(args.method_model_dir, f"{var_name}_res.json")
                
                # 如果结果已存在，跳过
                if os.path.exists(res_path): continue

                # 获取遗忘计划 (200个 IN 样本)
                forget_indices = scheduler.get_plan(mode, i, v)
                retain_indices = np.setdiff1d(train_indices, forget_indices)
                
                # 克隆模型
                variant_model = deepcopy(base_model)
                variant_model.params['epochs'] = args.epochs
                variant_model.params['maxlr'] = args.lr
                
                # --- Unlearning 执行 ---
                if not os.path.exists(var_path):
                    # 构建数据分割
                    splits = {
                        'train': CustomSubset(full_dataset, train_indices),
                        'forget': CustomSubset(full_dataset, forget_indices),
                        'retain': CustomSubset(full_dataset, retain_indices[:-2500]), # 简单留出验证集
                        'valid': CustomSubset(full_dataset, retain_indices[-2500:]),
                        'test': test_set
                    }

                    # GENM 特殊逻辑 (带修复)
                    if args.unlearn_method == 'GENM':
                        genm_init_path = f"{base_path}ORG_GENE_M_0.pt"
                        try:
                            # logger.info(f"Running GENM for {var_name}")
                            mu_er = GeneModUnlearn(args.log_dir, f"{var_name}_log", args.method_model_dir, name=f"{var_name}_genm")
                            dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
                            mu_er.config_transform(dt_mean, dt_std, dt_size)
                            
                            genun_args = argparse.Namespace(
                                dataset=args.dataset, arch=args.arch, num_samples=len(full_dataset),
                                valid_size=len(splits['valid']), num_to_forget=len(forget_indices),
                                batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
                                weight_decay=5e-5, optim='adam', scheduler='CosineAnnealingWarmRestarts',
                                online_train_aug='none', online_forget_aug='none', over_forget=True,
                                regularizer='l1', gamma=1e-5, alpha=1.0, no_reg_epochs=0,
                                dynamic_weight=False, class_wise=True, sample_ratio=1.0,
                                save_checkpoint=False, num_workers=4, cuda=args.gpu_id, seed=args.seed,
                                model_path=genm_init_path if os.path.exists(genm_init_path) else base_path
                            )
                            mu_er.set_params(**vars(genun_args))
                            mu_er.set_data(splits, num_classes, args.batch_size, 4)
                            unlearned_net, _ = mu_er.unlearn(variant_model._model, None, None, 
                                                             args.epochs, 'CosineAnnealingWarmRestarts', 'adam',
                                                             'none', 'none', device, var_path, 
                                                             model_type=args.arch, over_forget=True)
                            variant_model._model.load_state_dict(unlearned_net.state_dict())
                        except Exception as e:
                            logger.error(f"GENM FAILED on {var_name}: {e}")
                    else:
                        # 通用方法 (FT, SCRUB, etc.)
                        unlearner = None
                        kwargs = {}
                        if args.unlearn_method == 'FT': unlearner = UnlearnFinetune(args.log_dir, "ul")
                        elif args.unlearn_method == 'RT': unlearner = UnlearnRetrain(args.log_dir, "ul")
                        elif args.unlearn_method == 'SCRUB': unlearner = UnlearnSCRUB(args.log_dir, "ul")
                        elif args.unlearn_method == 'L1FT': 
                            unlearner = UnlearnSparseL1(args.log_dir, "ul")
                            kwargs['lamb'] = args.lamb
                        
                        if unlearner:
                            variant_model = unlearner.model_unlearn(variant_model, splits, args.batch_size, device, var_path, **kwargs)
                    
                    torch.save(variant_model._model.state_dict(), var_path)
                else:
                    variant_model._model.load_state_dict(torch.load(var_path, map_location=device))
                
                # --- 评估 (生成 IN 和 OUT 分数) ---
                
                # 1. 计算 IN 分数 (遗忘样本)
                # 这些是本变体被要求遗忘的样本
                in_loader = DataLoader(CustomSubset(full_dataset, forget_indices), 
                                      batch_size=args.batch_size, shuffle=False, num_workers=2)
                
                in_labels = np.array(full_dataset.targets)[forget_indices] if hasattr(full_dataset, 'targets') \
                            else np.array([y for _, y in CustomSubset(full_dataset, forget_indices)])
                
                in_scores = compute_scaled_logit(variant_model, in_loader, device, in_labels)

                # 2. 计算 OUT 分数 (未见过的 Class 5 样本)
                # 逻辑：找出 Class 5 中，既不在 Base 训练集，也不在 forget set (显然) 的样本
                all_class_5 = scheduler.class_5_indices
                # Base模型没见过的
                out_candidates = np.setdiff1d(all_class_5, train_indices)
                
                # 随机采样 200 个作为 OUT 测试点
                rng_out = np.random.RandomState(int(base_seed + v + 2024))
                if len(out_candidates) > 200:
                    out_indices = rng_out.choice(out_candidates, size=200, replace=False)
                else:
                    out_indices = out_candidates
                
                out_loader = DataLoader(CustomSubset(full_dataset, out_indices), 
                                       batch_size=args.batch_size, shuffle=False, num_workers=2)
                
                out_labels = np.array(full_dataset.targets)[out_indices] if hasattr(full_dataset, 'targets') \
                             else np.array([y for _, y in CustomSubset(full_dataset, out_indices)])
                
                out_scores = compute_scaled_logit(variant_model, out_loader, device, out_labels)

                # --- 保存结果 ---
                info = {
                    'base_id': int(i),
                    'var_id': int(v),
                    'mode': mode,
                    # IN Data
                    'in_indices': forget_indices.tolist(),
                    'in_scores': in_scores.tolist(),
                    # OUT Data
                    'out_indices': out_indices.tolist(),
                    'out_scores': out_scores.tolist()
                }
                
                with open(res_path, 'w') as f: json.dump(info, f)
                
                del variant_model

            # 清理显存
            del base_model
            torch.cuda.empty_cache()
            
            if (i - my_base_ids[0] + 1) % 1 == 0:
                logger.info(f"[GPU {gpu_id}] Processed Base {i} ({args.variants_per_model} variants)")

# ============================================================
# Evaluator：数据聚合与攻击模拟 (含 Log-Scale 绘图功能)
# ============================================================
def run_evaluation(args, scheduler, logger):
    logger.info(">>> Starting Aggregation Phase...")
    
    shadow_dist = collections.defaultdict(lambda: {'in': [], 'out': []})
    
    # === 第一步：收集 Shadow 数据 ===
    logger.info("1. Gathering SHADOW distributions (128 Bases x 40 Variants)...")
    shadow_files_count = 0
    
    for i in range(args.num_base_models):
        for v in range(args.variants_per_model):
            fname = f"shadow_base_{i}_var_{v}_{args.unlearn_method}_res.json"
            fpath = os.path.join(args.method_model_dir, fname)
            
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'r') as f: res = json.load(f)
                    for idx, sc in zip(res['in_indices'], res['in_scores']):
                        if np.isfinite(sc): shadow_dist[idx]['in'].append(sc)
                    for idx, sc in zip(res['out_indices'], res['out_scores']):
                        if np.isfinite(sc): shadow_dist[idx]['out'].append(sc)
                    shadow_files_count += 1
                except Exception as e:
                    logger.warning(f"Error reading {fname}: {e}")
        
        if (i + 1) % 10 == 0:
            logger.info(f"   Processed Shadow Bases: {i + 1}/{args.num_base_models}")
    
    logger.info(f"   Parsed {shadow_files_count} shadow files.")

    # === 第二步：攻击 Target 模型 ===
    logger.info("2. Attacking TARGET models...")
    
    y_true_all = []   
    y_scores_all = [] 
    skipped_count = 0
    
    for i in range(args.num_base_models):
        for v in range(args.variants_per_model):
            fname = f"target_base_{i}_var_{v}_{args.unlearn_method}_res.json"
            fpath = os.path.join(args.method_model_dir, fname)
            
            if not os.path.exists(fpath): continue
            
            try:
                with open(fpath, 'r') as f: res = json.load(f)
                
                # IN Samples
                for idx, target_score in zip(res['in_indices'], res['in_scores']):
                    if not np.isfinite(target_score): continue
                    s_in_dist = shadow_dist[idx]['in']
                    s_out_dist = shadow_dist[idx]['out']
                    
                    if len(s_in_dist) < 128 or len(s_out_dist) < 128:
                        skipped_count += 1
                        continue
                    
                    mu_in, std_in = np.mean(s_in_dist), np.std(s_in_dist) + 1e-10
                    mu_out, std_out = np.mean(s_out_dist), np.std(s_out_dist) + 1e-10
                    score_in = norm.logpdf(target_score, mu_in, std_in)
                    score_out = norm.logpdf(target_score, mu_out, std_out)
                    
                    y_true_all.append(1) 
                    y_scores_all.append(score_in - score_out)
                
                # OUT Samples
                for idx, target_score in zip(res['out_indices'], res['out_scores']):
                    if not np.isfinite(target_score): continue
                    s_in_dist = shadow_dist[idx]['in']
                    s_out_dist = shadow_dist[idx]['out']
                    
                    if len(s_in_dist) < 128 or len(s_out_dist) < 128:
                        continue
                        
                    mu_in, std_in = np.mean(s_in_dist), np.std(s_in_dist) + 1e-10
                    mu_out, std_out = np.mean(s_out_dist), np.std(s_out_dist) + 1e-10
                    score_in = norm.logpdf(target_score, mu_in, std_in)
                    score_out = norm.logpdf(target_score, mu_out, std_out)
                    
                    y_true_all.append(0) 
                    y_scores_all.append(score_in - score_out)
            
            except Exception as e:
                logger.warning(f"Error processing target {fname}: {e}")

        if (i + 1) % 10 == 0:
            logger.info(f"   Processed Target Bases: {i + 1}/{args.num_base_models}")

    logger.info(f"3. Evaluation Complete. Total Attacks: {len(y_true_all)}")

    if len(y_true_all) == 0:
        logger.error("No valid attack points found!")
        return

    # === 第三步：计算与绘图 ===
    logger.info("4. Calculating Stats & Plotting...")
    
    y_true_np = np.array(y_true_all)
    y_scores_np = np.array(y_scores_all)

    # 计算 ROC
    fpr, tpr, thresholds = roc_curve(y_true_np, y_scores_np)
    roc_auc = auc(fpr, tpr)
    
    # 快速计算最佳 Accuracy
    P = np.sum(y_true_np == 1)
    N = np.sum(y_true_np == 0)
    total = P + N
    accuracy_list = (tpr * P + (1 - fpr) * N) / total
    best_acc = np.max(accuracy_list)
    best_thresh = thresholds[np.argmax(accuracy_list)]
    
    logger.info(f"\n====== FINAL RESULTS ({args.unlearn_method}) ======")
    logger.info(f"AUC: {roc_auc:.4f}")
    logger.info(f"Best Acc: {best_acc:.4f} (at thr {best_thresh:.4f})")
    logger.info(f"================================================\n")
    
    try:
        # --- 1. 绘制线性尺度 ROC (Linear Scale) ---
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'U-LiRA ROC (Linear) - {args.unlearn_method}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        save_path_linear = os.path.join(args.out_dir, f"ROC_Linear_{args.unlearn_method}.png")
        plt.savefig(save_path_linear, dpi=300)
        plt.close()
        logger.info(f"Saved Linear ROC to: {save_path_linear}")

        # --- 2. 绘制对数尺度 ROC (Log Scale) [新增功能] ---
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='crimson', lw=2, label=f'AUC = {roc_auc:.4f}')
        
        # 设置对数坐标
        plt.xscale('log')
        plt.yscale('log')
        
        # 限制范围 (防止 Log(0) 报错，通常关注 10^-5 到 1 的范围)
        plt.xlim([1e-5, 1.0])
        plt.ylim([1e-5, 1.0])
        
        # 绘制对角线 (随机猜测线)
        plt.plot([1e-5, 1], [1e-5, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlabel('False Positive Rate (Log Scale)')
        plt.ylabel('True Positive Rate (Log Scale)')
        plt.title(f'U-LiRA ROC (Log Scale) - {args.unlearn_method}')
        plt.legend(loc="lower right")
        plt.grid(True, which="both", ls="-", alpha=0.2) # 同时显示主刻度和次刻度网格
        
        save_path_log = os.path.join(args.out_dir, f"ROC_Log_{args.unlearn_method}.png")
        plt.savefig(save_path_log, dpi=300)
        plt.close()
        logger.info(f"Saved Log-Scale ROC to: {save_path_log}")

    except Exception as e:
        logger.error(f"Plotting failed: {e}")

# ============================================================
# Main 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--total_gpus', type=int, default=1)
    parser.add_argument('--log_dir', default='.')
    parser.add_argument('--out_dir', default='./ulira_128_40_outs/')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--arch', default='resnet18')
    # 配置：128 Base, 40 Variants
    parser.add_argument('--num_base_models', type=int, default=128) 
    parser.add_argument('--variants_per_model', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shadow_seed_start', type=int, default=2000)
    
    parser.add_argument('--unlearn_method', default='FT')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_epochs', type=int, default=30) # Base epochs
    parser.add_argument('--train_lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2) # Unlearn epochs
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--lamb', type=float, default=1e-6)
    
    args = parser.parse_args()
    
    mkdir(args.out_dir)
    args.base_model_dir = os.path.join(args.out_dir, "common_bases")
    mkdir(args.base_model_dir)
    args.method_model_dir = os.path.join(args.out_dir, args.unlearn_method)
    mkdir(args.method_model_dir)
    
    logger = create_logger(args.log_dir, f'worker_{args.gpu_id}' if args.mode=='train' else 'evaluator')
    
    # 加载数据
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    processor = Transforms(dt_mean, dt_std, dt_size)
    full_train_dataset, official_test_dataset = DataLoaderTool.load_dataset(
        args.dataset, 
        processor.get_transform('normal', 10), 
        processor.get_transform('test', 10), 
        num_samples=-1
    )
    
    # 调度器初始化 (Force forget_size=200)
    scheduler = GlobalScheduler(full_train_dataset, seed=args.seed)
    scheduler.precompute_schedule(args.num_base_models, args.variants_per_model, args.seed, args.shadow_seed_start, forget_size=200)
    
    if args.mode == 'train':
        device = torch.device(f'cuda:{args.gpu_id}')
        run_worker(args.gpu_id, args.total_gpus, args, scheduler, full_train_dataset, official_test_dataset, device, logger)
    elif args.mode == 'eval':
        run_evaluation(args, scheduler, logger)

if __name__ == '__main__':
    main()
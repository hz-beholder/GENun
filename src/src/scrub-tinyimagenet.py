#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCRUB 遗忘脚本 – tiny-imagenet 专用最小改动版
仅替换数据集/模型/路径相关参数，其余与原脚本保持一致。
"""
import argparse
from copy import deepcopy
import os
import sys
import time
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 假设原脚本同目录下已有 logger / const / utils / learner / model_deep / transforms / data_tool 等模块
from logger import create_logger
from const import SingletonString
from utils import load_model, mkdir, save_model, seed_everything

# ================= 以下代码与原脚本完全一致，仅把 main 里的默认参数换成 argparse =================
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DistillKL(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss

def param_dist(model, swa_model, p):
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / output.size(0)))
        return res

def validate(val_loader, model, criterion, opt, logger, quiet=False):
    batch_time = AverageMeter(); losses = AverageMeter(); top1 = AverageMeter(); top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input, target = input.to(opt.device), target.to(opt.device)
            output = model(input)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0)); top5.update(acc5[0], input.size(0))
            batch_time.update(time.time() - end); end = time.time()
            if not quiet and idx % opt.print_freq == 0:
                logger.info(f'Test: [{idx}/{len(val_loader)}]\t'
                            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                            f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
        if not quiet:
            logger.info(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return top1.avg, top5.avg, losses.avg

def train_distill(epoch, train_loader, module_list, swa_model, criterion_list, optimizer,
                  opt, split, logger, quiet=False):
    for module in module_list: module.train()
    module_list[-1].eval()
    criterion_cls, criterion_div, criterion_kd = criterion_list
    model_s, model_t = module_list[0], module_list[-1]
    batch_time = AverageMeter(); data_time = AverageMeter(); losses = AverageMeter(); kd_losses = AverageMeter(); top1 = AverageMeter()
    start = time.time()
    for idx, data in enumerate(train_loader):
        input, target = data[0], data[1]
        data_time.update(time.time() - start)
        input, target = input.to(opt.device), target.to(opt.device)
        logit_s = model_s(input)
        with torch.no_grad(): logit_t = model_t(input)
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        loss_kd = 0
        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd if split == 'minimize' else -loss_div
        if swa_model is not None: loss += param_dist(model_s, swa_model, opt.smoothing)
        if split == 'minimize':
            acc1, _ = accuracy(logit_s, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0)); top1.update(acc1[0], input.size(0))
        else: kd_losses.update(loss.item(), input.size(0))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        batch_time.update(time.time() - start); start = time.time()
        if not quiet and idx % opt.print_freq == 0 and split == 'minimize':
            logger.info(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')
    return (top1.avg, losses.avg) if split == 'minimize' else kd_losses.avg

def adjust_learning_rate(epoch, opt, optimizer):
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    new_lr = opt.sgda_learning_rate * (opt.lr_decay_rate ** steps)
    for param_group in optimizer.param_groups: param_group['lr'] = new_lr
    return new_lr

def scrub_unlearning(args, teacher, student, retain_loader, forget_loader,
                     valid_loader_full, class_to_forget, logger, device):
    model_t, model_s = deepcopy(teacher), deepcopy(student)
    module_list = nn.ModuleList([model_s])
    trainable_list = nn.ModuleList([model_s])
    criterion_list = nn.ModuleList([
        nn.CrossEntropyLoss(),
        DistillKL(args.kd_T),
        DistillKL(args.kd_T)
    ])
    if args.optim == 'sgd':
        optimizer = optim.SGD(trainable_list.parameters(), lr=args.sgda_learning_rate,
                              momentum=args.sgda_momentum, weight_decay=args.sgda_weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(trainable_list.parameters(), lr=args.sgda_learning_rate,
                               weight_decay=args.sgda_weight_decay)
    else:
        optimizer = optim.RMSprop(trainable_list.parameters(), lr=args.sgda_learning_rate,
                                  momentum=args.sgda_momentum, weight_decay=args.sgda_weight_decay)
    module_list.append(model_t)
    module_list = module_list.to(device)
    criterion_list = criterion_list.to(device)
    logger.info("==> SCRUB unlearning ...")
    start_time = time.time()
    for epoch in range(1, args.sgda_epochs + 1):
        lr = adjust_learning_rate(epoch, args, optimizer)
        maximize_loss = 0
        if epoch <= args.msteps:
            maximize_loss = train_distill(epoch, forget_loader, module_list, None,
                                          criterion_list, optimizer, args, "maximize", logger, quiet=True)
        train_acc, train_loss = train_distill(epoch, retain_loader, module_list, None,
                                              criterion_list, optimizer, args, "minimize", logger, quiet=True)
        logger.info(f"maximize loss: {maximize_loss:.2f}\t minimize loss: {train_loss:.2f}\t train_acc: {train_acc:.2f}")
    logger.info(f" +++++++++++ Time taken: {time.time() - start_time:.4f} sec +++++++++++ ")
    return model_s

def data_model_test(model, data, batch_size, num_workers, device, transform=None):
    from learner import complete_test
    test_loader = DataLoader(data, batch_size, shuffle=False, num_workers=num_workers)
    acc, auc = complete_test(model, test_loader, device, transform)
    return acc, auc

# ========================== main ==========================
def parse_args():
    parser = argparse.ArgumentParser()
    # 数据集 & 数据增强
    parser.add_argument('--dataset', default='tiny_imagenet', choices=['cifar10','cifar100','svhn','tiny_imagenet'])
    parser.add_argument('--preproc_train_transform', default='normal', choices=['normal','test','none'])
    parser.add_argument('--preproc_test_transform', default='test')
    parser.add_argument('--select_classes', default=None, type=int, nargs='+')
    # 训练规模
    parser.add_argument('--num_samples', type=int, default=-1, help='-1 表示用全部训练集')
    parser.add_argument('--num_to_forget', type=int, default=10000)
    parser.add_argument('--valid_size', type=int, default=10000)
    # 模型
    parser.add_argument('--model', default='resnet50', choices=['resnet18','resnet50','densenet','vit','simple_cnn'])
    parser.add_argument('--model_path', required=True, help='预训练好的原始模型')
    # 优化 & 训练
    parser.add_argument('--optim', default='adam', choices=['sgd','adam','rmsp'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--device_id', type=int, default=0)
    # SCRUB 特殊超参
    parser.add_argument('--sgda_epochs', type=int, default=10)
    parser.add_argument('--sgda_learning_rate', type=float, default=1e-5)
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[3,5,9])
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--sgda_weight_decay', type=float, default=5e-4)
    parser.add_argument('--sgda_momentum', type=float, default=0.9)
    parser.add_argument('--retain_bs', type=int, default=128)
    parser.add_argument('--forget_bs', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--smoothing', type=float, default=0.5)
    parser.add_argument('--msteps', type=int, default=2)
    parser.add_argument('--kd_T', type=float, default=4.0)
    parser.add_argument('--print_freq', type=int, default=50)
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # 日志
    log_dir = 'logs'
    exp_name = f'{args.dataset}_{args.num_samples}_SCRUB'
    logger = create_logger(os.path.join(log_dir, exp_name), 'scrub')
    logger.info(args)

    # 数据
    from transforms import Transforms
    from data_tool import DataStore, DataLoaderTool, construct_data
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    if args.model in ['resnet50','densenet','vit']:
        dt_size = 224
        feature_dims = [3, dt_size, dt_size]
    processor = Transforms(dt_mean, dt_std, dt_size)
    train_transform = processor.get_transform(args.preproc_train_transform, num_classes)
    test_transform  = processor.get_transform(args.preproc_test_transform, num_classes)

    dataset_conf = {"num_samples": args.num_samples, "select_classes": args.select_classes}
    train_data, test_data = DataLoaderTool.load_dataset(args.dataset, train_transform, test_transform, **dataset_conf)

    data, data_unlearn, data_retain = construct_data(
        train_data, test_data,
        train_size=args.num_samples,
        valid_size=args.valid_size,
        forget_size=args.num_to_forget,
        forget_classes=None          # 按「样本」遗忘，而非「类别」遗忘
    )

    # 加载原始模型
    logger.info(f'Load original model from {args.model_path}')
    model_org = load_model(args.model_path).to(device)
    student = deepcopy(model_org)

    # 数据加载器
    retain_loader = DataLoader(data['retain'],   batch_size=args.retain_bs, shuffle=True,  num_workers=args.num_workers)
    forget_loader = DataLoader(data['forget'],   batch_size=args.forget_bs, shuffle=True,  num_workers=args.num_workers)
    valid_loader  = DataLoader(data['valid'],    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 执行 SCRUB
    start = time.time()
    model_res = scrub_unlearning(args, model_org, student,
                                 retain_loader, forget_loader,
                                 valid_loader, None, logger, device)
    logger.info(f' SCRUB finished in {time.time()-start:.2f} s')

    # 保存
    from tool import PathGenerator as pather
    out_path = pather.get_target_model_path(f'{args.dataset}_{args.model}/SCRUB/')
    out_fn = (f'scrub_retainbs{args.retain_bs}_forgetbs{args.forget_bs}_'
              f'optim{args.optim}_alpha{args.alpha}_gamma{args.gamma}_'
              f'beta{args.beta}_msteps{args.msteps}_sgdaepoch{args.sgda_epochs}_kdT{args.kd_T}.pth')
    os.makedirs(out_path, exist_ok=True)
    save_model(model_res, os.path.join(out_path, out_fn))
    logger.info(f'Saved unlearned model -> {out_path}/{out_fn}')

    # 测试
    def quick_test(loader):
        acc, _ = data_model_test(model_res, loader.dataset, args.batch_size, args.num_workers, device)
        return acc
    logger.info('=== Final accuracy ===')
    logger.info(f'Retain train acc: {quick_test(retain_loader):.4f}')
    logger.info(f'Forget train acc: {quick_test(forget_loader):.4f}')
    logger.info(f'Valid        acc: {quick_test(valid_loader):.4f}')
    logger.info('Done!')

if __name__ == '__main__':
    main()
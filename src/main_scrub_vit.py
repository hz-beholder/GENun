import argparse
from copy import deepcopy
import os
import sys
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from model_deep import get_model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from logger import create_logger
from const import SingletonString
from utils import load_model, mkdir, save_model, seed_everything
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
import time, os, argparse, sys
# … 其余 import 保持不变 …

# ========== 1. JSD ==========
def JSDiv(p, q):
    """p,q: [B,C] 概率（已 softmax）  return: scalar 0~ln2"""
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(torch.log(m.clamp_min(1e-12)), p, reduction='none').sum(dim=1)
    kl_qm = F.kl_div(torch.log(m.clamp_min(1e-12)), q, reduction='none').sum(dim=1)
    return 0.5 * (kl_pm + kl_qm).mean()

# ========== 2. ZRF ==========
@torch.no_grad()
def UnLearningScore(tmodel, gold_model, forget_dl, device):
    preds_t, preds_g = [], []
    for x, *_ in forget_dl:
        x = x.to(device)
        preds_t.append(F.softmax(tmodel(x), dim=1).cpu())
        preds_g.append(F.softmax(gold_model(x), dim=1).cpu())
    preds_t = torch.cat(preds_t, dim=0)
    preds_g = torch.cat(preds_g, dim=0)
    return 1 - JSDiv(preds_t, preds_g).item()

# ========== 3. 激活距离 ==========
@torch.no_grad()
def actv_dist(model1, model2, dataloader, device):
    dist = []
    for x, *_ in dataloader:
        x = x.to(device)
        p1 = F.softmax(model1(x), dim=1)
        p2 = F.softmax(model2(x), dim=1)
        dist.append(torch.norm(p1 - p2, p=2, dim=1).cpu())
    return torch.cat(dist).mean().item()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


def param_dist(model, swa_model, p):
    #This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist


def data_model_test(model, data, batch_size, num_workers, device, transform=None):
    from learner import complete_test
    test_loader = DataLoader(data, batch_size, False, num_workers=num_workers)
    acc, auc = complete_test(model, test_loader, device, transform)
    return acc, auc


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    new_lr = opt.sgda_learning_rate
    if steps > 0:
        new_lr = opt.sgda_learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr

def interclass_confusion(model, dataloader, class_to_forget, device):
    criterion = torch.nn.CrossEntropyLoss()
    dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=128, shuffle=False)
    model.eval()
    reals=[]
    predicts=[]
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        probs = torch.nn.functional.softmax(output, dim=1)
        predict = np.argmax(probs.cpu().detach().numpy(),axis=1)
        reals = reals + list(target.cpu().detach().numpy())
        predicts = predicts + list(predict)
    
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    cm = confusion_matrix(reals, predicts, labels=classes)
    counts = 0
    for i in range(len(cm)):
        if i != class_to_forget[0]:
            counts += cm[class_to_forget[0]][i]
        if i != class_to_forget[1]:
            counts += cm[class_to_forget[1]][i]
    
    ic_err = counts / (np.sum(cm[class_to_forget[0]]) + np.sum(cm[class_to_forget[1]]))
    fgt = cm[class_to_forget[0]][class_to_forget[1]] + cm[class_to_forget[1]][class_to_forget[0]]
    #print (cm)
    return ic_err, fgt


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(val_loader, model, criterion, opt, logger, quiet=False):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if not quiet:
                if idx % opt.print_freq == 0:
                    logger.info('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           idx, len(val_loader), batch_time=batch_time, loss=losses,
                           top1=top1, top5=top5))
        if not quiet:
            logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def train_distill(epoch, train_loader, module_list, swa_model, criterion_list, optimizer, 
                  opt, split, logger, quiet=False):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()

    start = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target = data
        data_time.update(time.time() - start)

        input = input.float()
        
        input, target = input.to(opt.device), target.to(opt.device)
        if opt.distill in ['crd']:
            contrast_idx = contrast_idx.to(opt.device)
            index = index.to(opt.device)
        # if torch.cuda.is_available():
            # input = input.cuda()
            # target = target.cuda()
            # if opt.distill in ['crd']:
            #     contrast_idx = contrast_idx.cuda()
            #     index = index.cuda()

        # ===================forward=====================
        # feat_s, logit_s = model_s(input, is_feat=True, preact=False)
        logit_s = model_s(input)
        with torch.no_grad():
            #feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            #feat_t = [f.detach() for f in feat_t]
            logit_t = model_t(input)

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        loss_kd = 0
        # # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        # elif opt.distill == 'hint':
        #     f_s = module_list[1](feat_s[opt.hint_layer])
        #     f_t = feat_t[opt.hint_layer]
        #     loss_kd = criterion_kd(f_s, f_t)
        # elif opt.distill == 'crd':
        #     f_s = feat_s[-1]
        #     f_t = feat_t[-1]
        #     loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        # elif opt.distill == 'attention':
        #     g_s = feat_s[1:-1]
        #     g_t = feat_t[1:-1]
        #     loss_group = criterion_kd(g_s, g_t)
        #     loss_kd = sum(loss_group)
        # elif opt.distill == 'nst':
        #     g_s = feat_s[1:-1]
        #     g_t = feat_t[1:-1]
        #     loss_group = criterion_kd(g_s, g_t)
        #     loss_kd = sum(loss_group)
        # elif opt.distill == 'similarity':
        #     g_s = [feat_s[-2]]
        #     g_t = [feat_t[-2]]
        #     loss_group = criterion_kd(g_s, g_t)
        #     loss_kd = sum(loss_group)
        # elif opt.distill == 'rkd':
        #     f_s = feat_s[-1]
        #     f_t = feat_t[-1]
        #     loss_kd = criterion_kd(f_s, f_t)
        # elif opt.distill == 'pkt':
        #     f_s = feat_s[-1]
        #     f_t = feat_t[-1]
        #     loss_kd = criterion_kd(f_s, f_t)
        # elif opt.distill == 'kdsvd':
        #     g_s = feat_s[1:-1]
        #     g_t = feat_t[1:-1]
        #     loss_group = criterion_kd(g_s, g_t)
        #     loss_kd = sum(loss_group)
        # elif opt.distill == 'correlation':
        #     f_s = module_list[1](feat_s[-1])
        #     f_t = module_list[2](feat_t[-1])
        #     loss_kd = criterion_kd(f_s, f_t)
        # elif opt.distill == 'vid':
        #     g_s = feat_s[1:-1]
        #     g_t = feat_t[1:-1]
        #     loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
        #     loss_kd = sum(loss_group)
        # else:
        #     raise NotImplementedError(opt.distill)

        if split == "minimize":
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        elif split == "maximize":
            loss = -loss_div

        if swa_model is not None:
            loss += param_dist(model_s, swa_model, opt.smoothing)

        if split == "minimize" and not quiet:
            acc1, _ = accuracy(logit_s, target, topk=(1,1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), input.size(0))
        elif split == "linear" and not quiet:
            acc1, _ = accuracy(logit_s, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            kd_losses.update(loss.item(), input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_value_(model_s.parameters(), clip)
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - start)
        start = time.time()

        if not quiet:
            if split == "mainimize":
                if idx % opt.print_freq == 0:
                    logger.info('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, idx, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1))
                    sys.stdout.flush()

    
    if split == "minimize":
        if not quiet:
            logger.info(' * Acc@1 {top1.avg:.3f} '
                  .format(top1=top1))

        return top1.avg, losses.avg
    else:
        return kd_losses.avg


def scrub_unlearning(args, teacher, student, retain_loader, forget_loader, 
                     valid_loader_full, class_to_forget, logger=None, device='cuda'):
    model_t = deepcopy(teacher)
    model_s = deepcopy(student)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    if args.optim == "sgd":
        optimizer = optim.SGD(trainable_list.parameters(), lr=args.sgda_learning_rate, momentum=args.sgda_momentum,
                            weight_decay=args.sgda_weight_decay)
    elif args.optim == "adam": 
        optimizer = optim.Adam(trainable_list.parameters(), lr=args.sgda_learning_rate, weight_decay=args.sgda_weight_decay)
    elif args.optim == "rmsp":
        optimizer = optim.RMSprop(trainable_list.parameters(), lr=args.sgda_learning_rate, momentum=args.sgda_momentum,
                            weight_decay=args.sgda_weight_decay)
    module_list.append(model_t)

    # if torch.cuda.is_available():
    module_list = module_list.to(device)
    criterion_list = criterion_list.to(device)
    # import torch.backends.cudnn as cudnn
    # cudnn.benchmark = True
    
    # acc_rs = []
    # acc_fs = []
    # acc_vs = []
    # ic_rs = []
    # ic_vs = []
    # fgt_rs = []
    # fgt_vs = []
    logger.info("==> SCRUB unlearning ...")
    start_time = time.time()
    for epoch in range(1, args.sgda_epochs + 1):
        lr = adjust_learning_rate(epoch, args, optimizer)
        
        # ic_r, fgt_r, ic_v, fgt_v = 0.0, 0.0, 0.0, 0.0
        # if class_to_forget is not None and len(class_to_forget) > 0:
        #     ic_r, fgt_r = interclass_confusion(model_s, retain_loader, class_to_forget, device)
        #     ic_v, fgt_v = interclass_confusion(model_s, valid_loader_full, class_to_forget, device)

        # ic_rs.append(ic_r)
        # ic_vs.append(ic_v)
        # fgt_rs.append(fgt_r)
        # fgt_vs.append(fgt_v)

        # acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls, args, logger, True)
        # acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, args, logger, True)
        # acc_v, acc5_v, loss_v = validate(valid_loader_full, model_s, criterion_cls, args, logger, True)
        # acc_rs.append(100-acc_r.item())
        # acc_fs.append(100-acc_f.item())
        # acc_vs.append(100-acc_v.item())

        maximize_loss = 0
        if epoch <= args.msteps:
            maximize_loss = train_distill(epoch, forget_loader, module_list, None, 
                                          criterion_list, optimizer, args, "maximize", logger)
        train_acc, train_loss = train_distill(epoch, retain_loader, module_list, None, 
                                              criterion_list, optimizer, args, "minimize", logger)
        
        logger.info("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))
    logger.info(" +++++++++++ Time taken: {0:.4f} sec +++++++++++ ".format(time.time() - start_time))
    
    # ic_r, fgt_r, ic_v, fgt_v = 0, 0, 0, 0
    # if class_to_forget is not None and len(class_to_forget) > 0:
    #     ic_r, fgt_r = interclass_confusion(model_s, retain_loader, class_to_forget, device)
    #     ic_v, fgt_v = interclass_confusion(model_s, valid_loader_full, class_to_forget, device)
    # ic_rs.append(ic_r)
    # ic_vs.append(ic_v)
    # fgt_rs.append(fgt_r)
    # fgt_vs.append(fgt_v)
    # acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls, args, logger, True)
    # acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, args, logger, True)
    # acc_tv, acc5_v, loss_v = validate(valid_loader_full, model_s, criterion_cls, args, logger, True)
    # acc_rs.append(100-acc_r.item())
    # acc_fs.append(100-acc_f.item())
    # acc_vs.append(100-acc_v.item())
    
    return  model_s  #, (acc_rs, acc_fs, acc_vs, ic_rs, ic_vs, fgt_rs, fgt_vs)
    


if __name__ == '__main__':    
    args = argparse.Namespace()
    args.cuda = 1
    args.model = "vit" #"vit"  #"simple_cnn"  #
    args.dataset = 'cifar100' #"cifar100"   # "svhn" #
    args.preproc_train_transform = 'normal'
    args.preproc_test_transform = 'test'
    args.select_classes = None
    args.batch_size = 256
    args.num_workers = 64
    
    args.optim = 'adam'  # 'sgd'  #
    args.log_dir = 'logs'
    args.unlearn_method = 'SCRUB'
        
    args.num_samples = -1
    args.forget_classes = []  #[2]  # 
    if args.forget_classes is None or len(args.forget_classes) == 0:
        args.num_to_forget = 5000
        args.valid_size = 5000
        # args.num_to_forget = 7000
        # args.valid_size = 7000
    else:
        args.num_to_forget = 10000
        args.valid_size = 10000
    # ****** 新增：金标模型路径 ******
    args.gold_model_path = './outs/cifar100_-1//model_bases/cifar100_vit/RT/cifar100_vit_seed-0_Nf-5000_ep-10_bs-256_lr-[0_0001-0_0002]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth'
    if args.forget_classes is not None and len(args.forget_classes) > 0:
        outfix = f"{outfix}_FGCLS-{args.forget_classes[0]}"
    
    
    args.out_dir = "./outs/cifar100_-1_genun"
    args.exp_name = "cifar100_-1_genun"
    
    # args.out_dir = f'./outs/{args.dataset}_{args.num_samples}{outfix}'
    # args.exp_name = f'{args.dataset}_{args.num_samples}{outfix}'
    
    if args.forget_classes is None or len(args.forget_classes) == 0:
        ## for all data and selective unlearn (random selected samples)
        args.model_path = "./outs/cifar100_-1//model_bases/cifar100_vit/ORG/cifar100_vit_seed-0_Nf-5000_ep-20_bs-256_lr-[0_0001-0_0002]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"
        # args.model_path = "./outs/cifar10_-1/model_bases/cifar10_resnet18/ORG/" + \
        #     "cifar10_resnet18_seed-3407_Nf-5000_ep-20_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-testregular_0.0_sched-None.pth"
        # args.model_path = "./outs/svhn_-1/model_bases/svhn_simple_cnn/ORG/" + \
        #       "svhn_simple_cnn_seed-0_Nf-7000_ep-40_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"
        # args.model_path = "./outs/cifar100_-1/model_bases/cifar100_vit/ORG/"+\
        #     "cifar100_vit_seed-0_Nf-5000_ep-40_bs-256_lr-[0_001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"
        # args.model_path = "./outs/cifar100_-1/model_bases/cifar100_vit/ORG/"+\
        #     "cifar100_vit_seed-0_Nf-5000_ep-20_bs-256_lr-[0_0001-0_0002]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"
    else:
        ## for class unlearn 
        args.model_path = "./outs/cifar10_-1_FGCLS-2/model_bases/cifar10_resnet18/ORG/" + \
            "cifar10_resnet18_seed-0_Nf-10000_Cf-[2]_ep-40_bs-256_lr-[0_001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"
    
    str_seed_sidx = args.model_path.find('seed-')
    str_seed_eidx = args.model_path.find('_Nf')
    args.seed = 0
    
    str_fgt_sidx = args.model_path.find('Cf-')
    str_fgt_eidx = args.model_path.find('_ep')
    if str_fgt_sidx > 0:
        args.forget_classes = [int(x) for x in args.model_path[str_fgt_sidx+4:str_fgt_eidx-1].split(',')]
    
    outpath = SingletonString()
    outpath.content = args.out_dir
    
    ### parameter settings specifical for SCRUB
    args.retain_bs = 128
    args.forget_bs = 64
    args.gamma = 1    # 分类损失权重
    args.alpha = 0.1    # KL散度权重
    args.beta = 0       # 其他蒸馏损失
    args.smoothing = 0.5  # 参数距离惩罚
    args.msteps = 2     # 最大化步数
    args.clip = 0.2
    args.sstart = 10
    args.kd_T = 4       # 蒸馏温度
    args.distill = 'kd'

    args.sgda_epochs = 5  # 增加训练轮次
    args.sgda_learning_rate = 0.00001  # 稍微增加学习率
    args.lr_decay_epochs = [3, 5, 9]  # 调整衰减点
    args.lr_decay_rate = 0.1  # 更平缓的衰减
    args.sgda_weight_decay = 5e-4
    args.sgda_momentum = 0.9
    
    # from model_deep import get_model
    from transforms import Transforms
    from tool import PathGenerator as pather
    
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    args.device = device

    logpath = os.path.join(args.log_dir, args.exp_name)
    logger = create_logger(logpath, f'trail_unlearning_{args.unlearn_method}')
    logger.info(args)
    
    from data_tool import DataLoaderTool, DataStore, construct_data
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    if args.model in ['resnet50', 'densenet', 'vit']:
        dt_size = 224
        feature_dims = [3, dt_size, dt_size]

    processor = Transforms(dt_mean, dt_std, dt_size)
    train_transform = processor.get_transform(args.preproc_train_transform, num_classes)
    test_transform = processor.get_transform(args.preproc_test_transform, num_classes)
    transform = None
    
    dataset_conf = { "num_samples": args.num_samples, "select_classes": args.select_classes}
    train_data, test_data = DataLoaderTool.load_dataset(args.dataset, train_transform, test_transform, **dataset_conf)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    data, data_unlearn, data_retain = construct_data(train_data, test_data, train_size=args.num_samples, valid_size=args.valid_size, 
                                                     forget_size=args.num_to_forget, forget_classes=args.forget_classes)

    ## load original model and test
    logger.info(f"Load original model from @ {args.model_path}")
    model_org = load_model(args.model_path).to(device)
    init_mod_ = get_model(args.model, feature_dims, num_classes, pretrained=False).to(device)
    student = deepcopy(model_org)
    
    retain_loader = DataLoader(data['retain'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    forget_loader = DataLoader(data['forget'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(data['valid'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    start_time = time.time()
    # perform scrub unlearning
    model_res = scrub_unlearning(args, model_org, student, retain_loader, forget_loader, 
                                       valid_loader, args.forget_classes, logger, device)
    logger.info(f" +++++++++++ Time taken: {time.time()-start_time:.4f} sec +++++++++++ ")
    
    out_path = pather.get_target_model_path(f"{args.dataset}_{args.model}/SCRUB/")
    out_fn = f'scrub_unlearned_model_retain_bs-{args.retain_bs}_forget_bs-{args.forget_bs}_optim-{args.optim}_alpha-{args.alpha}_gamma-{args.gamma}_beta-{args.beta}_msteps_{args.msteps}_sgda_epochs-{args.sgda_epochs}_kd_T-{args.kd_T}.pth'
    model_outpath = os.path.join(out_path, out_fn)
    logger.info(f"Save the model to @ {model_outpath}")
    save_model(model_res, model_outpath)
    
    # for i, acc in enumerate(accs):
    #     logger.info(f" +++++++++++ {i} +++++++++++ ")
    #     logger.info(f"Acc: {acc}")
    
    ## EVALUATION: using the test data to evaluate the model
    if args.forget_classes is None or len(args.forget_classes) == 0:
        logger.info(" +++++++++++ Unlearned Model Test +++++++++++ ")
        train_acc, train_auc =  data_model_test(model_res, data['train'], args.batch_size, args.num_workers, device, transform)
        train_retain_acc, train_retain_auc = data_model_test(model_res, data['retain'], args.batch_size, args.num_workers, device, transform)
        valid_acc, valid_auc =  data_model_test(model_res, data['valid'], args.batch_size, args.num_workers, device, transform)
        test_acc, test_auc =  data_model_test(model_res, data['test'], args.batch_size, args.num_workers, device, transform)
        forget_acc, forget_auc = data_model_test(model_res, data['forget'], args.batch_size, args.num_workers, device, transform)
        retain_acc, retain_auc = data_model_test(model_res, data_retain, args.batch_size, args.num_workers, device, transform)
        logger.info(
            " ###### [Train, Train_rem, Valid, Test, Forget, Remain-Data] "
            + f"  Acc: {train_acc:.4f}, {train_retain_acc:.4f}, {valid_acc:.4f}, {test_acc:.4f}, {forget_acc:.4f}, {retain_acc:.4f} ######"
        )
        #     + f"| Auc: {train_auc:.4f}, {valid_auc:.4f}, {test_auc:.4f}, {forget_auc:.4f}, {retain_auc:.4f}, {retain_auc:.4f}###### ")
    else:
        logger.info(f" +++++++++++ Forget classes: {args.forget_classes} +++++++++++ ")
        train_retain_acc, _ =  data_model_test(model_res, data['retain'], args.batch_size, args.num_workers, device, transform)
        train_forget_acc, _ =  data_model_test(model_res, data['forget'], args.batch_size, args.num_workers, device, transform)
        test_retain_acc, _ = data_model_test(model_res, data['test'], args.batch_size, args.num_workers, device, transform)
        test_forget_acc, _ = data_model_test(model_res, data_unlearn['test'], args.batch_size, args.num_workers, device, transform)
        valid_retain_acc, _ = data_model_test(model_res, data['valid'], args.batch_size, args.num_workers, device, transform)
        valid_forget_acc, _ = data_model_test(model_res, data_unlearn['valid'], args.batch_size, args.num_workers, device, transform)
        
        logger.info(" ###### Remain [Train, Valid, Test]:"
                            + f"  Acc: {train_retain_acc:.4f}, {valid_retain_acc:.4f}, {test_retain_acc:.4f} ######")
        logger.info(" ###### Forget [Train, Valid, Test]:"
                            + f"  Acc: {train_forget_acc:.4f}, {valid_forget_acc:.4f}, {test_forget_acc:.4f} ######")
    # ================= 新增：三大遗忘指标 =================
    logger.info("========== Unlearning Metrics ==========")

    # 1. 金标模型 = 原始未遗忘模型（与 BadT 保持一致）
    gold_path = args.gold_model_path if args.gold_model_path else args.model_path
    logger.info(f"Loading GOLD model from {gold_path}")
    gold_model = load_model(gold_path).to(device)

    # 2. 教师模型 = 随机初始化模型（init_mod_）
    teacher_model = init_mod_  # 就是你脚本里的随机初始化教师

    # 3. 遗忘集 dataloader
    forget_dl = DataLoader(data['forget'], batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)

    # 4. 计算
    with torch.no_grad():
        preds_g, preds_s = [], []
        for x, *_ in forget_dl:
            x = x.to(device)
            preds_g.append(F.softmax(gold_model(x), dim=1).cpu())
            preds_s.append(F.softmax(model_res(x), dim=1).cpu())
        preds_g = torch.cat(preds_g, dim=0)
        preds_s = torch.cat(preds_s, dim=0)
        jsd_val = JSDiv(preds_g, preds_s).item()
        act_dist = torch.norm(preds_g - preds_s, p=2, dim=1).mean().item()
        zrf_val = UnLearningScore(model_res, teacher_model, forget_dl, device)

    logger.info(f"ActDist(gold,student): {act_dist:.4f}")
    logger.info(f"JSD(gold,student) : {jsd_val:.4f}")
    logger.info(f"ZRF(student,teacher) : {zrf_val:.4f}")   # 越接近 1 越好
    # ================= 结束 =================
    logger.info("********************************* \n\n")
    print("Done!")
    
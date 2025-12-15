#!/usr/bin/env python3
# pyright: reportMissingImports=true, reportUntypedBaseClass=true, reportGeneralTypeIssues=true
from os import path
import time
import types
import contextlib
from copy import deepcopy
import numpy as np
from sklearn import model_selection
from tqdm import tqdm
from opacus import PrivacyEngine
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from early_stopping import EarlyStopping
from utils import AverageMeter, get_accuracy, set_batchnorm_mode, \
                    cutmix_data, mixup_data, save_model, trainable_params_


def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss == 'stce':
        criterion = SoftTargetCrossEntropy()
    elif loss == 'kldiv':
        criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        print("automatically assign mse loss function ...")
        criterion = nn.MSELoss()
    
    return criterion


def optimizer_picker(optimization, param, lr, momentum=0.1, weight_decay=0.0):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr, weight_decay=weight_decay)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr, weight_decay=weight_decay)
    return optimizer


def scheduler_picker(optimizer, scheduler_option, iter_per_epoch=10, 
                        lr=1e-3, batch_size=128, num_epochs=100):
    #TODO: change to other values?
    ## simple cosine annealing scheduler
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = None
    if scheduler_option == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=lr)
    elif scheduler_option == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr)
    elif scheduler_option == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=lr)
    elif scheduler_option == 'LRScheduler':
        scheduler = LR_Scheduler(optimizer, warmup_epochs=0, warmup_lr=0*batch_size/256, 
                        num_epochs=num_epochs, base_lr=lr*batch_size/256, 
                        final_lr=1e-5 *batch_size/256, iter_per_epoch=iter_per_epoch,
                        constant_predictor_lr=False)
    else:
        Warning('No Scheduler is applied.')
    return scheduler

def l2_penalty(model, model_init, weight_decay):
    l2_loss = 0
    for (k, p), (k_init, p_init) in zip(model.named_parameters(), model_init.named_parameters()):
        if p.requires_grad:
            l2_loss +=  (p - p_init).pow(2).sum()
    l2_loss *= (weight_decay/2.)
    return l2_loss


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))


def normalize_l2(x):
    x_reshaped = x.view(x.shape[0], -1, *(1 for _ in range(x.dim() - 2)))
    x /= torch.norm(x_reshaped, dim=1, keepdim=True) + 1e-8
    return x
    # norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
    # norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # return x / norm
    

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def l2_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=2)
    
class SoftTargetCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

    def __call__(self, x: torch.Tensor, target: torch.Tensor):
        return self.forward(x, target)

# Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.T = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        soft_loss = self.kl_div(
            nn.functional.log_softmax(student_logits / self.T, dim=1),
            nn.functional.softmax(teacher_logits / self.T, dim=1)
        ) * (self.T ** 2)
        hard_loss = 0.0 if self.alpha == 0 else self.ce_loss(student_logits, labels)
        # hard_loss = self.ce_loss(student_logits, labels)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss




# https://github.com/PatrickHua/SimSiam/blob/main/optimizers/lr_scheduler.py
class LR_Scheduler(object):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        warmup_lr,
        num_epochs,
        base_lr,
        final_lr,
        iter_per_epoch,
        constant_predictor_lr=False,
    ):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter)
        )

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group["name"] == "predictor":
                param_group["lr"] = self.base_lr
            else:
                lr = param_group["lr"] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr



@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


# https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans/torch/utils.py#L69
def clip_eta(eta, norm, eps):
    """
    PyTorch implementation of the clip_eta in utils_tf.

    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("L1 clip is not implemented.")
            norm = torch.max(
                avoid_zero_div, torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
            )
        elif norm == 2:
            norm = torch.sqrt(
                torch.max(
                    avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                )
            )
        factor = torch.min(
            torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
        )
        eta *= factor
    return eta

def get_or_guess_labels(model, x, **kwargs):
    """
    Get the label to use in generating an adversarial example for x.
    The kwargs are fed directly from the kwargs of the attack.
    If 'y' is in kwargs, then assume it's an untargeted attack and
    use that as the label.
    If 'y_target' in kwargs and is not none, then assume it's a
    targeted attack and use that as the label.
    Otherwise, use the model's prediction as the label and perform an
    untargeted attack.

    :param model: PyTorch model. Do not add a softmax gate to the output.
    :param x: Tensor, shape (N, d_1, ...).
    :param y: (optional) Tensor, shape (N).
    :param y_target: (optional) Tensor, shape (N).
    """
    if "y" in kwargs and "y_target" in kwargs:
        raise ValueError("Can not set both 'y' and 'y_target'.")
    if "y" in kwargs:
        labels = kwargs["y"]
    elif "y_target" in kwargs and kwargs["y_target"] is not None:
        labels = kwargs["y_target"]
    else:
        _, labels = torch.max(model(x), 1)
    return labels

def optimize_linear(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.

    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param norm: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = torch.sign(grad)
    elif norm == 1:
        abs_grad = torch.abs(grad)
        sign = torch.sign(grad)
        red_ind = list(range(1, len(grad.size())))
        abs_grad = torch.abs(grad)
        ori_shape = [1] * len(grad.size())
        ori_shape[0] = grad.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif norm == 2:
        square = torch.max(avoid_zero_div, torch.sum(grad ** 2, red_ind, keepdim=True))
        optimal_perturbation = grad / torch.sqrt(square)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = (
            optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        )
        one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + (
            square > avoid_zero_div
        ).to(torch.float)
        assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are " "currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps * optimal_perturbation
    return scaled_perturbation

def zero_out_clipped_grads(grad, x, clip_min, clip_max):
    """
    Helper function to erase entries in the gradient where the update would be
    clipped.
    :param grad: The gradient
    :param x: The current input
    :param clip_min: Minimum input component value
    :param clip_max: Maximum input component value
    """
    signed_grad = torch.sign(grad)

    # Find input components that lie at the boundary of the input range, and
    # where the gradient points in the wrong direction.
    clip_low = torch.le(x, clip_min) & torch.lt(signed_grad, 0)
    clip_high = torch.ge(x, clip_max) & torch.gt(signed_grad, 0)
    clip = clip_low | clip_high
    grad = torch.where(clip, torch.zeros_like(grad), grad)

    return grad


def generate_weight_mask(model, forget_loader, criterion, optimizer, threshold=0.5, device='cpu'):
    model = model.to(device)

    model.eval()
    gradients = {}
       
    for idx, (inputs, labels) in enumerate(forget_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = - criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and type(param.grad) != int:
                    gradients[name] = gradients.get(name, 0) + param.grad.data
    
    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    sorted_dict_positions = {}
    hard_dict = {}
    
    # Concatenate all tensors into a single tensor
    all_elems = - torch.cat([tensor.flatten() for tensor in gradients.values()])
    # Calculate the threshold index for the top 10% elements
    threshold_index = int(threshold * len(all_elems))
    
    # Calculate positions of all elements
    positions = torch.argsort(all_elems)
    ranks = torch.argsort(positions)
    
    start_index = 0
    for key, tensor in gradients.items():
        num_elements = tensor.numel()
        # tensor_positions = positions[start_index: start_index + num_elements]
        tensor_ranks = ranks[start_index : start_index + num_elements]

        sorted_positions = tensor_ranks.reshape(tensor.shape)
        sorted_dict_positions[key] = sorted_positions

        # Set the corresponding elements to 1
        threshold_tensor = torch.zeros_like(tensor_ranks)
        threshold_tensor[tensor_ranks < threshold_index] = 1
        threshold_tensor = threshold_tensor.reshape(tensor.shape)
        hard_dict[key] = threshold_tensor
        start_index += num_elements

    return hard_dict


def train_step(params, model, loader, criterion, optimizer, epoch, total_epochs, logger, device='cpu', mask=None):
    model = model.to(device)
    criterion = criterion.to(device)
    
    is_dp_defense = params.get('is_dp_defense', False)
    dp_delta = params.get('delta', 1e-5)
    privacy_engine = None
    if is_dp_defense:
        ## train the model with (differential privacy) DP-SGD with per-sample gradients clipping and noise addition to gradients 
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                sample_rate = params.get('sample_rate', 0.6),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier = params.get('sigma'),
                max_grad_norm = params.get('max_per_sample_grad_norm'),
                secure_rng = params.get('secure_rng'),
                epochs=params.get('epochs'),
                target_epsilon=params.get('epsilon'),
                target_delta=dp_delta,
            )
        
        ## For the opacus version <=0.15.0
        # @DeprecationWarning 
        # privacy_engine = PrivacyEngine(
        #     module=model,
        #     sample_rate = args.sample_rate,
        #     alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        #     noise_multiplier = args.sigma,
        #     max_grad_norm = args.max_per_sample_grad_norm,
        #     secure_rng = args.secure_rng,
        # )
        # privacy_engine.attach(optimizer)
    
    ## mixup augumentation parameters
    mx_type = params.get('mixup', 'none')
    mx_ratio = params.get('mixup_prob', 0.5)
    mx_alpha = params.get('mixup_alpha', 1.0)

    ## training parameters
    regular_lamb = params.get('lamb_reg', 0.0)
    dynamic_regular = params.get('dynamic_regular', False)
    no_reg_epochs = params.get('no_reg_epochs', 0)
    loss_sign = params.get('loss_sign', 1)
    clip_grad = params.get('clip', 1.0)
    use_wga = params.get('use_wga', False)
    wga_beta = params.get('wga_beta', 5.0)
    # print(f"regularization_lamba: {regular_lamb}, dynamic_regular: {dynamic_regular}")
    
    model.train()
    if params.get('disable_batchnorm'):
        set_batchnorm_mode(model, train=False)
            
    start = time.time()
    losses, data_time, batch_time = AverageMeter("loss"), AverageMeter("time_data"), AverageMeter("time_batch")
    for inputs, labels in loader:
        optimizer.zero_grad()
        ## Tweak inputs
        inputs, labels = inputs.to(device, non_blocking=True), (labels).to(device, non_blocking=True)
        # if dataset == 'MNIST':    # Flatten the inputs to 1D vector
        #     inputs = inputs.view(inputs.shape[0]*inputs.shape[1],-1)

        ## different data augmentation methods:  image-cut-mixup, mixup
        do_mixup = False
        if mx_type != 'none' and mx_ratio > 0:
            if np.random.rand(1)[0] <= mx_ratio:
                do_mixup = True
                ## cut-mixup
                if mx_type == 'cutmix':
                    inputs, labels_a, labels_b, lam = cutmix_data(x=inputs, y=labels, alpha=mx_alpha, device=device)
                ## mixup
                elif mx_type == 'mixup':
                    inputs, labels_a, labels_b, lam = mixup_data(x=inputs, y=labels, alpha=mx_alpha, device=device)
                
        outputs = model(inputs)
        # outputs = torch.squeeze(outputs)
        
        ## loss computation
        loss = None
        if use_wga and not do_mixup:
            # Weighted Gradient Ascent: per-sample CE, weight by exp(-loss)^beta, and negate
            per_sample_ce = F.cross_entropy(outputs, labels, reduction='none')
            with torch.no_grad():
                weights = torch.exp(-per_sample_ce).pow(wga_beta)
            loss = - (weights * per_sample_ce).mean()
        elif do_mixup:
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            loss = criterion(outputs, labels)
        
        ## model parameters regularization
        loss_reg = 0
        regs = params.get('regularization', 'none')
        if regs == 'none':
            loss_reg = 0        
        elif regs == 'l1':
            loss_reg = l1_regularization(model)
        elif regs == 'l2':
            loss_reg = l2_regularization(model)
        elif regs == 'l1+l2':
            loss_reg = l2_regularization(model) + l1_regularization(model)
        else:
            print(regs)
            raise ValueError("Invalid regularization type")
        
        # print(f"loss (no reg): {loss.item()}, regularization: {loss_reg} \n")
        cur_lamb = 0.0

        if dynamic_regular:
            if epoch < total_epochs - no_reg_epochs:
                cur_lamb = regular_lamb * (1 - (epoch / (total_epochs - no_reg_epochs)))
            elif epoch == total_epochs - no_reg_epochs:
                cur_lamb = regular_lamb
            else:
                cur_lamb = 0.0
        else:
            cur_lamb = regular_lamb
        
        if regs != 'none':
            loss += cur_lamb * loss_reg
    
        # print(f"loss (with reg): {loss.item()}, alpha: {current_alpha} \n")
        loss = loss_sign * loss
        loss.backward()
        
        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None and name in mask:
                    param.grad *= mask[name]
        
        ## gradient clipping to avoid exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        # Log loss
        losses.update(loss.data.item(), labels.size(0))
        batch_time.update(time.time() - start)
        start = time.time()
        
        if is_dp_defense:
            epsilon = privacy_engine.get_epsilon(dp_delta)
            logger.debug(f"\tTrain Epoch: {epoch} \t Loss: {losses.avg:.6f} " +
                        f"(ε = {epsilon:.2f}, δ = {dp_delta})")
            
            ##@DESCRIPTION For the opacus version <=0.15.0
            # epsilon, best_alpha = optimizer.get_privacy_spent(args.delta)
            # logger.debug(f"Train Epoch: {epoch} \t Loss: {losses.avg:.6f}\t"
            #     f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}")

    logger.info('==> Train:[{0}]\t Time:{batch_time.sum:.4f}\t Loss:{loss.avg:.4f}\t'.format(
                                epoch, batch_time=batch_time, loss=losses))
    return model, optimizer, losses.avg


def train(model, train_loader, valid_loader, test_loader, params, logger,
               lossfn = 'ce', epochs=40, patience=10, optim_type='adam', scheduler_type="", 
               momentum=0.1, maxlr=0.001, minlr=0.0001, weight_decay=0.0, device='cpu', checkpoint_dir="", mask=None):
    # Build optimizer on trainable params and log a quick sanity summary
    tparams = trainable_params_(model)
    n_trainable = sum(p.numel() for p in tparams)
    logger.info(f"Trainable param tensors: {len(tparams)}, elements: {n_trainable}")
    if len(tparams) == 0:
        raise RuntimeError("No trainable parameters found — check that the model parameters have requires_grad=True.")
    optimizer = optimizer_picker(optim_type, tparams, maxlr, momentum, weight_decay)
    
    scheduler = None
    if scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=minlr)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=minlr)
    else:
        Warning('No Scheduler is applied.')
    
    # initialize the early_stopping object
    early_stop = EarlyStopping(patience=patience, verbose=True, delta=0, path=f'{checkpoint_dir}_es_ckpt.pt')
    
    train_time = 0.0
    best_metric = 0.0
    best_model = None
    best_time = 0.0

    criterion = loss_picker(lossfn)
    print("==> Computing the initial loss and accuracy ++++++++++++++++++++++++++++++++++++++++++++++++++ ")
    tr_init_loss, tr_init_acc = test(model, train_loader, criterion, device)
    logger.info(" ++++++++++ Before train -- loss: {:.4f}, acc: {:.4f}".format(tr_init_loss, tr_init_acc))
    
    start_time = time.time()
    acc_diff = 0.0
    for iter_ in tqdm(range(epochs)):
        logger.info(f"==> Starting epoch: {iter_}, Learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        model, optimizer, _ = train_step(params, model, train_loader, criterion, optimizer, iter_, epochs, logger, device, mask)
        train_time = time.time() - start_time
        
        tr_loss, tr_acc = test(model, train_loader, criterion, device)
        val_loss, val_acc = test(model, valid_loader, criterion, device)
        ts_loss, ts_acc = test(model, test_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        
        logger.info(f"==> {iter_+1} - {train_time:2.1f}s: Loss: Tr: {tr_loss:.4f}, Val: {val_loss:.4f}, Ts: {ts_loss:.4f} | " +
                            f"Acc: Tr: {tr_acc:.4f}, Val:{val_acc:.4f}, Ts:{ts_acc:.4f}")
        
        ## epoch should be a multiple of 2, after that warm restarts
        # if (((iter_ + 2) & (iter_ + 1) == 0) and iter_ > 4) or iter_ == epochs:
        #     logger.info(f'==> Saving model at {iter_}-th epoch, Tr_Loss: {tr_loss:.4f}; ' +
        #                         f'Acc: Tr:{tr_acc:.4f}, Ts:{ts_acc:.4f}, DIFF.:{ts_acc - tr_acc:.4f}')

        ## @ save the well-generalized model during the training process
        if iter_ == 0 or (val_acc > best_metric and acc_diff >= tr_acc - val_acc):  # val_acc >= tr_acc
            logger.info('  +++> Well-generalized model checkpoint: [{:.4f}]\t'.format(best_metric) + 'current: [{:.4f}]'.format(val_acc))
            acc_diff = tr_acc - val_acc
            save_model(model, f'{checkpoint_dir}_GENE_M_{iter_}.pt')
                    
        ## Uncomment if need best training loss model
        if  val_acc > best_metric:
            logger.info('==> Best Validation Acc: previous: [{:.4f}]\t'.format(best_metric) + 'current: [{:.4f}]\t'.format(val_acc) + 'Time: [{:.4f}]'.format(train_time))
            best_metric = val_acc
            best_time = train_time
            best_model = deepcopy(model)
            # save_model(best_model, path.join(path.dirname(path_prefix), f'{out_model_name}_best.pt'))
     
        early_stop(val_loss, model)
        if early_stop.early_stop:
            logger.info("+++" * 10 + f"\t Early stopping at epoch: {iter_+1} \t" + "+++" * 10)
            break
    
    logger.info('==> Training completed!  Best Valid Accuracy: [{0:.4f}] @ [{1:.3f}]s \n'.format(best_metric, best_time))
    # out_path_ = path.join(path.dirname(path_prefix), f'{out_model_name}.pt')
    # save_model(best_model, out_path_)

    return best_model, train_time


def test(model, loader, criterion, device='cpu'):
    losses, acc_ = AverageMeter("loss"), AverageMeter("error")
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # outputs = torch.squeeze(outputs)
            loss = criterion(outputs, labels)
            losses.update(loss.data, inputs.size(0))
            acc_cur = get_accuracy(outputs, labels)
            acc_.update(acc_cur, inputs.size(0))
    return losses.avg, acc_.avg

def eval(model, loader, device='cpu'):
    model.eval()
    if not next(model.parameters()).is_cuda:
        model = model.to(device)
    
    y_preds = list()
    y_true = list()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            outputs = torch.squeeze(outputs)
            
            preds = torch.argmax(outputs, dim=1)
            y_preds.extend(preds.tolist())
            y_true.extend(targets.tolist())
    
    return y_preds, y_true

def test_acc(model, loader, device):
    model.eval()
    if not next(model.parameters()).is_cuda:
        model = model.to(device)
    
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return correct / total

def test_auc(model, loader, device):
    model.eval()
    if not next(model.parameters()).is_cuda:
        model = model.to(device)
    
    logits_ = list()
    y_true = list()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = torch.squeeze(outputs)
            outputs = torch.softmax(outputs, dim=1)
            
            logits_.extend(outputs.detach().cpu().numpy())
            y_true.extend(targets.detach().cpu().numpy())
            
    if len(np.unique(y_true)) == 2:
        return roc_auc_score(y_true, logits_[:, 1])
    else:
        return roc_auc_score(y_true, logits_, multi_class='ovr') 


def complete_test(model, loader, device, transform=None):
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    logits_ = list()
    y_true = list()
    
    with torch.no_grad():
        for inputs, targets in loader:
            if transform:
                inputs = transform(inputs)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = torch.squeeze(outputs)
            _, predicted = torch.max(outputs.data, 1)
            logits = torch.softmax(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            logits_.extend(logits.detach().cpu().numpy())
            y_true.extend(targets.detach().cpu().numpy())
    
    acc = correct / (total + 1e-5)
    try:
        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, logits_)
        else:
            auc = roc_auc_score(y_true, logits_, multi_class='ovr')
    except ValueError:
            auc = 0.0
    
    return acc, auc


##################################################################################################

def mia_cvs(sample_loss, attack_model, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
        sample_loss : array_like of shape (n,). objective function evaluated on n samples.
        members : array_like of shape (n,), whether a sample was used for training.
        n_splits: int
            number of splits to use in the cross-validation.
    Returns:
        scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    # attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    acc = model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    ).mean()
    auc = model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="roc_auc"
    ).mean()
    return (acc, auc)


######################################### Golatkar et al. #########################################

### Fisher forgetting
def fisher_hessian(model, loader, criterion, device):
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.grad_acc = 0
        param.grad2_acc = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        prob = F.softmax(outputs, dim=-1).data
        
        for c_i in range(outputs.shape[1]):
            pred_tar = torch.empty_like(labels).fill_(c_i)
            loss = criterion(outputs, pred_tar)
            model.zero_grad()
            loss.backward(retain_graph=True)
            for param in model.parameters():
                if param.requires_grad:
                    param.grad_acc += (labels == pred_tar).float() * param.grad.data
                    param.grad2_acc +=  prob[:, c_i] * param.grad.data.pow(2)
    
    for param in model.parameters():
        param.grad_acc /= len(loader)
        param.grad2_acc /= len(loader)
    
    return model
    

def get_mean_var(parameters, num_classes, is_forget_class=False, class_forget=0, lamb=3e-6):
    var = deepcopy(1./(parameters.grad2_acc + 1e-8))
    var = var.clamp(max=1e3)
    if parameters.size(0) == num_classes:
        var = var.clamp(max=1e2)
    var = lamb * var
    
    if parameters.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(parameters).clone()
    
    mu = deepcopy(parameters.data0.clone())
    if parameters.size(0) == num_classes and is_forget_class:
        mu[class_forget] = 0
        var[class_forget] = 0.0001
    if parameters.size(0) == num_classes:
        # Last layer
        var *= 10
    elif parameters.ndim == 1:
        # BatchNorm
        var *= 10
        # var*=1
    return mu, var

# Computes the amount of information not forgotten at all layers using the given lambda
def get_information_left(modelf, modelf0, num_classes, num_to_forget, lamb=1e-6):
    total_kl = 0.0
    for (k, p), (k0, p0) in zip(modelf.named_parameters(), modelf0.named_parameters()):
        mu0, var0 = get_mean_var(p, num_classes, num_to_forget, False, lamb=lamb)
        mu1, var1 = get_mean_var(p0, num_classes, num_to_forget, True, lamb=lamb)
        kl_val = ((mu1 - mu0).pow(2) / var0 + var1/var0 - torch.log(var1/var0) - 1).sum()
        total_kl += kl_val.item()
    return total_kl


### Variational forgetting: as a the way for estimation the Fisher information matrix (FIM) for Hessian
def get_variational_vars(model):
    """Returns all variables involved in optimizing the hessian estimation."""
    result = []
    if hasattr(model, 'logvar0'):
        result.append(model.logvar0)
        result.append(model.loglambda2)
    for l in model.children():
        result += get_variational_vars(l)
    return result

def get_compression_loss(model):
    """Get the model loss function for hessian estimation."""
    # Compute quantity of information in the weights
    # Assuming a normal prior p(w) ~ N(0, lambda**2 * I)
    # (where lambda is selected independently for each layer)
    # we have
    #    I(w; D) ~= KL(q(w|D) | p(w))
    # where the optimal posterior q(w|D) is q(w|D) ~ N(w, F**-1)
    # and F is the Fisher information matrix

    modules = [x for x in model.modules() if hasattr(x, 'logvar0')]
    k = sum([x.weight.numel() for x in modules])

    w_norm2 = sum([x.weight.pow(2).sum() / x.loglambda2.exp() for x in modules])
    logvar = sum([x.logvar.sum() for x in modules])
    trace = sum([x.logvar.exp().sum() / x.loglambda2.exp() for x in modules])
    lambda2_cost = sum([x.loglambda2 * x.weight.numel() for x in modules])

    # Standard formula for KL divergence of two normal distributions
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
    Lz = w_norm2 + trace + lambda2_cost - logvar - k
    return Lz
    # return -logvar

def variational_forward(module, input):
    """Modified forward pass that adds noise to the output."""
    # Recall that module.logvar0 is created by make_variational()
    # (specifically, by add_logvar())
    module.logvar = module.logvar0.expand_as(module.weight).clamp(max=-3.0)
    var = module.logvar.exp()

    if isinstance(module, torch.nn.modules.conv.Conv2d):
        output = F.conv2d(input, module.weight, module.bias, module.stride,
                            module.padding, module.dilation, module.groups)
        # From Variational Dropout and the Local reparametrization trick
        # (Kingma et al., 2015)
        output_var = F.conv2d(input ** 2 + 1e-5, var, None, module.stride,
                                module.padding, module.dilation, module.groups)
    elif isinstance(module, torch.nn.modules.linear.Linear):
        output = F.linear(input, module.weight, module.bias)
        output_var = F.linear(input ** 2 + 1e-5, var, None)
    else:
        raise NotImplementedError(f"Module {type(module)} not implemented.")

    eps = output.data.clone().normal_()
    # print(f"max: {var.max().item():.4f} min: {var.min().item():.4f} nan: {torch.isnan(var).sum()}")
    # if output_var.min().item() < 0:
    #     raise ValueError('Variance less than 0.')
    # Local reparemetrization trick
    if module.disable_noise:
        return output
    else:
        return output + torch.sqrt(output_var) * eps

def _reset_logvar(module, variance_scaling=0.005):
    if hasattr(module, 'logvar0'):
        w = module.weight.data
        # Initial ballpark estimate for optimal variance is the variance
        # of the weights in the kernel
        var = w.view(w.size(0), -1).var(dim=1).view(-1, *([1] * (w.ndimension() - 1)))  # .expand_as(w)
        # Further scale down the variance by some factor
        module.logvar0.data[:] = (var * variance_scaling + 1e-8).log()
        # Initial guess for lambda is the l2 norm of the weights
        module.loglambda2.data = (w.pow(2).mean() + 1e-8).log()

def _add_logvar(module):
    """Adds a parameter (logvar0) to store the noise variance for the weights.

    Also adds a scalar parameter loglambda2 to store the scaling coefficient
    for the layer.

    The variance is assumed to be the same for all weights in the same filter.
    The common value is stored in logvar0, which is expanded to the same
    dimension as the weight matrix in logvar.
    """
    if not hasattr(module, 'weight'):
        return
    if module.weight.data.ndimension() < 2:
        return
    if not hasattr(module, 'logvar0'):
        w = module.weight.data
        # w is of shape NUM_OUT x NUM_IN x K_h X K_w
        var = w.view(w.size(0), -1).var(dim=1).view(-1, *([1] * (w.ndimension() - 1)))
        # var is of shape NUM_OUT x 1 x 1 x 1
        # (so that it can be expanded to the same size as w by torch.expand_as())
        # The content does not matter since we will reset it later anyway
        module.logvar0 = Parameter(var.log())
        # log(lambda**2) is a scalar shared by all weights in the layer
        module.loglambda2 = Parameter(w.pow(2).mean().log())
        module.logvar = module.logvar0.expand_as(module.weight)
        _reset_logvar(module)

def make_variational(model):
    """Replaces the forward pass of the model layers to add noise."""
    model.apply(_add_logvar)
    for m in model.modules():
        if hasattr(m, 'logvar0'):
            m.forward = types.MethodType(variational_forward, m)
            m.disable_noise = False

def vairational_hessian(model, loader, criterion, device):
    model = model.to(device)
    make_variational(model)
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs =  variational_forward(model, inputs)
        loss = criterion(outputs, labels)
        loss += get_compression_loss(model)
        loss.backward()
    ## How to use the logvar0 and loglambda2 to compute the FIM?
    return model




def sam_grad(model, loss):
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = torch.autograd.grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)

def apply_perturb(model, v):
    curr = 0
    for param in model.parameters():
        length = param.view(-1).shape[0]
        param.view(-1).data += v[curr : curr + length].data
        curr += length

def woodfisher(model, loader, criterion, v, N=1000, device='cuda'):
    k_vec = torch.clone(v)
    o_vec = None
    model.eval()
    for idx, (data, label) in enumerate(loader):
        model.zero_grad()
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec






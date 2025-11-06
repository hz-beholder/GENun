from itertools import cycle
import os
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_tool import CustomerDataset
from logger import create_logger
from early_stopping import EarlyStopping
from utils import AverageMeter, cutmix_data, get_accuracy, mixup_data, save_model, trainable_params_

from transforms import Transforms
from losses import AdvLoss, TripletLoss
from learner import loss_picker, scheduler_picker, optimizer_picker, test, l1_regularization, l2_regularization

class UnlearnBasic():
    def __init__(self, logpath: str, logname: str, out_dir: str='results', name: str='model'):
        self.logger = create_logger(logpath, logname + "_" + self.__get_name__())
        self.out_dir = out_dir
        if name is not None and len(name) > 0:
            self._construct_paths_(out_dir, name)
        
        self.loader_dict = None
        self.online_transform = None
    
    def __get_name__(self):
        return 'Unlearn_Base'
    
    def _construct_paths_(self, out_dir, name='model'):
        self.path = dict()
        self.path['model'] = os.path.join(out_dir, name)
        # self.path['result'] = os.path.join(out_dir, 'result')
        for key in self.path.keys():
            if not os.path.exists(self.path[key]):
                os.makedirs(self.path[key])
    
    def set_params(self, **kwargs):
        self.logger.info(f"Set the parameters for the unlearning process")
        self.params = {
            'seed': kwargs.get('seed', 0),
            'epochs': kwargs.get('epochs', 10),
            'loss': kwargs.get('loss', 'ce'),
            'lr': kwargs.get('lr', 1e-3),
            'minlr': kwargs.get('minlr', 1e-4),
            'batch_size': kwargs.get('batch_size', 32),
            'weight_decay': kwargs.get('weight_decay', 5e-4),
            'clip': kwargs.get('clip', 1.0),
            'momentum': kwargs.get('momentum', 0.9),
            'save_checkpoints': kwargs.get('save_checkpoints', False),
            'regularizer': kwargs.get('regularizer', 'l1'), # option: 'l1', 'l2', 'l1+l2', 'l1_diff', 'l2_diff'
            'regular_gamma': kwargs.get('gamma', 1e-4),
            'no_reg_epochs': kwargs.get('no_reg_epochs', 0),
            'alpha': kwargs.get('alpha', 0.5),   # trade-off between loss of retain and forget data (Hyperparameter tuning is needed 0.1 - 0.8 suggested)
            # 'class_wise': kwargs.get('class_wise', False),
            # 'strength': kwargs.get('stength', 1.0),
            'dynamic_weight': kwargs.get('dynamic_weights', False),
        }
        
        self._suffix_ = "epochs-%s_lr-%s_wd-%s_seed-%s_regular-%s_gamma-%s_alpha-%s_noregepoch-%s" % (
            self.params['epochs'], self.params['lr'], self.params['weight_decay'], self.params['seed'],
            self.params['regularizer'], self.params['regular_gamma'], self.params['alpha'], self.params['no_reg_epochs'], 
        )
        self.logger.info(f"Parameters: {self.params}")
        
    @staticmethod
    def _set_lr_(optimizer, new_lr):
        """
        set the learn rate of the input optimizer
        """
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
    
    @staticmethod
    def _online_transform_(inputs, targets, augment):
        if augment == 'mixup':
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
        elif augment == 'cutmix':
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets)

        mixed_x = inputs
        mixed_y = lam * targets_a + (1 - lam) * targets_b          
        return mixed_x, mixed_y

    ##Optional for data transform
    def config_transform(self, mean, std, crop_size):
        self.logger.info(f"Set the online transforms for input data")
        self.online_transform = Transforms(mean, std, crop_size) #OnlineTransforms(mean, std, crop_size)
    
    ## set data sources for the unlearning process: dict: {retain, forget, valid, test}
    def set_data(self, data: dict, num_classes, batch_size: int=128, num_workers: int=4):
        self.logger.info(f"Set the data sources: {list(data.keys())}")
        self.loader_dict = dict()
        for key in data.keys():
            self.loader_dict[key] = DataLoader(data[key], batch_size, shuffle=True, num_workers=num_workers)
        self.num_classes = num_classes
    
    def unlearn(self, model, **kwargs):
        raise NotImplementedError("Unlearn method is not implemented.")
    
    ## model performance evaluation over all datasets
    def evaluation(self, model, loss='ce', device='cuda', augment=None):
        model.to(device)
        model.eval()
        criterion = loss_picker(loss)
        loss_, acc_, acc_cls = dict(), dict(), dict()
        self.logger.info(f"Model evaluation over all datasets: {list(self.loader_dict.keys())}")
        for key, loader_ in self.loader_dict.items():
            loss_k, acc_k = AverageMeter("loss"), AverageMeter("accuracy")
            acc_cls_k = dict()
            
            with torch.no_grad():
                for inputs, targets in loader_:
                    # if augment != "none" and augment != None and self.online_transform:
                    #     inputs, targets = self.online_transform(augment, inputs, targets, self.num_classes)
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    loss_k.update(loss.data, inputs.size(0))
                    acc_cur = get_accuracy(outputs, targets)
                    acc_k.update(acc_cur, inputs.size(0))
                    
                    for i in range(len(targets)):
                        y_i = targets[i].item()
                        if y_i not in acc_cls_k:
                            acc_cls_k[y_i] = AverageMeter(f"acc_{targets[i]}")
                        pred_i = torch.argmax(outputs[i]).item()
                        acc_cls_k[y_i].update( int(pred_i == y_i) )
            
            # acc_cls_str = ""
            # for c_i in sorted(acc_cls_k.keys()):
            #     acc_cls_str += f" | {c_i}: {acc_cls_k[c_i].avg:.4f}"
            # self.logger.info(f"{key}: loss: {loss_k.avg:.4f}, accuracy: {acc_k.avg:.4f}, class_acc: {acc_cls_str}")
            self.logger.info(f"{key}: loss: {loss_k.avg:.4f}, accuracy: {acc_k.avg:.4f}")
            loss_[key], acc_[key], acc_cls[key] = loss_k.avg, acc_k.avg, acc_cls_k
            
        return loss_, acc_, acc_cls
    
    @torch.no_grad()
    def model_evaluate(self, model, loader:DataLoader, device="cuda", transform=None):
        num_ = 0
        total_correct = 0
        logits_ = list()
        y_true = list()
        model.eval()
        with torch.no_grad():
            for inputs, targets in loader:
                # if transform != "none"  and transform != None and self.online_transform:
                #     inputs, targets = self.online_transform('test', inputs, targets, self.num_classes)
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                logits = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, dim=1)
                total_correct += preds.eq(targets).sum().item()
                logits_.extend(logits.detach().cpu().numpy())
                y_true.extend(targets.detach().cpu().numpy())
                num_ += targets.size(0)
        acc = total_correct / num_
        try:
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, logits_)
            else:
                auc = roc_auc_score(y_true, logits_, multi_class='ovr')
        except Exception as e:
            auc = 0.0
            
        return acc, auc


class GeneModUnlearn(UnlearnBasic):
    def __init__(self, logpath, logname, out_dir = 'results', name = 'model'):
        super().__init__(logpath, logname, out_dir, name)
        
    def __get_name__(self):
        return 'GenUn'

    def weight_regularization(self, model, regularizer='l1'):
        loss_reg = 0
        if regularizer == 'none':
            loss_reg = 0        
        elif regularizer == 'l1':
            loss_reg = l1_regularization(model)
        elif regularizer == 'l2':
            loss_reg = l2_regularization(model)
        elif regularizer == 'l1+l2':
            loss_reg = l2_regularization(model) + l1_regularization(model)
        else:
            print(regularizer)
            raise ValueError("Invalid regularization type")
        return loss_reg
    
    def __weight_diff__(self, model, model_init, regulartizer):
        l_diff = 0.0
        for (_, p), (_, p_init) in zip(model.named_parameters(), model_init.named_parameters()):
            if p.requires_grad:
                l_diff += (p - p_init).pow(regulartizer).sum()
        
        if 'l2' in regulartizer:
            l_diff /= 2
        return l_diff


    def unlearn(self, init_model, adv_perturb_loss=None, feedback_loss=None, patience=20, scheduler_option="", 
                optimization='adam', run_train_aug="none", run_forget_aug="none", device='cpu', checkpoint_path='.', **kwargs):
        ## finetune the init_model with better generalization ability with smallest gap between acc_train and acc_valid
        protocal = 'genun'
        model_type = kwargs.get('model_type', 'unknown')
        over_forget = kwargs.get('over_forget', False)
        sfx = kwargs.get('suffix', '')
        clip_grad = self.params.get('clip', 1.0)

        # basic info.
        self.logger.info(f"Checkpoint name: {model_type}_{protocal}")
        self.logger.info(f"Unlearning from well-generalized model | Adv: {adv_perturb_loss} | Feedback: {feedback_loss} | " 
                 f"Optimizer: {optimization} | Train Aug: {run_train_aug} | Forget Aug: {run_forget_aug} | Scheduler: {scheduler_option}")
        self.logger.info(f"Dataset: Train: {len(self.loader_dict['retain'].dataset)} | Valid: {len(self.loader_dict['valid'].dataset)} | "+ \
                            f"Forget: {len(self.loader_dict['forget'].dataset)} | Test: {len(self.loader_dict['test'].dataset)}")

        ## dataloader and model
        model = deepcopy(init_model)
        retain_loader = self.loader_dict['retain']
        forget_loader = self.loader_dict['forget']
        
        optimizer = optimizer_picker(optimization, trainable_params_(model), lr=self.params['lr'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler = scheduler_picker(optimizer, scheduler_option, 1, self.params['minlr'], self.params['batch_size'], self.params['epochs'])
        criterion = loss_picker(self.params['loss']).to(device)
        
        ersp_fn = os.path.join(checkpoint_path, f'{self.__get_name__()}_{protocal}_{self._suffix_}_es_ckpt.pt')
        early_stop = EarlyStopping(patience=patience, verbose=True, delta=0, path=ersp_fn)
        
        ## model test pre-finetuning
        tr_init_loss, tr_init_acc = test(model, retain_loader, criterion, device)
        val_loss, val_acc = test(model, self.loader_dict['valid'], criterion, device)
        self.logger.info(f"++++++++++++++++ Before finetuning +++++++++++++++ ")
        self.logger.info(f"Loss: (retain - {tr_init_loss:.4f}, val - {val_loss:.3f}), Acc: (retain - {tr_init_acc:.4f}, val - {val_acc:.4f})")

        best_model = None
        cum_time, best_time = 0.0, 0.0
        best_metric, best_iter = 0.0, -1
        met_loss, met_tm = AverageMeter("loss"), AverageMeter("time")
        model = model.to(device)
        model.train()
        
        self.logger.info(f"Start the training process for {self.params['epochs']} epochs")
        for epoch in tqdm(range(self.params['epochs'])):
            self.logger.info(f"==> Starting epoch: {epoch+1}, Learning rate: {optimizer.param_groups[0]['lr']:.5f}")
            start_time = time.time()
            
            ## the dynamic gamma factor for regualarization term.
            cur_gamma = 0.0
            if self.params['dynamic_weight']:
                if epoch < self.params['epochs'] - self.params['no_reg_epochs']:
                    cur_gamma = self.params['regular_gamma'] * (1 - epoch / (self.params['epochs'] - self.params['no_reg_epochs']))
                elif epoch == self.params['epochs'] - self.params['no_reg_epochs']:
                    cur_gamma = self.params['regular_gamma']
                else:
                    cur_gamma = 0.0
            else:
                cur_gamma = self.params['regular_gamma']

            ## loss computing for retain dataloader (and forget dataloader)
            if self.params['alpha'] == 1.0 or not over_forget:  # only retain data for training
                for in_r, tar_r in retain_loader:
                    optimizer.zero_grad()
                    in_r, tar_r = in_r.to(device), tar_r.to(device)
                    out_r = model(in_r)
                    
                    ## regularization for model weights
                    if self.params['regularizer'] in ['l1_diff', 'l2_diff']:
                            regular = self.__weight_diff__(model, init_model, self.params['regularizer'])
                    elif self.params['regularizer'] in ['l1', 'l2', 'l1+l2']:
                        regular = self.weight_regularization(model, self.params['regularizer'])
                    else:
                        regular = 0.0
                    
                    loss_r = criterion(out_r, tar_r) + cur_gamma * regular
                    loss_r.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()
                    met_loss.update(loss_r, in_r.size(0))
            else:               # retain and forget data for training
                for (in_r, tar_r), (in_f, tar_f) in zip(retain_loader, cycle(forget_loader)):
                    optimizer.zero_grad()
                    # wrapper the smaller one with cycle to repeat its batches indefinitely
                    in_r, tar_r = in_r.to(device), tar_r.to(device)
                    in_f, tar_f = in_f.to(device), tar_f.to(device)
                    out_r, out_f = model(in_r), model(in_f)
                    
                    ## regularization for model weights
                    if self.params['regularizer'] in ['l1_diff', 'l2_diff']:
                            regular = self.__weight_diff__(model, init_model, self.params['regularizer'])
                    elif self.params['regularizer'] in ['l1', 'l2', 'l1+l2']:
                        regular = self.weight_regularization(model, self.params['regularizer'])
                    else:
                        regular = 0.0
                                        
                    loss_r = criterion(out_r, tar_r) +  cur_gamma * regular
                    loss = self.params['alpha'] * loss_r - (1 - self.params['alpha']) * criterion(out_f, tar_f)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()
                    met_loss.update(loss_r, in_r.size(0))
                    
            met_tm.update(time.time() - start_time, in_r.size(0))
                                
            if scheduler is not None:
                scheduler.step()
        
            tr_time = time.time() - start_time
            self.logger.info(f"Epoch: {epoch+1}/{self.params['epochs']} |  Time: {tr_time:.3f} sec")
            self.logger.info(f"AVERAGE: Loss-{met_loss.avg:.4f}, Time-{met_tm.avg:.4f}")
            
            ### Testing and Determining the best one after one-train step
            val_loss, val_acc = test(model, self.loader_dict['valid'], criterion, device)
            tr_loss, tr_acc = test(model, retain_loader, criterion, device)
            ts_loss, ts_acc = test(model, self.loader_dict['test'], criterion, device)
            self.logger.info(f" ==> {epoch + 1} @ {tr_time:.3f}s: " + \
                             f"Loss-train:{tr_loss:.3f}, val:{val_loss:.3f}, test:{ts_loss:.3f};" + \
                             f"Acc-train:{tr_acc:.4f}, val:{val_acc:.4f}, test:{ts_acc:.4f}")

            metric = val_acc
            if metric > best_metric:
                self.logger.info('==+++++===> Best Validation Acc: previous: [{:.4f}]\t'.format(best_metric) + \
                                     'current: [{:.4f}]\t'.format(val_acc) + 'Time: [{:.4f}]'.format(tr_time))
                best_acc = val_acc
                best_metric = metric
                best_model = deepcopy(model)
                best_iter = epoch
                best_time = cum_time + time.time() - start_time
            # _, val_acc = test(model, self.loader_dict['valid'], criterion, device)
            cum_time += time.time() - start_time
            
            for key_, loader_ in self.loader_dict.items():
                _, acc_ = test(model, loader_, criterion, device)
                self.logger.info(f"Testing [ {key_:6} ] :  Acc: {acc_:.4f}")
                # acc_, auc_ = self._model_evaluate_(model, loader_, device)
            
            if self.params['save_checkpoints'] and (epoch % 10 == 0) and epoch > 0:
                outfn = f"{model_type}_{protocal}_{sfx}_adv_{adv_perturb_loss}_feedback_{feedback_loss}_{self._suffix_}_ep-{epoch}.pth"
                out_path_ = os.path.join(self.path['model'], self.__get_name__(), outfn)
                save_model(model, out_path_)
            
            early_stop(met_loss.avg, model)
            if early_stop.early_stop:
                self.logger.info("+++" * 10 + f"\t Early stopping at epoch: {epoch+1} \t" + "+++" * 10)
                break
        
        self.logger.info("\n\n")
        _, test_acc = test(best_model, self.loader_dict['test'], criterion, device)
        self.logger.info(f"Best model test accuracy: {test_acc:.4f} @  {best_iter} epoch: {best_time:.3f} sec")
        checkpoint = {
            'epoch': -1,
            'state_dict': best_model.state_dict(),
            'best_acc': test_acc,
            'protocal': protocal,
        }
        
        return best_model, checkpoint
import os
import time
import math
from os import path
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from logger import create_logger
from early_stopping import EarlyStopping
from model_deep import reset_final_layers
from data_tool import RelabeledDataset, flatten_subset
from utils import AverageMeter, cutmix_data, get_accuracy, mixup_data, save_model

from transforms import OnlineTransforms
from losses import AdvLoss, TripletLoss
from learner import DistillationLoss, loss_picker, scheduler_picker, optimizer_picker, test

torch.autograd.set_detect_anomaly(True)

class UnlearnBasic():
    def __init__(self, logpath: str, logname: str, out_dir: str='results', name: str='model'):
        self.logger = create_logger(logpath, logname + "_" + self.__get_name__())
        self.out_dir = out_dir
        if name is not None and len(name) > 0:
            self._construct_paths_(out_dir, name)
        
        self.dataloaders = None
        self.online_transform = None
    
    def __get_name__(self):
        return 'UnlearnBasic'
    
    def _construct_paths_(self, out_dir, name='model'):
        self.path = dict()
        self.path['model'] = path.join(out_dir, name)
        # self.path['result'] = os.path.join(out_dir, 'result')
        for key in self.path.keys():
            if not path.exists(self.path[key]):
                os.makedirs(self.path[key])
    
    def set_params(self,
                epochs: int = 10,
                batch_size: int = 32,
                lr: float = 0.01,
                lr_feedback: float = 0.001,
                adv_lamb: float = 1.0,
                feed_lamb: float = 0.1,
                weight_decay: float = 5e-4,
                momentum: float = 0.9,
                save_checkpoints: bool = False,
                class_wise: bool = False,
                seed: int = 42,
                dynamic_weights: bool = False,
                ):
        self.logger.info(f"Set the parameters for the unlearning process")
        self.params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'lr_feedback': lr_feedback,
            'adv_lamb': adv_lamb,
            'feed_lamb': feed_lamb,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'save_checkpoints': save_checkpoints,
            'class_wise': class_wise,
            'seed': seed,
            'dynamic_weight': dynamic_weights,
        }
        self._suffix_ =  f"epochs-{epochs}_lr-{lr}_lr_fd-{lr_feedback}_clswise-{class_wise}_advreg-{adv_lamb}_feedreg-{feed_lamb}_seed-{seed}"
        # self._suffix_ =  f"epochs-{epochs}_lr-{lr}_clswise-{class_wise}_seed-{seed}"
        self.logger.info(f"Parameters: {self.params}")
        
    @staticmethod
    def _set_lr_(optimizer, new_lr):
        """
        set the learn rate of the input optimizer
        """
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
    
    @staticmethod 
    def _onlinemix_transform_(inputs, targets, augment):
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
        self.online_transform = OnlineTransforms(mean, std, crop_size)
    
    ## set data sources for the unlearning process: dict: {retain, forget, valid, test}
    def set_data(self, data: dict, num_classes, batch_size: int=128, num_workers: int=4):
        self.logger.info(f"Set the data sources: {list(data.keys())}")
        self.dataloaders = dict()
        for key in data.keys():
            self.dataloaders[key] = DataLoader(data[key], batch_size, shuffle=True, num_workers=num_workers)
        self.num_classes = num_classes
    
    def unlearn(self, model, **kwargs):
        raise NotImplementedError("The unlearning method is not implemented.")
    
    ## model performance evaluation over all datasets
    def evaluation(self, model, loss='ce', device='cuda', augment=None):
        model.to(device)
        model.eval()
        criterion = loss_picker(loss)
        loss_, acc_, acc_cls = dict(), dict(), dict()
        self.logger.info(f"Model evaluation over all datasets: {list(self.dataloaders.keys())}")
        for key, loader_ in self.dataloaders.items():
            loss_k, acc_k = AverageMeter("loss"), AverageMeter("accuracy")
            acc_cls_k = dict()
            
            with torch.no_grad():
                for inputs, targets in loader_:
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
    
    @staticmethod
    def _generate_scores_w_masks(model, inputs: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor=None, 
                            mode="train", dim: int = 1, device="cuda"):
        """
            This function is just to handle the input mask used in NLP tasks
        """
        model.train() if mode == "train" else model.eval()
        model.to(device)
        criterion = loss_picker('ce')
        inputs, targets = inputs.to(device), targets.to(device)
        if masks is not None:
            masks = masks.to(device)
            outputs = model(inputs, masks)
        else:
            outputs = model(inputs)
        losses = criterion(outputs, targets)
        new_score = (losses[:, None] if dim == 1 else torch.cat((outputs, losses[:, None]), axis=1))
        return new_score
    
    @torch.no_grad()
    def model_evaluate(self, model, loader:DataLoader, device="cuda", transform=None):
        cnt = 0
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
                cnt += targets.size(0)
        acc = total_correct / cnt
        try:
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, logits_)
            else:
                auc = roc_auc_score(y_true, logits_, multi_class='ovr')
        except Exception as e:
            auc = 0.0
            
        return acc, auc


class DirectUnlearn(UnlearnBasic):    
    def __init__(self, logpath: str, logname: str, out_dir: str='results', out_name: str='model'):
        super(DirectUnlearn, self).__init__(logpath, logname, out_dir, out_name)
        
    def __get_name__(self):
        return 'DirectUnlearn'

    def unlearn(self, model, init_model=None, adv_perturb_losser=None, feedback_losser=None, patience=10, loss='ce', scheduler_option="", 
                optimization='sgd', run_train_aug="none", run_forget_aug="none", device='cpu', checkpoint_path='.', **kwargs):
        self.logger.info(f"Unlearning with Adv-loss: {adv_perturb_losser} | Feedback-loss: {feedback_losser} | " 
                            f"| Optimizer: {optimization} | Train Aug: {run_train_aug} | Forget Aug: {run_forget_aug}")
        
        is_finetune = kwargs.get('is_finetune', True)
        model_type = kwargs.get('model_type', 'unknown')
        class_forget = kwargs.get('forget_classes', None)
        cls_lamb = kwargs.get('classify_lamb', 1.0)
        alpha = kwargs.get('alpha', 0.5)
        T = kwargs.get('temperature', 3.0)
        
        model.train()
        cls_criterion = loss_picker(loss)
        softdis_criterion = DistillationLoss(alpha=alpha, temperature=T)
        
        optimizer_finetune = optimizer_picker(optimization, model.parameters(), lr=self.params['lr'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler_finetune = scheduler_picker(optimizer_finetune, scheduler_option, len(self.dataloaders['retain']))

        optimizer_feedback = optimizer_picker(optimization, model.parameters(), lr=self.params['lr_feedback'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler_feedback = scheduler_picker(optimizer_feedback, scheduler_option, len(self.dataloaders['forget']))
        
        ### applying the adversarial perturbation loss for the model training?
        is_adv_loss = isinstance(adv_perturb_losser, AdvLoss)
        
        ersp_fn = path.join(checkpoint_path, f'{self.__get_name__()}_{self._suffix_}_es_ckpt.pt')
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0, path=ersp_fn)
        
        cum_time, best_time = 0.0, 0.0
        best_acc, best_metric = 0.0, 0.0
        best_epoch, best_model = -1, None
        self.logger.info(f"Start the training process for {self.params['epochs']} epochs")
        time_start = time.time()
        for epoch in tqdm(range(1, self.params['epochs'] + 1)):
            tm_ft = 0.0
            tm_adv = 0.0
    
            loss_tol, loss_finetune, loss_feed = 0.0, 0.0, 0.0
            loss_cls_, loss_adv_, loss_feed_ = AverageMeter('cls'), AverageMeter("adv"), AverageMeter("feed")
            tm_ei = time.time()
            if is_finetune:
                for _, (inputs, targets) in enumerate(self.dataloaders['retain']):
                    optimizer_finetune.zero_grad()
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    loss_finetune = 0.0
                    ## perturbation based robust training
                    if is_adv_loss and self.params['adv_lamb'] > 0.0:
                        # LDS should be calculated before the forward for cross entropy
                        loss_adv = adv_perturb_losser(model, inputs, targets)
                        loss_finetune += loss_adv * self.params['adv_lamb']
                        loss_adv_.update(loss_adv, inputs.size(0))
                    tm_adv += (time.time() - tm_ei)
                    
                    outputs = model(inputs)
                    loss_cls = cls_criterion(outputs, targets)
                    loss_cls_.update(loss_cls, inputs.size(0))
                    loss_finetune += loss_cls * cls_lamb
                    loss_finetune.backward(retain_graph=True) 
                    optimizer_finetune.step()
                    loss_tol += loss_finetune
                
                if scheduler_finetune is not None:
                    scheduler_finetune.step()
            tm_ft = (time.time() - tm_ei)
            
            tm_feed = 0.0
            if self.params['feed_lamb'] > 0.0:  # feedback_losser is not None and 
                for _, (inputs, targets) in enumerate(self.dataloaders['forget']):
                    loss_feed = 0.0
                    optimizer_feedback.zero_grad()
                    if class_forget is None or len(class_forget) <= 0:   ## for random subset unlearning
                        inputs_var, _ = self.online_transform(run_forget_aug, inputs, targets, self.num_classes)
                        pred_var = model(inputs_var.to(device))
                        pred_org = model(inputs.to(device))

                        if feedback_losser == 'softdis':
                            loss_soft = softdis_criterion(pred_org, pred_var, targets.to(device))
                        elif feedback_losser == 'kl':
                            loss_soft = F.kl_div(F.log_softmax(pred_org, dim=1), F.softmax(pred_var, dim=1), reduction='batchmean')
                        elif feedback_losser == 'l2':
                            loss_soft = torch.norm(pred_org - pred_var, p=2, dim=1).mean()
                        elif feedback_losser == 'cosine':
                            loss_soft = 1 - F.cosine_similarity(pred_org, pred_var, dim=-1).mean()
                        elif feedback_losser == 'mse':
                            loss_soft = F.mse_loss(pred_org, pred_var)
                        loss_feed += loss_soft
                    else:                                    ## for the class-wise unlearning
                        loss_reg = 0.0
                        pred_org = model(inputs.to(device))
                        probs = F.softmax(pred_org, dim=1)
                        
                        probs_tar = None
                        ##  using the prediction from the initial model (same architecture with the target model)
                        if init_model is not None: 
                            pred_tar = init_model(inputs.to(device))
                            probs_tar = F.softmax(pred_tar, dim=1)
                        else:
                            probs_tar = torch.ones_like(probs) / probs.size(1)
                        
                        if feedback_losser == 'max_entropy':
                            entropy = -torch.sum(probs * torch.log(probs_tar + 1e-10), dim=-1)
                            loss_reg = - 1.0 * entropy.mean()
                        elif feedback_losser == 'kl':
                            loss_reg = F.kl_div(probs, probs_tar, reduction='batchmean')
                        elif feedback_losser == 'mse':
                            loss_reg = F.mse_loss(probs, probs_tar)
                        elif feedback_losser == 'l2':
                            loss_reg = torch.norm(probs, p=2, dim=1).mean()
                        
                        loss_feed += loss_reg
                    loss_feed_.update(loss_feed, inputs.size(0))
                    loss_feed *= self.params['feed_lamb']
                    loss_feed.backward()  # retain_graph=True
                    loss_tol += loss_feed
                    optimizer_feedback.step()
                
                if scheduler_feedback is not None:
                    scheduler_feedback.step()
                tm_feed = time.time() - tm_ei - tm_ft
            
            self.logger.info(f"Epoch: {epoch}/{self.params['epochs']} / Time (sec): Epoch: {time.time() - tm_ei:.3f} | Finetune: {tm_ft:.3f} | " +
                                f" Feedback: {tm_feed:.3f} || Up-to-now: {cum_time + time.time() - time_start:.3f}")
            self.logger.info(f"Loss: CLS:{loss_cls_.avg:.4f}, Adv:{loss_adv_.avg:.4f}, Feed:{loss_feed_.avg:.4f} | Total:{loss_tol:.4f}")
            
            _, val_acc_ = test(model, self.dataloaders['valid'], cls_criterion, device)
            _, for_acc_ = test(model, self.dataloaders['forget'], cls_criterion, device)
            
            metric = val_acc_
            if self.params['feed_lamb'] > 0.0:  # feedback_losser is not None and
                metric = val_acc_ - np.abs(for_acc_ - val_acc_)
            
            if metric > best_metric:
                best_time = cum_time + time.time() - time_start
                _, test_acc = test(model, self.dataloaders['test'], cls_criterion, device)
                true_met = test_acc - np.abs(for_acc_ - test_acc)
                self.logger.info('==> Best Validation [previous => current]:' + 
                            f' acc-[{best_acc :.4f} => {val_acc_ :.4f}], metric-[{best_metric :.4f} => {metric :.4f}] ' +
                            f'TRUE-metric: {true_met:.4f} | Epoch: {epoch+1} @ {best_time:.3f} sec')
                best_model = deepcopy(model)
                best_metric = metric
                best_acc = val_acc_
                best_epoch = epoch
            cum_time += time.time() - time_start
            
            for key_ in ['train', 'retain', 'valid', 'test', 'forget']: # self.loader_dict.items():  #
                loader_ = self.dataloaders[key_]
                _, acc_ = test(model, loader_, cls_criterion, device)
                self.logger.info(f"Testing [ {key_:6} ] :  Acc: {acc_:.4f}")
            
            if self.params['save_checkpoints'] and (epoch % 10 == 0) and epoch > 0:
                outfn = f"{model_type}_{self._suffix_}_adv_{adv_perturb_losser}_fdbk_{feedback_losser}_ep-{epoch}.pt"
                out_path_ = path.join(self.path['model'], self.__get_name__(), outfn)
                save_model(model, out_path_)
            
            early_stopping(loss_finetune + loss_feed, model)  # err_valid
            if early_stopping.early_stop:
                self.logger.info("+++" * 10 + f"\t Early stopping at epoch: {epoch} \t" + "+++" * 10)
                break
            
        _, test_acc = test(best_model, self.dataloaders['test'], cls_criterion, device)
        self.logger.info(f"Best model test accuracy: {test_acc:.4f} @  {best_epoch} epoch: {best_time:.3f} sec")
        
        checkpoint = {
            'epoch': best_epoch,
            'best_acc': test_acc,
            'state_dict': best_model.state_dict(),
        }
        return best_model, checkpoint
            

class RobustUnlearn(UnlearnBasic):
    def __init__(self, logpath: str, logname: str, out_dir: str='results', out_name: str='model'):
        super(RobustUnlearn, self).__init__(logpath, logname, out_dir, out_name)

    def __get_name__(self):
        return 'RobUnlearn'
    
    def __indentify_informative_samples__(self, model, loader, top_k=5000, device='cpu'):
        model.eval()
        samples = list()
        uncertainties = list()
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                uncertainty = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                uncertainties.extend(uncertainty.cpu().numpy())
                samples.extend(list(zip(inputs.cpu().numpy(), labels.cpu().numpy())))
        
        informative_samples = [samples[i] for i in np.argsort(uncertainties)[-top_k:]]
        return informative_samples
    
    ## 0. using different protocal for learning:
    ##      finetune (FT), last-k layers retraining (LKL), etc
    ## 1. uisng different augmentation for unlearn: 
    ##       virtual adversarial training (VAT), adversarial training (FGSM), uncertainty-driven perturbation (UDP), etc
    ## 2. using different feedback for aligning the model:
    ##       contrastive loss, soft nearest neighbors loss, mmd loss, optimal transport loss, etc
    ##### USING THE SINGLE OPTIMIER AND SCHEDULER FOR THE TRAINING
    def unlearn_v2(self, model, protocal, adv_perturb_losser=None, feedback_losser=None, patience=10, loss='ce', 
                scheduler_option="", optimization='sgd', run_train_aug="none", run_forget_aug="none", 
                device='cpu', checkpoint_path='.', **kwargs):
        assert protocal in ['FT', 'LASTK'], f"Invalid protocal: {protocal}"
        self.logger.info(f"Unlearning with protocal: {protocal} | Adv-loss: {adv_perturb_losser} | Feedback-loss: {feedback_losser} | " 
                            f"| Optimizer: {optimization} | Train Aug: {run_train_aug} | Forget Aug: {run_forget_aug}")

        suffix = kwargs.get('suffix', '')
        model_type = kwargs.get('model_type', 'unknown')
        class_forget = kwargs.get('forget_classes', None)
        
        criterion_cls = loss_picker('ce')
        criterion = loss_picker("stce") if run_train_aug in ['mixup', 'cutmix'] else loss_picker(loss)
        
        if protocal == 'LASTK':
            # combine the last-k layer retraining with the fine-tuning as linear probing
            ## freeze the previous layers and retrain the last k layers, like linear probing
            last_k = kwargs.get('last_k', 1)
            re_init = kwargs.get('re_init', True)
            model = reset_final_layers(model, last_k, self.logger, re_init)
        else:
            model.train()

        num_to_forget = len(self.dataloaders['forget'].dataset)
        top_k = kwargs.get('top_k', 2 * num_to_forget)
        train_data = self.dataloaders['retain']
        if top_k < 0 or top_k >= len(self.dataloaders['retain'].dataset):
            top_k = len(train_data.dataset)
        num_train_batch = math.ceil(top_k / self.params['batch_size'])
        
        ## optimizer and scheduler for the model training
        optimizer = optimizer_picker(optimization, model.parameters(), lr=self.params['lr'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler = scheduler_picker(optimizer, scheduler_option, num_train_batch)
        
        ## start the training process for train data: finetune, last-k retrain
        ersp_fn = path.join(checkpoint_path, f'{self.__get_name__()}_{protocal}_{self._suffix_}_es_ckpt.pt')
        esp = EarlyStopping(patience=patience, verbose=True, delta=0, path=ersp_fn)
        
        best_epoch, best_model = -1, None
        cum_time, best_time = 0.0, 0.0
        best_acc, best_metric = 0.0, 0.0
        is_adv_loss = isinstance(adv_perturb_losser, AdvLoss)
        
        self.logger.info(f"Start the training process for {self.params['epochs']} epochs")
    
        for epoch in tqdm(range(1, self.params['epochs'] + 1)):
            loss_total = 0.0  ## total loss 
            l_cls_avg = 0.0  # consistency loss
            l_adv_avg = 0.0   ## adversarial loss
            l_feed_avg = 0.0  ## feedback loss
            l_tot_avg = 0.0  ## total loss
            time_adv = 0.0
            embs_, labs_ = defaultdict(list), defaultdict(list)
            
            # ## choose the informative samples from the retain set for speedup training
            # if top_k < len(self.loader_dict['retain'].dataset):
            #     # if epoch % 1 == 0:
            #     info_samples = self.__indentify_informative_samples__(model, self.loader_dict['retain'], top_k, device)
            #     info_set = ListToDataset(info_samples)
            #     train_data = DataLoader(info_set, self.params['batch_size'], shuffle=True)
            
            start_time = time.time()
            for _, (inputs, targets) in enumerate(train_data):
                t_ei_start = time.time()
                optimizer.zero_grad()
                
                ## applying in-running transform for data augmentation for the inputs, i.e., cutout or cutmix
                if run_train_aug in ["cutmix", "mixup"]:  # and run_train_aug != 'none': # 
                    inputs, targets = self.online_transform(run_train_aug, inputs, targets, self.num_classes)
                labs_['retain'].extend(targets)
                inputs, targets = inputs.to(device), targets.to(device)
                
                loss_adv = 0.0
                ## perturbation based robust training
                if is_adv_loss and self.params['adv_lamb'] > 0.0:
                    # LDS should be calculated before the forward for cross entropy
                    loss_adv = adv_perturb_losser(model, inputs, targets)
                    loss_total += loss_adv * self.params['adv_lamb']
                time_adv += (time.time() - t_ei_start)
                    
                outputs = model(inputs)
                loss_cls = criterion(outputs, targets)
                embs_['retain'].extend(outputs)
                loss_total += loss_cls
                
                l_adv_avg += loss_adv
                l_cls_avg += loss_cls.item()
                l_tot_avg += loss_total.item() / num_train_batch
            t_ft_adv = (time.time() - start_time)
            
            ## Applying the feedback from forget set with triplet loss
            t_feed_start = time.time()
            if feedback_losser is not None and self.params['feed_lamb'] > 0.0:
                ## extract feature and labels for the validation and forget samples
                for inputs, targets in self.dataloaders['valid']:
                    embs_['valid'].extend(model(inputs.to(device)))
                    labs_['valid'].extend(targets.cpu().numpy())
                
                for inputs, targets in self.dataloaders['forget']:
                    if run_forget_aug != None and run_forget_aug != 'none': #in ["cutmix", "mixup"]:
                        inputs, targets = self.online_transform(run_forget_aug, inputs, targets, self.num_classes)
                    embs_['forget'].extend(model(inputs.to(device)))
                    labs_['forget'].extend(targets.cpu().numpy())
                
                ## construct the contact features and labels for the triplet loss
                for key_ in ['retain', 'valid', 'forget']:
                    embs_[key_] = torch.stack(embs_[key_])
                    labs_[key_] = torch.tensor(labs_[key_], dtype=torch.long)
                
                if class_forget is None:
                    class_forget = len(np.intersect1d(np.unique(labs_['forget']), np.unique(labs_['retain']))) == 0

                loss_feed = 0.0
                # compute the alignment loss as feedback from forget set for the model
                if isinstance(feedback_losser, TripletLoss):
                    loss_feed = feedback_losser(embs_['retain'], labs_['retain'], embs_['valid'], 
                                            labs_['valid'], embs_['forget'], labs_['forget'])
                elif class_forget is not None:
                    ## maximizing the entroy for the forget samples
                    forget_probs = F.softmax(embs_['forget'], dim=1)
                    uniform_probs = torch.ones_like(forget_probs) / forget_probs.size(1)
                    
                    if feedback_losser == 'max_entropy':
                        entropy = -torch.sum(forget_probs * torch.log(forget_probs + 1e-10), dim=-1)
                        loss_feed = - 1.0 * entropy.mean()
                    elif feedback_losser == 'kl':
                        loss_feed = F.kl_div(forget_probs, uniform_probs, reduction='batchmean')
                    elif feedback_losser == 'mse':
                        loss_feed = F.mse_loss(forget_probs, uniform_probs)
                    elif feedback_losser == 'l2':
                        loss_feed = torch.norm(forget_probs, p=2, dim=1).mean()
                
                # loss_total += loss_feed * self.params['feed_lamb']
                l_feed_avg += loss_feed.item()
                l_tot_avg += loss_feed.item() / len(self.dataloaders['forget'])
            t_feed = (time.time() - t_feed_start)
            
            loss = loss_cls + loss_adv * self.params['adv_lamb'] + loss_feed  * self.params['feed_lamb']
            loss.backward()
            # loss_total.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            l_adv_avg /= num_train_batch
            l_feed_avg /= num_train_batch
            l_cls_avg /= num_train_batch
            
            self.logger.info(f"Epoch: {epoch+1}/{self.params['epochs']} |  Finetune+ADV: {t_ft_adv:.3f} sec | Feed: {t_feed:.3f} sec | " +
                                f"Up-to-now (Time): {cum_time + time.time() - start_time:.3f} sec")
            self.logger.info(f"Loss: clean: {l_cls_avg:.4f} | adv: {l_adv_avg:.4f} | feedback: {l_feed_avg:.4f} | Totoal: {loss_total:.4f}")
            
            _, val_acc_ = test(model, self.dataloaders['valid'], criterion_cls, device)
            _, for_acc_ = test(model, self.dataloaders['forget'], criterion_cls, device)
            
            metric = val_acc_
            if feedback_losser is not None and self.params['feed_lamb'] > 0.0:
                metric = val_acc_ - np.abs(for_acc_ - val_acc_) #* 0.5
            
            ## update the best model based on the metric
            if metric > best_metric:
                best_time = cum_time + time.time() - start_time
                _, test_acc = test(model, self.dataloaders['test'], criterion_cls, device)
                true_met = test_acc - np.abs(for_acc_ - test_acc)
                self.logger.info('==> Best Validation [previous => current]:' + 
                     f' acc-[{best_acc :.4f} => {val_acc_ :.4f}], metric-[{best_metric :.4f} => {metric :.4f}]  TRUE-metric: {true_met:.4f} | Epoch: {epoch+1} @ {best_time:.3f} sec')
                best_model = deepcopy(model)
                best_metric = metric
                best_acc = val_acc_
                best_epoch = epoch
            cum_time += time.time() - start_time
            
            ## evaluate the model performance over the test and forget set
            acc = dict()
            for key_ in ['train', 'retain', 'valid', 'test', 'forget']: # self.loader_dict.items():  #
                loader_ = self.dataloaders[key_]
                _, acc_ = test(model, loader_, criterion_cls, device)
                acc[key_] = acc_
                self.logger.info(f"Testing [ {key_:6} ] :  Acc: {acc_:.4f}")
            
            ## save the model checkpoints
            if self.params['save_checkpoints'] and (epoch % 10 == 0) and epoch > 0:
                outfn = f"{model_type}_{protocal}_{suffix}_adv_{adv_perturb_losser}_fdbk_{feedback_losser}_{self._suffix_}_ep-{epoch}.pt"
                out_path_ = path.join(self.path['model'], self.__get_name__(), outfn)
                save_model(model, out_path_)

            esp(loss_total + self.params['feed_lamb'] * l_feed_avg * num_train_batch, model)  # err_valid
            if esp.early_stop:
                self.logger.info("+++" * 10 + f"\t Early stopping at epoch: {epoch} \t" + "+++" * 10)
                break
        
        _, test_acc = test(best_model, self.dataloaders['test'], criterion_cls, device)
        self.logger.info(f"Best model test accuracy: {test_acc:.4f} @  {best_epoch} epoch: {best_time:.3f} sec")
        
        checkpoint = { 
            'epoch': best_epoch,
            'best_acc': test_acc,
            'protocal': protocal,
            'state_dict': best_model.state_dict(),
        }
        return best_model, checkpoint


    def unlearn(self, model, protocal, adv_perturb_losser=None, feedback_losser=None, patience=10, loss='ce', scheduler_option="", 
                optimization='sgd', run_train_aug="none", run_forget_aug="none", device='cpu', checkpoint_path='.', **kwargs):
        assert protocal in ['FT', 'LASTK'], f"Invalid protocal: {protocal}"
        self.logger.info(f"Unlearning with protocal: {protocal} | Adv-er: {adv_perturb_losser} | Feedback-er: {feedback_losser} | " 
                            f"| Optimizer: {optimization} | Train Aug: {run_train_aug} | Forget Aug: {run_forget_aug}")

        suffix = kwargs.get('suffix', '')
        cls_lamb = kwargs.get('classify_lamb', 1.0)
        model_type = kwargs.get('model_type', 'unknown')
        class_forget = kwargs.get('forget_classes', None)
        
        ## criterion for the model training and testing
        trcls_criterion = loss_picker("stce") if run_train_aug in ['mixup', 'cutmix'] else loss_picker(loss)
        tscls_criterion = loss_picker('ce')
        
        if protocal == 'LASTK':
            last_k = kwargs.get('last_k', 1)
            re_init = kwargs.get('re_init', True)
            # combine the last-k layer retraining with the fine-tuning as linear probing
            ## freeze the previous layers and retrain the last k layers, like linear probing
            model = reset_final_layers(model, last_k, self.logger, re_init)
        else:
            model.train()

        train_data = self.dataloaders['retain']
        num_to_forget = len(self.dataloaders['forget'].dataset)
        top_k = kwargs.get('top_k', 2 * num_to_forget)
        if top_k <= 0 or top_k >= len(self.dataloaders['retain'].dataset):
            top_k = len(train_data.dataset)
            num_train_batch = math.ceil(top_k / self.params['batch_size'])
        else:
            num_train_batch = len(train_data)
        
        ## optimizers and schedulers for the model training
        ## for finetune part w.r.t. retain set
        optimizer_finetune = optimizer_picker(optimization, model.parameters(), lr=self.params['lr'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler_finetune = scheduler_picker(optimizer_finetune, scheduler_option, num_train_batch)
        
        ## for feedback from forget set
        optimizer_feedback = optimizer_picker(optimization, model.parameters(), lr=self.params['lr_feedback'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler_feedback = scheduler_picker(optimizer_feedback, scheduler_option, len(self.dataloaders['forget']))
        
        # initialize the early_stopping object
        ersp_fn = os.path.join(checkpoint_path, f'{self.__get_name__()}_{protocal}_{self._suffix_}_es_ckpt.pt')
        earlystop = EarlyStopping(patience=patience, verbose=True, delta=0, path=ersp_fn)
        
        best_epoch, best_model = -1, None
        best_acc, best_metric = 0.0, 0.0
        cum_time, best_time = 0.0, 0.0
        is_adv_loss = isinstance(adv_perturb_losser, AdvLoss)
        
        ## start the training process
        self.logger.info(f"Start the training process for {self.params['epochs']} epochs")
        for epoch in tqdm(range(1, self.params['epochs'] + 1)):
            l_tol_ = 0.0
            l_cls, l_adv, l_feed = AverageMeter("loss_cls"), AverageMeter("loss_adv"), AverageMeter("loss_feed")
            
            ## restore the embeddings and labels for the triplet loss
            embs_, labs_ = defaultdict(list), defaultdict(list)
            
            # ## choose the informative samples from the retain set for speedup training
            # if top_k < len(self.loader_dict['retain'].dataset):
            #     # if epoch % 1 == 0:
            #     info_samples = self.__indentify_informative_samples__(model, self.loader_dict['retain'], top_k, device)
            #     info_set = ListToDataset(info_samples)
            #     train_data = DataLoader(info_set, self.params['batch_size'], shuffle=True)
            
            tm_ei = time.time()
            tm_adv, tm_feed = 0.0, 0.0
            for _, (inputs, targets) in enumerate(train_data):
                tm_i = time.time()
                optimizer_finetune.zero_grad()
                
                ## applying running transform for data augmentation for the inputs
                if run_train_aug != 'none' and run_train_aug is not None: #  #and run_train_aug in ["cutmix", "mixup"]:
                    inputs, targets = self.online_transform(run_train_aug, inputs, targets, self.num_classes)
                inputs, targets = inputs.to(device), targets.to(device)
                # embs_['retain'].extend(model(inputs))
                # labs_['retain'].extend(targets.cpu().numpy())
                
                loss_finetune = 0.0
                ## perturbation-based robust training
                if is_adv_loss and self.params['adv_lamb'] > 0.0:
                    # LDS should be calculated before the forward for cross entropy
                    l_adv_batch = adv_perturb_losser(model, inputs, targets)
                    l_adv.update(l_adv_batch.item(), inputs.size(0))
                    loss_finetune += l_adv_batch * self.params['adv_lamb']
                tm_adv += (time.time() - tm_i)
                
                outputs = model(inputs)
                loss_cls = trcls_criterion(outputs, targets)
                loss_finetune += loss_cls * cls_lamb
                loss_finetune.backward(retain_graph=True)
                optimizer_finetune.step()
                l_cls.update(loss_cls.item(), inputs.size(0))
                l_tol_ += loss_finetune.item()
            
            if scheduler_finetune is not None:
                scheduler_finetune.step()
            tm_ft_adv = (time.time() - tm_ei)
            
            loss_feed = 0.0
            tm_fd_start = time.time()
            if self.params['feed_lamb'] > 0.0:  # feedback_losser is not None and 
                optimizer_feedback.zero_grad()

                # ## extract feature and labels for the validation and forget samples
                # for inputs, targets in self.dataloaders['valid']:
                #     embs_['valid'].extend(model(inputs.to(device)))
                #     labs_['valid'].extend(targets.cpu().numpy())
                
                # for inputs, targets in self.dataloaders['forget']:
                #     if run_forget_aug != None and run_forget_aug != 'none': #in ["cutmix", "mixup"]:
                #         inputs, targets = self.online_transform(run_forget_aug, inputs, targets, self.num_classes)
                #         # ## TEST: treat the tranformed forget samples as the postive samples for the triplet loss w.r.t. the forget samples
                #         # embs_['valid'].extend(model(inputs.to(device)))                    
                #         # labs_['valid'].extend(targets.cpu().numpy())
                #     embs_['forget'].extend(model(inputs.to(device)))
                #     labs_['forget'].extend(targets.cpu().numpy())
                
                ## data preparation for the triplet loss
                for key_ in ['retain', 'valid', 'forget']:
                    aug_opt = run_forget_aug if key_ == 'forget' else "none"
                    for inputs, targets in self.dataloaders[key_]:
                        if aug_opt != "none" and aug_opt is not None: # and aug_opt in ["cutmix", "mixup"]:
                            inputs, targets = self.online_transform(aug_opt, inputs, targets, self.num_classes)
                        outputs = model(inputs.to(device))
                        labs_[key_].extend(targets.cpu().numpy())
                        embs_[key_].extend(outputs)
                
                for key_ in ['retain', 'valid', 'forget']:
                    embs_[key_] = torch.stack(embs_[key_])
                    labs_[key_] = torch.tensor(labs_[key_], dtype=torch.long)
                
                # if class_forget is None:
                #     class_forget = len(np.intersect1d(np.unique(labs_['forget']), np.unique(labs_['retain']))) == 0
               
                ## compute the alignment loss as feedback from the forget set for the model
                ## Applying the feedback from forget set with triplet loss
                if isinstance(feedback_losser, TripletLoss):
                    loss_feed = feedback_losser(embs_['retain'], labs_['retain'], embs_['valid'], labs_['valid'], embs_['forget'], labs_['forget'])
                else:
                    if class_forget:  ## only apply for the class-forget case.
                        ## maximizing the entroy for the forget samples
                        forget_probs = F.softmax(embs_['forget'], dim=1)
                        uniform_probs = torch.ones_like(forget_probs) / forget_probs.size(1)
                        
                        if feedback_losser == 'max_entropy':
                            entropy = -torch.sum(forget_probs * torch.log(forget_probs + 1e-10), dim=-1)
                            loss_feed = - 1.0 * entropy.mean()
                        elif feedback_losser == 'kl':
                            loss_feed = F.kl_div(forget_probs, uniform_probs, reduction='batchmean')
                        elif feedback_losser == 'mse':
                            loss_feed = F.mse_loss(forget_probs, uniform_probs)
                        elif feedback_losser == 'l2':
                            loss_feed = torch.norm(forget_probs, p=2, dim=1).mean()
                    
                loss_feed *= self.params['feed_lamb']
                loss_feed.backward()  # retain_graph=True
                optimizer_feedback.step()
                l_feed.update(loss_feed.item(), inputs.size(0))
                l_tol_ += loss_feed.item()
                tm_feed = (time.time() - tm_fd_start)
            
            if scheduler_feedback is not None:
                scheduler_feedback.step()
            
            ### logging the training process
            self.logger.info(f"Epoch: {epoch}/{self.params['epochs']} | Time (sec): Finetune+ADV: {tm_ft_adv:.3f} | Feed: {tm_feed:.3f} | " +
                                f"Up-to-now: {cum_time + time.time() - tm_ei:.3f}")
            self.logger.info(f"Loss: clean: {l_cls.avg:.4f} | adversarial: {l_adv.avg:.4f} | feedback: {l_feed.avg:.4f} | totoal: {l_tol_:.4f}")
            
            _, acc_val = test(model, self.dataloaders['valid'], tscls_criterion, device)
            _, acc_for = test(model, self.dataloaders['forget'], tscls_criterion, device)
            
            metric = acc_val
            if self.params['feed_lamb'] > 0.0:   # feedback_losser is not None and 
                metric = acc_val - np.abs(acc_for - acc_val)
            
            ## update the best model based on the metric
            if metric > best_metric:
                best_time = cum_time + time.time() - tm_ei
                _, acc_test = test(model, self.dataloaders['test'], tscls_criterion, device)
                true_met = acc_test - np.abs(acc_for - acc_test)
                best_model = deepcopy(model)
                best_metric = metric
                best_epoch, best_acc = epoch, acc_val
                self.logger.info('==> Best Validation [previous => current]:' + 
                     f' acc-[{best_acc :.4f} => {acc_val :.4f}], metric-[{best_metric :.4f} => {metric :.4f}] ' +
                     f'TRUE-metric: {true_met:.4f} | Epoch: {epoch} @ {best_time:.3f} sec')
            cum_time += time.time() - tm_ei
            
            ## evaluate the model performance over the test and forget set
            accs = dict()
            for key_ in ['train', 'retain', 'valid', 'test', 'forget']: # self.loader_dict.items():  #
                loader_ = self.dataloaders[key_]
                _, accs[key_] = test(model, loader_, tscls_criterion, device)
                self.logger.info(f"Testing [ {key_:6} ] :  Acc: {accs[key_]:.4f}")
            
            ## save the model checkpoints
            if self.params['save_checkpoints'] and (epoch % 10 == 0) and epoch > 0:
                outfn = f"{model_type}_{protocal}_{suffix}_adv_{adv_perturb_losser}_fdbk_{feedback_losser}_{self._suffix_}_ep-{epoch}.pt"
                out_path_ = path.join(self.path['model'], self.__get_name__(), outfn)
                save_model(model, out_path_)

            earlystop(loss_finetune + loss_feed, model)  # err_valid
            if earlystop.early_stop:
                self.logger.info("+++" * 10 + f"\t Early stopping at epoch: {epoch} \t" + "+++" * 10)
                break
        
        _, acc_test = test(best_model, self.dataloaders['test'], tscls_criterion, device)
        self.logger.info(f"Best model test accuracy: {acc_test:.4f} @  {best_epoch} epoch: {best_time:.3f} sec")
        
        checkpoint = { 
            'epoch': best_epoch,
            'best_acc': acc_test,
            'protocal': protocal,
            'state_dict': best_model.state_dict(),
        }
        return best_model, checkpoint



## Robust unlearning with the protocal of Gradient Ascending over forget set, or Finetune over random-relabeled forget set
## random relabeling (RL), gradient ascent (GA)
class RobustForget(UnlearnBasic):
    def __init__(self, logpath: str, logname: str, out_dir: str='results', out_name: str='model'):
        super(RobustForget, self).__init__(logpath, logname, out_dir, out_name)
    
    def __get_name__(self):
        return 'Rob_FG'
    
    @staticmethod
    def __get_random_label__(true_labels, num_classes):
        cand_classes = set(np.arange(num_classes))
        random_labels = list()
        for i in range(len(true_labels)):
            random_labels.append(np.random.choice(list(cand_classes - set([true_labels[i]]))))
        return random_labels
    
    def unlearn(self, model, protocal, finetune_over_retain=False, adv_perturb_losser=None, feedback_losser=None, 
                patience=20, loss='ce', scheduler_option="", optimization='sgd', run_train_aug="none",
                run_forget_aug="none", device='cpu', checkpoint_path='.', **kwargs):
        assert protocal in ['RL', 'GA'], f"Invalid protocal {protocal}"
        self.logger.info(f"Unlearning with protocal: {protocal} | Adv: {adv_perturb_losser} | Feedback: {feedback_losser} | " 
                            f"| Optimizer: {optimization} | Train Aug: {run_train_aug} | Forget Aug: {run_forget_aug}")

        sfx = kwargs.get('suffix', '')
        cls_lamb = kwargs.get('classify_lamb', 1.0)
        model_type = kwargs.get('model_type', 'unknown')
        ## construct the train dataset
        train_data = None
        if protocal == 'RL':
            ## random relabel the forget set and finetune over the new set
            flatten_data = flatten_subset(deepcopy(self.dataloaders['forget'].dataset))
            new_labels = self.__get_random_label__(flatten_data.targets, self.num_classes)
            forget_set = RelabeledDataset(flatten_data.dataset, flatten_data.targets, new_labels)
            train_data = DataLoader(forget_set, batch_size=self.params['batch_size'], shuffle=True)
        elif protocal == 'GA':
            ## using the retain set for the training
            train_data = self.dataloaders['forget']
            
        num_data = len(train_data)
        optimizer = optimizer_picker(optimization, model.parameters(), lr=self.params['lr'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler = scheduler_picker(optimizer, num_data, scheduler_option)
        
        tscls_criterion = loss_picker('ce')
        trcls_criterion = loss_picker("stce") if run_forget_aug in ['mixup', 'cutmix'] else loss_picker(loss)            
        
        ersp_fn = path.join(checkpoint_path, f'{self.__get_name__()}_{protocal}_{self._suffix_}_es_ckpt.pt')
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0, path=ersp_fn)
        
        best_epoch, best_model = -1, None
        best_acc, best_metric = 0.0, 0.0
        cum_time, best_time = 0.0, 0.0
        is_adv_loss = isinstance(adv_perturb_losser, AdvLoss)
        
        self.logger.info(f"Start the training process for {self.params['epochs']} epochs")
        tm_ti = time.time()
        for epoch in tqdm(range(self.params['epochs'])):
            l_tol_, l_cls_ = 0.0, AverageMeter("loss_cls")
            l_adv_ = AverageMeter("loss_adv")
            
            ## restore the embeddings and labels for the triplet loss
            embs_, labs_ = defaultdict(list), defaultdict(list)
            
            tm_start = time.time()
            ## Finetune over forget set RL-forget (random label) or GA-forget  (gradient ascent)
            for _, (inputs, targets) in enumerate(train_data):
                optimizer.zero_grad()
                if run_forget_aug is not None and run_forget_aug != "none": #or len(run_forget_aug) > 0) and run_forget_aug in ["cutmix", "mixup"]:
                    inputs, targets = self.online_transform(run_forget_aug, inputs, targets, self.num_classes)
                labs_['forget'].extend(targets.cpu().numpy())
                inputs, targets = inputs.to(device), targets.to(device)
                
                loss_adv_ = 0.0
                loss_comb = 0.0
                ## perturbation based robust training
                if is_adv_loss and self.params['adv_lamb'] > 0.0:
                    # LDS should be calculated before the forward for cross entropy
                    loss_adv_ = adv_perturb_losser(model, inputs, targets)
                    loss_comb += loss_adv_ * self.params['adv_lamb']
                    l_adv_.update(loss_adv_, inputs.size(0))
                
                outputs = model(inputs)
                embs_['forget'].extend(outputs)
                loss_cls = trcls_criterion(outputs, targets)
                loss_comb += loss_cls * cls_lamb
                if protocal == 'GA':   # gradient ascent over forget set
                    loss_comb *= -1
                loss_comb.backward(retain_graph=True)
                optimizer.step()
                l_cls_.update(loss_cls.item(), inputs.size(0))
                l_tol_ += loss_comb.item()
            
            ## Finetune over retain set
            if finetune_over_retain:
                for _, (inputs, targets) in enumerate(self.dataloaders['retain']):
                    optimizer.zero_grad()
                    if (run_train_aug != None or len(run_train_aug) > 0): # and run_train_aug in ["cutmix", "mixup"]:
                        inputs, targets = self.online_transform(run_train_aug, inputs, targets, self.num_classes)
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    rt_adv_ = 0.0
                    rt_l_comb = 0.0
                    ## perturbation based robust training
                    if is_adv_loss and self.params['adv_lamb'] > 0.0:
                        # LDS should be calculated before the forward for cross entropy
                        rt_adv_ = adv_perturb_losser(model, inputs, targets)
                        rt_l_comb += self.params['adv_lamb'] * rt_adv_
                        l_adv_.update(rt_adv_, inputs.size(0))
                    
                    outputs = model(inputs)
                    rt_loss_clean = trcls_criterion(outputs, targets)
                    rt_l_comb += rt_loss_clean
                    rt_l_comb.backward(retain_graph=True)
                    optimizer.step()
                    l_tol_ += rt_l_comb.item()
                        
            ## Applying the feedback from forget set with triplet loss
            scaler = 1.0
            if self.params['dynamic_weight']:
                scaler = (epoch + 1) * 1.5 / self.params['epochs']
            t_fed_start = time.time()
            self._set_lr_(optimizer, self.params['lr_feedback'])
            if feedback_losser is not None and self.params['feed_lamb'] > 0.0:
                for key_ in ['retain', 'valid']:
                    for inputs, targets in self.dataloaders[key_]:
                        optimizer.zero_grad()
                        labs_[key_].extend(targets.cpu().numpy())
                        outputs = model(inputs.to(device))
                        embs_[key_].extend(outputs)
                
                for key_ in ['retain', 'valid', 'forget']:
                    embs_[key_] = torch.stack(embs_[key_])
                    labs_[key_] = torch.tensor(labs_[key_], dtype=torch.long)
                
                feed_avg = 0.0
                # compute the alignment loss as feedback from the forget set for the model
                if isinstance(feedback_losser, TripletLoss):
                    u_fed = feedback_losser(embs_['retain'], labs_['retain'], embs_['valid'], 
                                          labs_['valid'], embs_['forget'], labs_['forget'])
                    u_fed *= self.params['feed_lamb'] * scaler
                    feed_avg += u_fed.item()
                    u_fed.backward()  # retain_graph=True
                    l_tol_ += u_fed.item()
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            ## revert the lr back
            self._set_lr_(optimizer, self.params['lr'])
            
            ## time in minutes and average the losses
            tm_feed = (time.time() - t_fed_start)
            tm_tol = (time.time() - tm_start)
            
            self.logger.info(f"Epoch: {epoch+1}/{self.params['epochs']} |  Time: {tm_tol:.3f} sec | Feed: {tm_feed:.3f} sec")
            self.logger.info(f"Loss: | Feedback: {feed_avg:.4f} | Totoal: {l_tol_:.4f}")
            
            _, acc_val = test(model, self.dataloaders['valid'], tscls_criterion, device)
            _, acc_for = test(model, self.dataloaders['forget'], tscls_criterion, device)
            metric = acc_val
            if feedback_losser is not None and self.params['feed_lamb'] > 0.0:
                metric = acc_val - np.abs(acc_for - acc_val)
            
            ## save the best model based on the validation accuracy
            if metric > best_metric:
                best_time = cum_time + time.time() - tm_ti
                _, acc_test = test(model, self.dataloaders['test'], tscls_criterion, device)
                true_met = acc_test - np.abs(acc_for - acc_test)
                self.logger.info(f'==> Best Validation: [previous => current]: acc-[{best_acc:.4f} => {acc_val:.4f}], metric-[{best_metric :.4f} => {metric :.4f}] ' +
                             f'TRUE-metric: {true_met:.4f} |Epoch: {epoch} @ {best_time:.3f} sec')
                best_metric =   metric
                best_epoch, best_acc = epoch, acc_val
                best_model = deepcopy(model)
            cum_time += time.time() - tm_ti
            
            ## evaluate the model performance over the test and forget set
            accs = dict()
            for key_ in ['train', 'retain', 'valid', 'test', 'forget']: # self.loader_dict.items():  #
                loader_ = self.dataloaders[key_]
                _, accs[key_] = test(model, loader_, tscls_criterion, device)
                self.logger.info(f"Testing [ {key_:6} ] :  Acc: {accs[key_]:.4f}")

            ## save the model checkpoints
            if self.params['save_checkpoints'] and (epoch % 10 == 0) and epoch > 0:
                outfn = f"{sfx}_{model_type}_{protocal}_adv_{adv_perturb_losser}_feedback_{feedback_losser}_{self._suffix_}_ep-{epoch}.pth"
                out_path_ = path.join(self.path['model'], self.__get_name__(), outfn)
                save_model(model, out_path_)

            early_stopping(loss_comb + self.params['feed_lamb'] * feed_avg * num_data, model)  # err_valid
            if early_stopping.early_stop:
                self.logger.info("+++" * 10 + f"\t Early stopping at epoch: {epoch} \t" + "+++" * 10)
                break
        
        _, acc_test = test(best_model, self.dataloaders['test'], tscls_criterion, device)
        self.logger.info(f"==============> Best model test accuracy: {acc_test:.4f} @  {best_epoch} epoch: {best_time:.3f} sec  <==============")
        
        checkpoint = {
            'epoch': -1,
            'state_dict': best_model.state_dict(),
            'best_acc': acc_test,
            'protocal': protocal,
        }
        return best_model, checkpoint


## Distill retain data with softlabel and train a same-architecture model as the original model
class DistillUnlearn(UnlearnBasic):
    def __init__(self, logpath: str, logname: str, out_dir: str='results', out_name: str='model'):
        super(DistillUnlearn, self).__init__(logpath, logname, out_dir, out_name)
        # self.alpha = alpha
        # self.T = temperature
    
    def __get_name__(self):
        return 'Rob_Dist'
    
    @staticmethod
    def init_student_model(teacher, student, layers_to_copy):
        teacher_dict = teacher.state_dict()
        student_dict = student.state_dict()
        
        for name, param in teacher_dict.items():
            if any(layer in name for layer in layers_to_copy):
                if name in student_dict:
                    student_dict[name].data.copy_(param.data)
                else:
                    print(f"Missing key: {name} in the student model")
        student.load_state_dict(student_dict)
    
    def preprocess(self, teacher_model, prune_amount=0.1, device='cuda'):
        ## prunning
        import torch.nn.utils.prune as prune
        self.logger.info("Prunning the teacher model for distillation")
        for name, module in teacher_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=prune_amount, n=2, dim=0)

        ## calibration step
        teacher_model.cpu()
        teacher_model.eval()
        teacher_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(teacher_model, inplace=True)
        
        self.logger.info("Calibrating the teacher model for quantization")
        with torch.no_grad():
            for i, (inputs, _) in enumerate(self.dataloaders['valid']):
                if i >= 10:         # Calibrate using a subset of data
                    break
                teacher_model(inputs)
        
        torch.quantization.convert(teacher_model, inplace=True)
        self.logger.info("Calibrated completed!")
        teacher_model.to(device)
        return teacher_model
        
    
    def unlearn(self, teacher_model, model, adv_perturb_losser=None, feedback_losser=None, patience=20, # correct_senstive=True,  
                scheduler_option="", optimization='sgd', run_train_aug="none", run_forget_aug="none", device='cpu', checkpoint_path='.', **kwargs):
        protocal = 'distill'
        self.logger.info(f"Unlearning with distillation model | Adv: {adv_perturb_losser} | Feedback: {feedback_losser} | " 
                            f"| Optimizer: {optimization} | Train Aug: {run_train_aug} | Forget Aug: {run_forget_aug} | Scheduler: {scheduler_option}")
        
        model_type = kwargs.get('model_type', 'unknown')
        sfx = kwargs.get('suffix', '')
        alpha = kwargs.get('alpha', 0.5)
        T = kwargs.get('temperature', 3.0)
        ## construct the train dataset with the soft labels from the teacher model
        # dataset = self._construct_softlabel_dataset_(teacher_model, device, correct_senstive)
        # train_loader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)
        # num_data = len(train_loader.dataset)
        
        teacher_model.eval()
        model.to(device)
        model.train()
        train_data = self.dataloaders['retain']
        num_data = len(train_data)
        
        optimizer = optimizer_picker(optimization, model.parameters(), lr=self.params['lr'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler = scheduler_picker(optimizer, num_data, scheduler_option)
        test_creiterion = loss_picker('ce')
        self.logger.info(f"Distillation with alpha: {alpha} | Temperature: {T}")
        criterion = DistillationLoss(alpha=alpha, temperature=T)
        # criterion = torch.nn.CrossEntropyLoss() #loss_picker('kldiv')

        ## start the training process for train data: finetune, last-k retrain
        ## initialize the early_stopping object
        ersp_fn = path.join(checkpoint_path, f'{self.__get_name__()}_{protocal}_{self._suffix_}_es_ckpt.pt')
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0, path=ersp_fn)
        
        best_acc = 0.0
        best_metric, best_iter = 0.0, -1
        best_model = None
        is_adv_loss = isinstance(adv_perturb_losser, AdvLoss)
        self.logger.info(f"Start the training process for {self.params['epochs']} epochs")
        cum_time, best_time = 0.0, 0.0
        for epoch in tqdm(range(self.params['epochs'])):
            start_time = time.time()
            loss_avg = 0.0          ## total loss
            adv_avg = 0.0           ## adversarial loss
            feed_avg = 0.0          ## feedback loss
            clean_loss_avg = 0.0    ## consistency loss
            embs_, labs_ = defaultdict(list), defaultdict(list)
            
            t_start = time.time()
            for _, (inputs, labels) in enumerate(train_data):
                optimizer.zero_grad()
                if (run_train_aug != None or len(run_train_aug) > 0) and run_train_aug in ["cutmix", "mixup"]:
                    # inputs, labels = self.online_transform(run_train_aug, inputs, labels, self.num_classes)
                    inputs, targets = self._onlinemix_transform_(inputs, targets, run_train_aug)
                labs_['retain'].extend(labels)  # torch.argmax(labels, dim=1).cpu().numpy()
                inputs, labels = inputs.to(device), labels.to(device)
                
                loss_adv_ = 0.0
                loss_comb = 0.0
                ## perturbation based robust training
                if is_adv_loss and self.params['adv_lamb'] > 0.0:
                    # LDS should be calculated before the forward for cross entropy
                    loss_adv_ = adv_perturb_losser(model, inputs, labels)
                    loss_comb += self.params['adv_lamb'] * loss_adv_
                
                ## normal distillation from the teacher model to the student model
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                student_logits = model(inputs)
                embs_['retain'].extend(student_logits)
                loss_dist_ = criterion(student_logits, teacher_logits, labels)
                
                # outputs = model(inputs.to(device))
                # embs_['retain'].extend(outputs)
                # outputs = torch.log_softmax(outputs, dim=1)
                # loss_clean = criterion(outputs, labels.to(device))
                
                loss_comb += loss_dist_
                loss_comb.backward(retain_graph=True)
                optimizer.step()

                adv_avg += loss_adv_ * self.params['adv_lamb']
                clean_loss_avg += loss_dist_.item()
                loss_avg += loss_comb.item()
            
            ## Applying the feedback from the forget set with triplet loss
            scaler = 1.0
            if self.params['dynamic_weight']:
                scaler = (epoch + 1) * 1.5 / self.params['epochs']
            t_fed_start = time.time()
            if feedback_losser is not None and self.params['feed_lamb'] > 0.0:
                self._set_lr_(optimizer, self.params['lr_feedback'])
                for key_ in ['valid', 'forget']:
                    aug_option = run_forget_aug if key_ == 'forget' else "none"
                    for inputs, targets in self.dataloaders[key_]:
                        optimizer.zero_grad()
                        if aug_option != None and aug_option in ["cutmix", "mixup"]:
                            inputs, targets = self.online_transform(aug_option, inputs, targets, self.num_classes)
                            # inputs, targets = self._online_transform_(inputs, targets, aug_option)
                        labs_[key_].extend(targets.cpu().numpy())
                        outputs = model(inputs.to(device))
                        embs_[key_].extend(outputs)
                
                for key_ in ['retain', 'valid', 'forget']:
                    embs_[key_] = torch.stack(embs_[key_])
                    labs_[key_] = torch.tensor(labs_[key_], dtype=torch.long)
                
                # compute the alignment loss as feedback from the forget set for the model
                if isinstance(feedback_losser, TripletLoss):
                    u_fed = feedback_losser(embs_['retain'], labs_['retain'], embs_['valid'], labs_['valid'], embs_['forget'], labs_['forget'])
                    u_fed = self.params['feed_lamb'] * u_fed * scaler
                    feed_avg += u_fed.item()
                    u_fed.backward()  # retain_graph=True
                optimizer.step()
                loss_avg += feed_avg
            if scheduler is not None:
                scheduler.step()
            
            self._set_lr_(optimizer, self.params['lr'])
            t_fed = (time.time() - t_fed_start)
            t_all = (time.time() - t_start)
            adv_avg /= num_data
            feed_avg /= num_data
            clean_loss_avg /= num_data
            loss_avg /= num_data
            
            self.logger.info(f"Epoch: {epoch+1}/{self.params['epochs']} |  Time: {t_all:.3f} sec | Feed: {t_fed:.3f} sec")
            self.logger.info(f"Loss: clean: {clean_loss_avg:.4f} | Adv: {adv_avg:.4f} | Feedback: {feed_avg:.4f} | Totoal: {loss_avg:.4f}")
            
            _, val_acc_ = test(model, self.dataloaders['valid'], test_creiterion, device)
            # _, for_acc_ = test(model, self.loader_dict['forget'], test_creiterion, device)
            ## save the best model based on the validation accuracy
            metric = val_acc_
            # if feedback_loss is not None and self.params['feed_lamb'] > 0.0:
            #     metric = val_acc_ - torch.abs(for_acc_ - val_acc_) #* 0.5
            
            # if val_acc_ > best_acc:
            if metric >= best_metric:
                best_time = cum_time + time.time() - start_time
                self.logger.info(f'==> Best Validation [previous => current]: acc-[{best_acc :.4f} => {val_acc_ :.4f}] | ' + 
                                 f'Epoch: {epoch+1} @ {best_time:.3f} sec') # metric-[{best_metric :.4f} => {metric :.4f}]
                best_acc = val_acc_
                best_metric = metric
                best_iter = epoch
                best_time = cum_time + time.time() - start_time
                best_model = deepcopy(model)  
            
            cum_time += time.time() - start_time
                  
            ## evaluate the model performance over the test and forget set
            val_acc_ = 0.0
            for key_, loader_ in self.dataloaders.items():
                _, acc_ = test(model, loader_, test_creiterion, device)
                self.logger.info(f"Testing [ {key_:6} ] :  Acc: {acc_:.4f}")
                # acc_, auc_ = self._model_evaluate_(model, loader_, device)
                # self.logger.info(f"Testing [ {key_:6} ]: Acc: {acc_:.4f} | AUC: {auc_:.4f}")
                # _, acc_, acc_cls_ = test(model, loader_, device, criterion, transforms.ToTensor())
                # self.logger.info(f"Testing [ {key_:6} ]: Acc: {acc_:.4f} | Class Acc: {dict_to_str(acc_cls_)}")
                if key_ == 'valid':
                    val_acc_ = acc_
                
            ## save the model checkpoints
            if self.params['save_checkpoints'] and (epoch % 10 == 0) and epoch > 0:
                outfn = f"{model_type}_{protocal}_{sfx}_adv_{adv_perturb_losser}_feedback_{feedback_losser}_{self._suffix_}_ep-{epoch}.pth"
                out_path_ = path.join(self.path['model'], self.__get_name__(), outfn)
                save_model(model, out_path_)

            early_stopping(loss_comb + self.params['feed_lamb'] * feed_avg * num_data, model)  # err_valid
            if early_stopping.early_stop:
                self.logger.info("+++" * 10 + f"\t Early stopping at epoch: {epoch} \t" + "+++" * 10)
                break
            
        # outfn = f"{model_type}_{protocal}_{sfx}_adv_{adv_perturb_loss}_feedback_{feedback_loss}_{self._suffix_}_best.pt"
        # out_path_ = path.join(self.path['model'], self.__get_name__(), outfn)
        # save_model(best_model, out_path_)
        
        _, test_acc = test(best_model, self.dataloaders['test'], test_creiterion, device)
        self.logger.info(f"Best model test accuracy: {test_acc:.4f} @  {best_iter} epoch: {best_time:.3f} sec")
        # test_loader = DataLoader(self.loader_dict['test'], batch_size=self.params['batch_size'], shuffle=False)
        # _, test_acc, test_acc_cls_ = test(best_model, test_loader, loss_picker('ce'), device)
        # self.logger.info(f"Best model test accuracy: {test_acc:.4f} | Class Acc: {dict_to_str(test_acc_cls_)}")
        checkpoint = {
            'epoch': -1,
            'state_dict': best_model.state_dict(),
            'best_acc': test_acc,
            'protocal': protocal,
        }
        return best_model, checkpoint

    
    
       
    # def _construct_softlabel_dataset_(self, model, device='cpu', correct_senstive=True, smoothing=0.15):
    #     def _sparse_and_normalize_probs_(probabilities, threshold=0.1):
    #         probs = probabilities.clone()
    #         probs[probs < threshold] = 0.0
    #         probs = probs / torch.sum(probs, dim=1, keepdim=True)
    #         probs[torch.isnan(probs)] = 0.0
    #         return probs
    
    #     def _smooth_one_hot_(true_labels, num_classes, smoothing=0.1):
    #         confidence = 1.0 - smoothing
    #         label_shape = torch.Size((true_labels.size(0), num_classes))
    #         with torch.no_grad():
    #             true_dist = torch.empty(size=label_shape, device=true_labels.device)
    #             true_dist.fill_(smoothing / (num_classes - 1))
    #             true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    #         return true_dist    
    #         assert 0 <= smoothing < 1, "smoothing must be in range [0, 1)"
        
    #     ins = list()
    #     soft_labels = list()
    #     model.eval()
    #     with torch.no_grad():
    #         for _, (inputs, targets) in enumerate(self.dataset['retain']):
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             outs = model(inputs)
    #             probs = torch.softmax(outs, dim=1)
    #             _, preds = torch.max(outs, dim=1)
                
    #             probs_mod = _sparse_and_normalize_probs_(probs, threshold=0.1)
    #             if correct_senstive:
    #                 ## correct sensitive for the label construction
    #                 adjust_idxs = torch.where(preds != targets)[0]
    #                 for idx in adjust_idxs:
    #                     smoothed_label = _smooth_one_hot_(targets[idx:idx+1], self.num_classes, smoothing)
    #                     probs_mod[idx] = smoothed_label.squeeze()
    #             soft_labels.append(probs_mod.cpu().numpy())
    #             ins.append(inputs.cpu().numpy())
                
    #         ins = np.concatenate(ins, axis=0)
    #         soft_labels = np.concatenate(soft_labels, axis=0)
        
    #     return SoftLabelDataset(ins, soft_labels)


## Other unlearn model implementations with different types of feedback from the attacker.
### calculate feedback based on the attack model's prediction results.
### refer to the paper: "Adversarial Machine Unlearning: A Stackelberg Game Approach"
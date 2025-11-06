import os
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader

from data_tool import CustomerDataset
from logger import create_logger
from early_stopping import EarlyStopping
from utils import AverageMeter, cutmix_data, get_accuracy, mixup_data, save_model

from transforms import Transforms
from losses import AdvLoss, TripletLoss
from learner import DistillationLoss, loss_picker, scheduler_picker, optimizer_picker, test, l1_regularization, l2_regularization


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
    
    def set_params(self,
                epochs: int = 10,
                batch_size: int = 32,
                lr: float = 0.01,
                lr_feedback: float = 0.001,
                rob_reg: float = 1.0,
                strength: float = 0.1,
                weight_decay: float = 5e-4,
                momentum: float = 0.9,
                save_checkpoints: bool = False,
                class_wise: bool = False,
                seed: int = 42,
                dynamic_weights: bool = False,
                no_reg_epochs: int = 0,
                regularizer: str = 'none',
                gamma: float = 5e-4,
                dynamic_regularization: bool = False,
                ):
        self.logger.info(f"Set the parameters for the unlearning process")
        self.params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'lr_feedback': lr_feedback,
            'rob_reg': rob_reg,
            'strength': strength,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'save_checkpoints': save_checkpoints,
            'class_wise': class_wise,
            'seed': seed,
            'dynamic_weight': dynamic_weights,
            'regularizer': regularizer,
            'no_reg_epochs': no_reg_epochs,
            'gamma': gamma,
            'dynreg': dynamic_regularization,
        }
        self._suffix_ =  f"epochs-{epochs}_lr-{lr}_lr_feedback-{lr_feedback}_clswise-{class_wise}_advreg-{rob_reg}_feedreg-{strength}_seed-{seed}_regular-{regularizer}_noregepoch-{no_reg_epochs}_gamma-{gamma}_dynreg-{dynamic_regularization}"
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


## Distill retain data with softlabel and train a same-architecture model as the original model
class DistillUnlearn(UnlearnBasic):
    def __init__(self, logpath: str, logname: str, out_dir: str='results', out_name: str='model'):
        super(DistillUnlearn, self).__init__(logpath, logname, out_dir, out_name)

    def __get_name__(self):
        return 'DisUn'
    
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
            for i, (inputs, _) in enumerate(self.loader_dict['valid']):
                if i >= 10:         # Calibrate using a subset of data
                    break
                teacher_model(inputs)
        
        torch.quantization.convert(teacher_model, inplace=True)
        self.logger.info("Calibrated completed!")
        teacher_model.to(device)
        return teacher_model
    
    def construct_softlabel_dataset(self, teacher_model, data_loader, device='cuda'):
        ins = list()
        true_labels = list()
        predicts = list()
        teacher_model.eval()
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(data_loader):
                ins.extend(inputs)
                true_labels.extend(targets)
                outs = teacher_model(inputs.to(device))
                predicts.extend(outs.cpu().numpy())
        dataset = CustomerDataset(ins, np.array(true_labels), np.array(predicts))
        return dataset
    

    def unlearn_sequential(self, model, train_dataloader, adv_perturb_loss=None, patience=20,
                           scheduler_option="", optimization='sgd', run_train_aug="none", device='cpu', checkpoint_path='.', **kwargs):
        protocal = 'distill'
        self.logger.info(f"Unlearning with distillation model | Adv: {adv_perturb_loss} | " 
                            f"| Optimizer: {optimization} | Train Aug: {run_train_aug} | Scheduler: {scheduler_option}")
        
        model_type = kwargs.get('model_type', 'unknown')
        sfx = kwargs.get('suffix', '')
        alpha = kwargs.get('alpha', 0.5)
        T = kwargs.get('temperature', 3.0)
        
        model = model.to(device)
        model.train()
        
        num_data = len(train_dataloader)
        optimizer = optimizer_picker(optimization, model.parameters(), lr=self.params['lr'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler = scheduler_picker(optimizer, scheduler_option, num_data, self.params['lr'], self.params['weight_decay'], self.params['epochs'])
        test_creiterion = loss_picker('ce')
        self.logger.info(f"Distillation with alpha: {alpha} | Temperature: {T}")
        criterion = DistillationLoss(alpha=alpha, temperature=T)
        
        ersp_fn = os.path.join(checkpoint_path, f'{self.__get_name__()}_{protocal}_{self._suffix_}_es_ckpt.pt')
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0, path=ersp_fn)
        
        best_acc = 0.0
        best_metric, best_iter = 0.0, -1
        best_model = None
        is_adv_loss = isinstance(adv_perturb_loss, AdvLoss)
        self.logger.info(f"Start the training process for {self.params['epochs']} epochs")
        self.logger.info(f"Total data: {num_data} | Batch size: {self.params['batch_size']}")
        cum_time, best_time = 0.0, 0.0
        for epoch in tqdm(range(self.params['epochs'])):
            start_time = time.time()
            loss_avg = 0.0          ## total loss
            adv_avg = 0.0           ## adversarial loss
            feed_avg = 0.0          ## feedback loss
            clean_loss_avg = 0.0    ## consistency loss
            
            t_start = time.time()
            for id, (inputs, labels, t_logits) in enumerate(train_dataloader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                t_logits = torch.tensor(t_logits).to(device)
                
                loss_adv_ = 0.0
                loss_comb = 0.0

                ## perturbation based robust training
                if is_adv_loss and self.params['rob_reg'] > 0.0:
                    # LDS should be calculated before the forward for cross entropy
                    loss_adv_ = adv_perturb_loss(model, inputs, labels)
                    loss_comb += self.params['rob_reg'] * loss_adv_
                
                student_logits = model(inputs)
                # print("student prediction done:", student_logits.shape)
                # embs_['retain'].extend(student_logits)
                loss_dist_ = criterion(student_logits, t_logits, labels)
                
                loss_comb += loss_dist_
                loss_comb.backward(retain_graph=True)
                optimizer.step()

                adv_avg += loss_adv_ * self.params['rob_reg']
                clean_loss_avg += loss_dist_.item()
                loss_avg += loss_comb.item()
                
            if scheduler is not None:
                scheduler.step()
            
            # print("compute time & loss:")
            self._set_lr_(optimizer, self.params['lr'])
            t_all = (time.time() - t_start)
            adv_avg /= num_data
            feed_avg /= num_data
            clean_loss_avg /= num_data
            loss_avg /= num_data
            
            self.logger.info(f"Epoch: {epoch+1}/{self.params['epochs']} |  Time: {t_all:.3f} sec")
            self.logger.info(f"Loss: clean: {clean_loss_avg:.4f} | Adv: {adv_avg:.4f} | Feedback: {feed_avg:.4f} | Totoal: {loss_avg:.4f}")
            
            _, val_acc_ = test(model, self.loader_dict['valid'], test_creiterion, device)
            ## save the best model based on the validation accuracy
            metric = val_acc_
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
            for key_, loader_ in self.loader_dict.items():
                _, acc_ = test(model, loader_, test_creiterion, device)
                self.logger.info(f"Testing [ {key_:6} ] :  Acc: {acc_:.4f}")
                
            ## save the model checkpoints
            if self.params['save_checkpoints'] and (epoch % 10 == 0) and epoch > 0:
                outfn = f"{model_type}_{protocal}_{sfx}_adv_{adv_perturb_loss}_feedback_None_{self._suffix_}_ep-{epoch}.pth"
                out_path_ = os.path.join(self.path['model'], self.__get_name__(), outfn)
                save_model(model, out_path_)

            early_stopping(loss_comb + self.params['strength'] * feed_avg * num_data, model)  # err_valid
            if early_stopping.early_stop:
                self.logger.info("+++" * 10 + f"\t Early stopping at epoch: {epoch} \t" + "+++" * 10)
                break
        
        _, test_acc = test(best_model, self.loader_dict['test'], test_creiterion, device)
        self.logger.info(f"Best model test accuracy: {test_acc:.4f} @  {best_iter} epoch: {best_time:.3f} sec")
        checkpoint = {
            'epoch': -1,
            'state_dict': best_model.state_dict(),
            'best_acc': test_acc,
            'protocal': protocal,
        }
        return best_model, checkpoint

    
    def model_prune(self, model, device='cuda'):
        pass
    
    
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
    

    def unlearn(self, teacher_model, model, adv_perturb_loss=None, feedback_loss=None, patience=20, # correct_senstive=True,  
                scheduler_option="", optimization='sgd', run_train_aug="none", run_forget_aug="none", 
                device='cpu', device_teacher='cpu', checkpoint_path='.', **kwargs):
        protocal = 'distill'
        self.logger.info(f"Unlearning with distillation model | Adv: {adv_perturb_loss} | Feedback: {feedback_loss} | " 
                            f"| Optimizer: {optimization} | Train Aug: {run_train_aug} | Forget Aug: {run_forget_aug} | Scheduler: {scheduler_option}")
        
        model_type = kwargs.get('model_type', 'unknown')
        sfx = kwargs.get('suffix', '')
        alpha = kwargs.get('alpha', 0.5)
        T = kwargs.get('temperature', 3.0)
        
        teacher_model.eval()
        model = model.to(device)
        model.train()
        train_data = self.loader_dict['retain']
        num_data = len(train_data)
        
        optimizer = optimizer_picker(optimization, model.parameters(), lr=self.params['lr'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler = scheduler_picker(optimizer, num_data, scheduler_option)
        test_criterion = loss_picker('ce')
        self.logger.info(f"Distillation with alpha: {alpha} | Temperature: {T}")
        criterion = DistillationLoss(alpha=alpha, temperature=T)

        ## start the training process for train data: finetune, last-k retrain
        ## initialize the early_stopping object
        ersp_fn = os.path.join(checkpoint_path, f'{self.__get_name__()}_{protocal}_{self._suffix_}_es_ckpt.pt')
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0, path=ersp_fn)
        
        best_acc = 0.0
        best_model = None
        best_metric, best_iter = 0.0, -1
        is_adv_loss = isinstance(adv_perturb_loss, AdvLoss)
        self.logger.info(f"Start the training process for {self.params['epochs']} epochs")
        cum_time, best_time = 0.0, 0.0        
        for epoch in tqdm(range(self.params['epochs'])):
            start_time = time.time()
            loss_avg = 0.0          ## total loss
            adv_avg = 0.0           ## adversarial loss
            feed_avg = 0.0          ## feedback loss
            clean_loss_avg = 0.0    ## clean loss
            
            t_start = time.time()
            for _, (inputs, labels) in enumerate(train_data):
                optimizer.zero_grad()
                if (run_train_aug != None or len(run_train_aug) > 0) and run_train_aug in ["cutmix", "mixup"]:
                    # inputs, labels = self.online_transform(run_train_aug, inputs, labels, self.num_classes)
                    inputs, targets = self._online_transform_(inputs, targets, run_train_aug)
                # labs_['retain'].extend(labels)  # torch.argmax(labels, dim=1).cpu().numpy()
                inputs, labels = inputs.to(device), labels.to(device)
                
                loss_adv_ = 0.0
                loss_comb = 0.0
                ## perturbation based robust training
                if is_adv_loss and self.params['rob_reg'] > 0.0:
                    # LDS should be calculated before the forward for cross entropy
                    loss_adv_ = adv_perturb_loss(model, inputs, labels)
                    loss_comb += self.params['rob_reg'] * loss_adv_

                student_logits = model(inputs)
                ## normal distillation from the teacher model to the student model
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs.to(device_teacher))

                # embs_['retain'].extend(student_logits)
                teacher_logits = teacher_logits.to(device)
                loss_dist_ = criterion(student_logits, teacher_logits, labels)
                
                loss_comb += loss_dist_
            
                ## model parameters regularization
                current_alpha = 0.0
                if self.params['dynreg']:
                    if epoch < self.params['epochs'] - self.params['no_reg_epochs']:
                        current_alpha = self.params['gamma'] * (1 - (epoch / (self.params['epochs'] - self.params['no_reg_epochs'])))
                    elif epoch == self.params['epochs'] - self.params['no_reg_epochs']:
                        current_alpha = self.params['gamma']
                    else:
                        current_alpha = 0.0
                else:
                    current_alpha = self.params['gamma']
                
                if self.params['regularizer'] != 'none':
                    loss_reg = self.weight_regularization(model, self.params['regularizer'])
                    # print(f"loss (no reg): {loss_comb.item()}, regularization: {loss_reg} \n")
                    loss_comb += current_alpha * loss_reg
                    # print(f"loss (with reg): {loss_comb.item()} \n")
                
                loss_comb.backward(retain_graph=True)
                optimizer.step()

                adv_avg += loss_adv_ * self.params['rob_reg']
                clean_loss_avg += loss_dist_.item()
                loss_avg += loss_comb.item()
            
            ## Applying the feedback from the forget set with triplet loss
            scaler = 1.0
            if self.params['dynamic_weight']:
                scaler = (epoch + 1) * 1.5 / self.params['epochs']
            t_fed_start = time.time()
            if feedback_loss is not None and self.params['strength'] > 0.0:
                self._set_lr_(optimizer, self.params['lr_feedback'])
                for key_ in ['valid', 'forget']:
                    aug_option = run_forget_aug if key_ == 'forget' else "none"
                    for inputs, targets in self.loader_dict[key_]:
                        optimizer.zero_grad()
                        if aug_option != None and aug_option in ["cutmix", "mixup"]:
                            inputs, targets = self.online_transform(aug_option, inputs, targets, self.num_classes)
                        # outputs = model(inputs.to(device))

                # compute the alignment loss as feedback from the forget set for the model
                if isinstance(feedback_loss, TripletLoss):
                    u_fed = 0
                    u_fed = self.params['strength'] * u_fed * scaler
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
            
            _, val_acc_ = test(model, self.loader_dict['valid'], test_criterion, device)
            ## save the best model based on the validation accuracy
            metric = val_acc_
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
            for key_, loader_ in self.loader_dict.items():
                _, acc_ = test(model, loader_, test_criterion, device)
                self.logger.info(f"Testing [ {key_:6} ] :  Acc: {acc_:.4f}")
                # acc_, auc_ = self._model_evaluate_(model, loader_, device)
                # self.logger.info(f"Testing [ {key_:6} ]: Acc: {acc_:.4f} | AUC: {auc_:.4f}")
                # _, acc_, acc_cls_ = test(model, loader_, device, criterion, transforms.ToTensor())
                # self.logger.info(f"Testing [ {key_:6} ]: Acc: {acc_:.4f} | Class Acc: {dict_to_str(acc_cls_)}")
                # if key_ == 'valid':
                #     val_acc_ = acc_
                
            ## save the model checkpoints
            if self.params['save_checkpoints'] and (epoch % 10 == 0) and epoch > 0:
                outfn = f"{model_type}_{protocal}_{sfx}_adv_{adv_perturb_loss}_feedback_{feedback_loss}_{self._suffix_}_ep-{epoch}.pth"
                out_path_ = os.path.join(self.path['model'], self.__get_name__(), outfn)
                save_model(model, out_path_)

            early_stopping(loss_comb + self.params['strength'] * feed_avg * num_data, model)  # err_valid
            if early_stopping.early_stop:
                self.logger.info("+++" * 10 + f"\t Early stopping at epoch: {epoch} \t" + "+++" * 10)
                break
        
        _, test_acc = test(best_model, self.loader_dict['test'], test_criterion, device)
        self.logger.info(f"Best model test accuracy: {test_acc:.4f} @  {best_iter} epoch: {best_time:.3f} sec")
        checkpoint = {
            'epoch': -1,
            'state_dict': best_model.state_dict(),
            'best_acc': test_acc,
            'protocal': protocal,
        }
        return best_model, checkpoint

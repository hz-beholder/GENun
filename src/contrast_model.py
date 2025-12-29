import os
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F
# from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, SubsetRandomSampler

from early_stopping import EarlyStopping
from model_deep import reset_final_layers
from utils import AverageMeter, save_model
from learner import loss_picker, scheduler_picker, optimizer_picker, test

from models import UnlearnBasic



class ContrastModelWrapper(nn.Module):
    def __init__(self, base_model, projector_dim=128, output_dim=64):
        super(ContrastModelWrapper, self).__init__()
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])   ## remove the last layer
        self.fc = base_model.fc
        
        ## Projector head: MLP to project features into contrastive space
        self.projector = nn.Sequential(
            nn.Linear(base_model.fc.in_features, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(inplace=True),   # hidden layer
            nn.Linear(projector_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        
    def parameters(self):
        return list(self.backbone.parameters()) + list(self.fc.parameters()) + list(self.projector.parameters())
    
    def recover(self):
        org_model = nn.Sequential(*list(self.backbone.children())[:-1], self.fc)
        return org_model
    
    def forward(self, x, use_projection=False):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)   # flatten
        logits = self.fc(features)
        
        if use_projection:
            projected_features = self.projector(features)
            projected_features = F.normalize(projected_features, dim=1) 
            return logits, projected_features
        else:
            return logits

class ContrastiveSimSiamModelWrapper(nn.Module):
    def __init__(self, base_model, projector_dim=128, predictor_dim=128):
        super(ContrastiveSimSiamModelWrapper, self).__init__()
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  ## remove the last layer
        self.fc = base_model.fc   
        
        ## Projector head: MLP to project features into contrastive space
        self.projector = nn.Sequential(
            nn.Linear(base_model.fc.in_features, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(inplace=True),   # hidden layer
            nn.Linear(projector_dim, projector_dim),
            # nn.BatchNorm1d(projector_dim)
        )

        ## Predictor head: MLP to predict the features
        self.predictor = nn.Sequential(
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(predictor_dim),
            nn.ReLU(inplace=True),   # hidden layer
            nn.Linear(predictor_dim, predictor_dim),
            # nn.BatchNorm1d(predictor_dim)
        )

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.fc.parameters()) + \
                list(self.projector.parameters()) + list(self.predictor.parameters())
    
    def forward(self, x, use_projection=True):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # flatten
        
        if use_projection:
            projections = self.projector(features)
            normed_projections = F.normalize(projections, dim=1)
            predictions = self.predictor(normed_projections)
            return features, projections, predictions
        else:
            logits = self.fc(features)
            return features, None, logits
    
    

class ContrastRunningUnlearn(UnlearnBasic):
    # Define augmentation pipeline for generating positive samples
    __aug_pip__ = v2.Compose([
        v2.RandomResizedCrop(size=32),  #, scale=(0.8, 1.0)
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        v2.RandomRotation(degrees=15),
        v2.GaussianBlur(kernel_size=(5, 5)),
        # v2.ToTensor(),
    ])
    
    __pos_aug_pip__ = v2.Compose([
        v2.Pad(padding=4, fill=(125, 123, 113)),
        v2.RandomCrop(32, padding=0),
        v2.RandomHorizontalFlip(),
        # v2.ToTensor(),
        v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    
    augment_pipeline = v2.Compose([
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(0.4, 0.4, 0.4, 0.4),
        v2.RandomGrayscale(p=0.2),
        v2.GaussianBlur(kernel_size=3),
        # v2.ToTensor(),
    ])
    
    def __init__(self, logpath: str, logname: str, out_dir: str='results', out_name: str='model'):
        super(ContrastRunningUnlearn, self).__init__(logpath, logname, out_dir, out_name)
    
    def __get_name__(self):
        return 'ContrastRunningUnlearn'

    def contrastive_loss(self, p, z, negatives=None):
        z = z.detach()   # stop_gradient features
        p = F.normalize(p, dim=1)  
        z = F.normalize(z, dim=1)
        pos_loss = - (p * z).sum(dim=-1).mean()
        
        neg_loss = (p * negatives).sum(dim=-1).mean()        
        return pos_loss + neg_loss

    def _generate_augment_samples_(self, batch, augmentation_pipeline=None):
        if augmentation_pipeline is None:
            augmentation_pipeline = self.__aug_pip__
        return torch.stack([augmentation_pipeline(img) for img in batch])
        
    ## using the contrastive way for unlearning
    def unlearn(self, model, protocal="FT", adv_perturb_loss=None, feedback_loss=None, patience=20, loss='ce',
                scheduler_option="", optimization='sgd', run_train_aug=False, run_forget_aug=False,
                device='cpu', checkpoint_path='.', **kwargs):
        self.logger.info(f"Unlearning with Contrastive Augmentation | Optimizer: {optimization} | " 
                            f"| Train Aug: {run_train_aug} | Forget Aug: {run_forget_aug}")
        
        def _generate_negative_samples(retain, device=device):
            batch, _ = next(iter(retain))
            return batch.to(device)

        # if protocal == 'LASTK':
        #     # combine the last-k layer retraining with the fine-tuning as linear probing
        #     ## freeze the previous layers and retrain the last k layers, like linear probing
        #     last_k = kwargs.get('last_k', 1)
        #     re_init = kwargs.get('re_init', True)
        #     model = reset_final_layers(model, last_k, self.logger, re_init)
        # else:
        #     model.train()
        
        model_type = kwargs.get('model_type', 'unknown')
        init_wt_f = kwargs.get('weight_forget', 10.0)
        init_wt_r = kwargs.get('weight_retain', 1.0)
        # cls_wt = kwargs.get('classify_weight', 1.0)
        # cont_wt = kwargs.get('contrast_weight', 1.0)
        aug_retain = kwargs.get('augument_retain', False)
        epochs = self.params['epochs']
        
        retain_data = self.dataloaders['retain']
        forget_data = self.dataloaders['forget']
        
        model = model.to(device)
        model.train()
        optimizer = optimizer_picker(optimization, model.parameters(), lr=self.params['lr'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler = None
        if scheduler_option != "":
            scheduler = scheduler_picker(optimizer, scheduler_option, lr=self.params['lr'],
                                     batch_size=self.params['batch_size'], num_epochs=epochs)
        criterion = loss_picker(loss)

        ersp_fn = os.path.join(checkpoint_path, f'{self.__get_name__()}_{protocal}_{self._suffix_}_es_ckpt.pt')
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0, path=ersp_fn)
    
        best_acc, best_metric = 0.0, 0.0
        self.logger.info(f"Start the training process for {epochs} epochs")
        time_s = time.time()
        cum_time, best_time = 0.0, 0.0
        for epoch in tqdm(range(epochs)):
            if self.params['dynamic_weight']:  ## curriculum learning with dynamic weight
                # Gradually shift the focus from retain set R to forget set F:  curriculum learning
                wt_r = max(init_wt_r * (1 - epoch / epochs), 1.0)
                wt_f = min(init_wt_f * (epoch / epochs), init_wt_r)
            else:
                wt_r = init_wt_r
                wt_f = init_wt_f
            running_loss = 0.0
            
            #TODO: the zip is problem: retain data is larger than forget data
            for (batch_r, labs_r), (batch_f, labs_f) in zip(retain_data, forget_data):
                batch_r, labs_r = batch_r.to(device), labs_r.to(device)
                batch_f, labs_f = batch_f.to(device), labs_f.to(device)
                
                if aug_retain:
                    batch_r = self._generate_augment_samples_(batch_r, self.__pos_aug_pip__).to(device)
                
                logits_r = model(batch_r)
                loss_cls = criterion(logits_r, labs_r)
                
                _, proj_f = model(batch_f, use_projection=True)
                pos_f = self._generate_augment_samples_(batch_f).to(device)
                _, proj_f_aug_pos = model(pos_f, use_projection=True)
                
                size_ = proj_f.size(0)
                neg_f = _generate_negative_samples(retain_data).to(device)
                _, proj_f_aug_neg = model(neg_f[:size_], use_projection=True)
                
                loss_cont = self.contrastive_loss(proj_f, proj_f_aug_pos, proj_f_aug_neg)
                
                # total_loss = loss_cls * cls_wt + loss_cont * cont_wt
                total_loss = loss_cls * wt_r + loss_cont * wt_f
                running_loss += total_loss.item()
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()    
            
            if scheduler is not None:
                scheduler.step()
            
            running_loss = running_loss / len(retain_data)
            t_all = time.time() - time_s
            self.logger.info(f"Epoch: {epoch+1}/{self.params['epochs']} |  Time: {t_all:.3f} sec: Loss: {running_loss}")
            
            _, val_acc = test(model, self.dataloaders['valid'], criterion, device)
            _, for_acc = test(model, self.dataloaders['forget'], criterion, device)

            metric = val_acc - np.abs(for_acc - val_acc)
            
            if metric > best_metric:
                best_time = cum_time + t_all
                self.logger.info('==> Best Validation [previous => current]:' + 
                            f' acc-[{best_acc :.4f} => {val_acc :.4f}], metric-[{best_metric :.4f} => {metric :.4f}] | Epoch: {epoch+1} @ {best_time:.3f} sec')
                best_model = deepcopy(model)
                best_iter = epoch
                best_acc = val_acc
                best_metric = metric
            cum_time += time.time() - time_s
            
            for key_ in ['train', 'retain', 'valid', 'test', 'forget']:
                loader_ = self.dataloaders[key_]
                _, acc_ = test(model, loader_, criterion, device)
                self.logger.info(f"Testing [ {key_:6} ] :  Acc: {acc_:.4f}")
            
            if self.params['save_checkpoints'] and (epoch % 10 == 0) and epoch > 0:
                outfn = f"{model_type}_{protocal}_adv_{adv_perturb_loss}_feedback_{feedback_loss}_{self._suffix_}_ep-{epoch}.pt"
                out_path_ = os.path.join(self.path['model'], self.__get_name__(), outfn)
                save_model(model, out_path_)

            early_stopping(running_loss, model)  # err_valid
            if early_stopping.early_stop:
                self.logger.info("+++" * 10 + f"\t Early stopping at epoch: {epoch} \t" + "+++" * 10)
                break
        
        _, test_acc = test(best_model, self.dataloaders['test'], criterion, device)
        self.logger.info(f"Best model test accuracy: {test_acc:.4f} @  {best_iter} epoch: {best_time:.3f} sec")
        
        checkpoint = {
            'epoch': -1,
            'state_dict': best_model.state_dict(),
            'best_acc': test_acc,
            'protocal': protocal,
        }
        return best_model, checkpoint
    
    

class MoCoVar(nn.Module):
    def __init__(self, base_model, output_dim=64, queue_size=4096, momentum=0.999, temperature=0.07):
        super(MoCoVar, self).__init__()
        
        ## initialized the encoder and momentum encoder
        self.encoder_q = base_model
        self.encoder_k = deepcopy(self.encoder_q)
        
        ## initialize the queue
        self.queue_size = queue_size
        self.register_buffer('queue', torch.randn(output_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.momentum = momentum
        self.temperature = temperature
        
        ## ensure the momentum encoder starts with the same weights as the encoder    
        self._initilize_momentum_encoder_()
    
    def recover(self):
        return self.encoder_q.recover()
    
    def _initilize_momentum_encoder_(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  ## intialize
            param_k.requires_grad = False   ## not update by gradient
    
    @torch.no_grad()
    def _momentum_update_by_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue_(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        ## replace the keys at ptr with the new keys
        # self.queue[:, ptr:ptr+batch_size] = keys.T
        
        # If the batch is larger than the remaining space in the queue
        if ptr + batch_size > self.queue_size:
            # Part 1: Fill up the remaining space in the queue (from ptr to the end)
            remaining_space = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining_space].T
            # Part 2: Wrap around to the start of the queue
            self.queue[:, :batch_size - remaining_space] = keys[remaining_space:].T
        else:
            # Normal enqueue if the batch fits into the current queue window
            self.queue[:, ptr:ptr + batch_size] = keys.T
            
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
        
    def forward(self, im_q, im_k):
        _, q = self.encoder_q(im_q, use_projection=True)  # queries: N x C
        q = F.normalize(q, dim=1)
        
        ## compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_by_encoder()   # update the momentum encoder
            _, k = self.encoder_k(im_k, use_projection=True)
            k = F.normalize(k, dim=1)
        
        # compute logits
        # Einstein sum is more intuitive
        ## positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        ## retrival negative samples from the queue
        negatives = self.queue.clone().detach()   # negatives: D x C
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, negatives])  # N x K
        
        ## logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.temperature
     
        return logits, q, k
    

class ContrastMomentUnlearn(UnlearnBasic):
    best__aug_pip__ = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ])
    
    __aug_pip__ = v2.Compose([
        # v2.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # v2.RandomRotation(degrees=15),
        # v2.GaussianBlur(kernel_size=(5, 5)),
        # nn.Identity(),
    ])
    
    __pos_aug_pip__ = v2.Compose([
        # v2.Pad(padding=4, fill=(125, 123, 113)),
        # v2.RandomCrop(32, padding=0),
        v2.RandomHorizontalFlip(),
        v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    
    augment_pipeline = v2.Compose([
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(0.4, 0.4, 0.4, 0.4),
        v2.RandomGrayscale(p=0.2),
        v2.GaussianBlur(kernel_size=3),
    ])
    
    
    def __init__(self, logpath: str, logname: str, out_dir: str='results', out_name: str='model'):
        super(ContrastMomentUnlearn, self).__init__(logpath, logname, out_dir, out_name)
        
    def __get_name__(self):
        return 'ContrastMomentUnlearn'

    def _generate_augment_samples_(self, batch, augmentation_pipeline=None):
        if augmentation_pipeline is None:
            augmentation_pipeline = self.__aug_pip__
        return torch.stack([augmentation_pipeline(img) for img in batch])


    def contrastive_loss(self, query, key, queue, temperature=0.07):
        pos_logits = torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)     ## postive similarity between query and key
        neg_logits = torch.einsum('nc,kc->nk', [query, queue.clone().detach()])   ## negative similarity between query and queue
        
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        logits = logits / temperature
        
        # Labels: the first one is always the positive sample
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(query.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

    ## version 0
    def unlearn_v0(self, model, protocal="FT", adv_perturb_loss=None, feedback_loss=None, patience=20, loss='ce',
                scheduler_option="", optimization='sgd', run_train_aug=False, run_forget_aug=False,
                device='cpu', checkpoint_path='.', **kwargs):
        self.logger.info(f"Unlearning with MoCo Augmentation | Optimizer: {optimization} | " 
                            f"| Train Aug: {run_train_aug} | Forget Aug: {run_forget_aug}")
        
        # if protocal == 'LASTK':
        #     # combine the last-k layer retraining with the fine-tuning as linear probing
        #     ## freeze the previous layers and retrain the last k layers, like linear probing
        #     last_k = kwargs.get('last_k', 1)
        #     re_init = kwargs.get('re_init', True)
        #     model = reset_final_layers(model, last_k, self.logger, re_init)
        # else:
        #     model.train()
        
        model_type = kwargs.get('model_type', 'unknown')
        outs_dim = kwargs.get('output_dim', 64)
        init_wt_f = kwargs.get('weight_forget', 10.0)
        init_wt_r = kwargs.get('weight_retain', 1.0)
        # cls_wt = kwargs.get('classify_weight', 1.0)
        # cont_wt = kwargs.get('contrast_weight', 1.0)
        epochs = self.params['epochs']
        aug_retain = kwargs.get('augument_retain', False)
        queue_size = kwargs.get('queue_size', 4096)
        
        retain_data = self.dataloaders['retain']
        forget_data = self.dataloaders['forget']
        
        ## initialize the MoCo Queue
        moco_model = MoCoVar(model, outs_dim, queue_size=queue_size, momentum=0.8, temperature=0.1)
        moco_model.to(device)
        moco_model.train()
        optimizer = optimizer_picker(optimization, moco_model.parameters(), lr=self.params['lr'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler = None
        if scheduler_option != "":
            scheduler = scheduler_picker(optimizer, scheduler_option, lr=self.params['lr'],
                                     batch_size=self.params['batch_size'], num_epochs=epochs)
        criterion = loss_picker(loss)

        ersp_fn = os.path.join(checkpoint_path, f'{self.__get_name__()}_{protocal}_{self._suffix_}_es_ckpt.pt')
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0, path=ersp_fn)
        
        best_epoch, best_model = -1, None
        best_acc, best_metric = 0.0, 0.0
        cum_time, best_time = 0.0, 0.0
        
        time_s = time.time()
        self.logger.info(f"Start the training process for {epochs} epochs")
        for epoch in tqdm(range(epochs)):
            l_cls, l_con = AverageMeter("loss_cls"), AverageMeter("loss_cont")
            
            if self.params['dynamic_weight'] == 1:  ## curriculum learning with dynamic weight
                # Gradually shift the focus from retain set R to forget set F:  curriculum learning
                wt_r = max(init_wt_r * (1 - epoch / epochs), 1.0)
                wt_f = min(init_wt_f * (epoch / epochs), init_wt_r)
            elif self.params['dynamic_weight'] == -1:
                ### dynamic weight for curriculum learning: easier to harder
                wt_r = min(init_wt_r * (epoch / epochs), init_wt_r)
                wt_f = max(init_wt_f * (1 - epoch / epochs), 1.0)
            else:
                wt_r = init_wt_r
                wt_f = init_wt_f
            loss_run = 0.0
            
            for (batch_f, labs_f), (batch_r, labs_r) in zip(forget_data, retain_data):
                batch_r, labs_r = batch_r.to(device), labs_r.to(device)
                batch_f = batch_f.to(device)   #, labs_f, labs_f.to(device)
                
                ## forward pass with moco for forget set
                aug_f = self._generate_augment_samples_(batch_f).to(device)
                logits, queries, keys = moco_model(batch_f, aug_f)
                loss_cont = F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long).to(device))
                l_con.update(loss_cont.item(), batch_f.size(0))
                    
                ## forward pass with moco for retain set
                if aug_retain:
                    batch_r = self._generate_augment_samples_(batch_r, self.__pos_aug_pip__).to(device)
                
                logits_r, proj_feat_r = moco_model.encoder_q(batch_r, use_projection=True)
                loss_cls = criterion(logits_r, labs_r)
                l_cls.update(loss_cls.item(), batch_f.size(0))

                # moco_model._dequeue_and_enqueue_(proj_feat_r)
                moco_model._dequeue_and_enqueue_(keys)
                
                total_loss = loss_cls * wt_r + loss_cont * wt_f
                loss_run += total_loss.item()
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()    
            
            if scheduler is not None:
               scheduler.step()
            
            model_cur = moco_model.encoder_q
            loss_run = loss_run / len(retain_data)
            t_all = time.time() - time_s
            self.logger.info(f"Epoch: {epoch+1}/{self.params['epochs']} |  Time: {t_all:.3f} sec: Loss: {loss_run}")
            self.logger.info(f"Loss: [ Cls: {l_cls.avg:.4f} | Cont: {l_con.avg:.4f} ]")
            
            _, val_acc = test(model_cur, self.dataloaders['valid'], criterion, device)
            _, for_acc = test(model_cur, self.dataloaders['forget'], criterion, device)
            metric = val_acc - np.abs(for_acc - val_acc)
            
            if metric > best_metric:
                best_time = cum_time + t_all
                _, acc_test = test(model, self.dataloaders['test'], criterion, device)
                true_met = acc_test - np.abs(for_acc - acc_test)
                self.logger.info('==> Best Validation [previous => current]:' + 
                            f' acc-[{best_acc :.4f} => {val_acc :.4f}], metric-[{best_metric :.4f} => {metric :.4f}] TRUE-metric: {true_met:.4f} | ' +
                            f' Epoch: {epoch+1} @ {best_time:.3f} sec')
                best_model = deepcopy(model_cur)
                best_epoch = epoch
                best_acc = val_acc
                best_metric = metric
            cum_time += time.time() - time_s
            
            for key_ in ['train', 'retain', 'valid', 'test', 'forget']:
                loader_ = self.dataloaders[key_]
                _, acc_ = test(model_cur, loader_, criterion, device)
                self.logger.info(f"Testing [ {key_:6} ] :  Acc: {acc_:.4f}")
            
            if self.params['save_checkpoints'] and (epoch % 10 == 0) and epoch > 0:
                outfn = f"{model_type}_{protocal}_adv_{adv_perturb_loss}_feedback_{feedback_loss}_{self._suffix_}_ep-{epoch}.pt"
                out_path_ = os.path.join(self.path['model'], self.__get_name__(), outfn)
                save_model(model_cur, out_path_)
            
            early_stopping(loss_run, model_cur)
            if early_stopping.early_stop:
                self.logger.info("+++" * 10 + f"\t Early stopping at epoch: {epoch} \t" + "+++" * 10)
                break
            
        _, acc_test = test(best_model, self.dataloaders['test'], criterion, device)
        self.logger.info(f"Best model test accuracy: {acc_test:.4f} @  {best_epoch} epoch: {best_time:.3f} sec")
        
        checkpoint = {
            'epoch': -1,
            'state_dict': best_model.state_dict(),
            'best_acc': acc_test,
            'protocal': protocal,
        }
        return best_model, checkpoint
    
    ## version 1
    def unlearn(self, model, protocal="FT", adv_perturb_loss=None, feedback_loss=None, patience=20, loss='ce',
                scheduler_option="", optimization='sgd', run_train_aug=False, run_forget_aug=False,
                device='cpu', checkpoint_path='.', **kwargs):
        self.logger.info(f"Unlearning with MoCo Augmentation | Optimizer: {optimization} | " 
                            f"| Train Aug: {run_train_aug} | Forget Aug: {run_forget_aug}")
        
        if protocal == 'LASTK':
            # combine the last-k layer retraining with the fine-tuning as linear probing
            ## freeze the previous layers and retrain the last k layers, like linear probing
            last_k = kwargs.get('last_k', 1)
            re_init = kwargs.get('re_init', True)
            model = reset_final_layers(model, last_k, self.logger, re_init)
        else:
            model.train()
        
        model_type = kwargs.get('model_type', 'unknown')
        outs_dim = kwargs.get('output_dim', 64)
        init_wt_f = kwargs.get('weight_forget', 10.0)
        init_wt_r = kwargs.get('weight_retain', 1.0)
        epochs = self.params['epochs']
        # aug_retain = kwargs.get('augument_retain', False)
        queue_size = kwargs.get('queue_size', 4096)
        
        retain_data = self.dataloaders['retain']
        forget_data = self.dataloaders['forget']
        
        ## initialize the MoCo Queue
        moco_model = MoCoVar(model, outs_dim, queue_size=queue_size, momentum=0.8, temperature=0.1)
        moco_model.to(device)
        # moco_model.train()
        optimizer = optimizer_picker(optimization, moco_model.parameters(), lr=self.params['lr'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler = None
        if scheduler_option != "":
            scheduler = scheduler_picker(optimizer, scheduler_option, lr=self.params['lr'],
                                     batch_size=self.params['batch_size'], num_epochs=epochs)
        criterion = loss_picker(loss)

        ersp_fn = os.path.join(checkpoint_path, f'{self.__get_name__()}_{protocal}_{self._suffix_}_es_ckpt.pt')
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0, path=ersp_fn)
        
        best_epoch, best_model = -1, None
        best_acc, best_metric = 0.0, 0.0
        cum_time, best_time = 0.0, 0.0
        
        time_s = time.time()
        self.logger.info(f"Start the training process for {epochs} epochs")
        for epoch in tqdm(range(epochs)):
            l_cls, l_con = AverageMeter("loss_cls"), AverageMeter("loss_cont")
            
            if self.params['dynamic_weight'] == 1:  ## curriculum learning with dynamic weight
                # Gradually shift the focus from retain set R to forget set F:  curriculum learning
                wt_r = max(init_wt_r * (1 - epoch / epochs), 1.0)
                wt_f = min(init_wt_f * (epoch / epochs), init_wt_r)
            elif self.params['dynamic_weight'] == -1:
                ### dynamic weight for curriculum learning: easier to harder
                wt_r = min(init_wt_r * (epoch / epochs), init_wt_r)
                wt_f = max(init_wt_f * (1 - epoch / epochs), 1.0)
            else:
                wt_r = init_wt_r
                wt_f = init_wt_f
            loss_run = 0.0
            
            for (batch_f, labs_f) in forget_data:
                r_idx = np.random.choice(len(retain_data.dataset), size=batch_f.shape[0], replace=False)
                sampler = SubsetRandomSampler(r_idx)
                batch_r, labs_r = next(iter(DataLoader(retain_data.dataset, batch_size=batch_f.shape[0], sampler=sampler)))
            
                batch_r, labs_r = batch_r.to(device), labs_r.to(device)
                batch_f = batch_f.to(device)   #, labs_f, labs_f.to(device)
                
                ## forward pass with moco for forget set
                aug_f = self._generate_augment_samples_(batch_f).to(device)
                logits, queries, keys = moco_model(batch_f, aug_f)
                loss_cont = F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long).to(device))
                l_con.update(loss_cont.item(), batch_f.size(0))
                
                ## forward pass with moco for retain set
                # if aug_retain:
                #     batch_r = self._generate_augment_samples_(batch_r, self.__pos_aug_pip__).to(device)
                
                logits_r, proj_feat_r = moco_model.encoder_q(batch_r, use_projection=True)
                loss_cls = criterion(logits_r, labs_r)
                l_cls.update(loss_cls.item(), batch_r.size(0))

                # moco_model._dequeue_and_enqueue_(proj_feat_r)
                moco_model._dequeue_and_enqueue_(keys)
                
                total_loss = loss_cls * wt_r + loss_cont * wt_f
                loss_run += total_loss.item()
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()    
            
            if scheduler is not None:
               scheduler.step()
            
            model_cur = moco_model.encoder_q
            loss_run = loss_run / len(retain_data)
            t_all = time.time() - time_s
            self.logger.info(f"Epoch: {epoch+1}/{self.params['epochs']} |  Time: {t_all:.3f} sec: Loss: {loss_run}")
            self.logger.info(f"Loss: [ Cls: {l_cls.avg:.4f} | Cont: {l_con.avg:.4f} ]")
            
            _, val_acc = test(model_cur, self.dataloaders['valid'], criterion, device)
            _, for_acc = test(model_cur, self.dataloaders['forget'], criterion, device)
            metric = val_acc - np.abs(for_acc - val_acc)
            
            if metric > best_metric:
                best_time = cum_time + t_all
                _, acc_test = test(model, self.dataloaders['test'], criterion, device)
                true_met = acc_test - np.abs(for_acc - acc_test)
                self.logger.info('==> Best Validation [previous => current]:' + 
                            f' acc-[{best_acc :.4f} => {val_acc :.4f}], metric-[{best_metric :.4f} => {metric :.4f}] TRUE-metric: {true_met:.4f} | ' +
                            f' Epoch: {epoch+1} @ {best_time:.3f} sec')
                best_model = deepcopy(model_cur)
                best_epoch = epoch
                best_acc = val_acc
                best_metric = metric
            cum_time += time.time() - time_s
            
            for key_ in ['train', 'retain', 'valid', 'test', 'forget']:
                loader_ = self.dataloaders[key_]
                _, acc_ = test(model_cur, loader_, criterion, device)
                self.logger.info(f"Testing [ {key_:6} ] :  Acc: {acc_:.4f}")
            
            if self.params['save_checkpoints'] and (epoch % 10 == 0) and epoch > 0:
                outfn = f"{model_type}_{protocal}_adv_{adv_perturb_loss}_feedback_{feedback_loss}_{self._suffix_}_ep-{epoch}.pt"
                out_path_ = os.path.join(self.path['model'], self.__get_name__(), outfn)
                save_model(model_cur, out_path_)
            
            early_stopping(loss_run, model_cur)
            if early_stopping.early_stop:
                self.logger.info("+++" * 10 + f"\t Early stopping at epoch: {epoch} \t" + "+++" * 10)
                break
            
        _, acc_test = test(best_model, self.dataloaders['test'], criterion, device)
        self.logger.info(f"Best model test accuracy: {acc_test:.4f} @  {best_epoch} epoch: {best_time:.3f} sec")
        
        checkpoint = {
            'epoch': -1,
            'state_dict': best_model.state_dict(),
            'best_acc': acc_test,
            'protocal': protocal,
        }
        return best_model, checkpoint
    





class MoCoQueue(nn.Module):
    def __init__(self, queue_size, feature_dim, device):
        super(MoCoQueue, self).__init__()
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.queue = torch.randn(self.feature_dim, self.queue_size)
        self.queue = nn.functional.normalize(self.queue, dim=0).to(device)
        self.ptr = 0

    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.ptr)
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Part 1: Fill up the remaining space in the queue (from ptr to the end)
            remaining_space = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining_space].T
            # Part 2: Wrap around to the start of the queue
            self.queue[:, :batch_size - remaining_space] = keys[remaining_space:].T
            # self.queue[:, ptr:] = keys[:self.queue_size - ptr].T
            # self.queue[:, :ptr + batch_size - self.queue_size] = keys[self.queue_size - ptr:].T
        
        self.ptr = (ptr + batch_size) % self.queue_size

    def forward(self, query, key):
        # Normalize features
        query = nn.functional.normalize(query, dim=1)
        key = nn.functional.normalize(key, dim=1)

        # Positive logits (contrast query with key)
        positive_logits = torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)

        # Negative logits (contrast query with all entries in the queue)
        negs = self.queue.clone().detach()
        negative_logits = torch.einsum('nc,ck->nk', [query, negs])

        # Concatenate positive and negative logits
        logits = torch.cat([positive_logits, negative_logits], dim=1)

        # Dequeue and enqueue keys (update the queue)
        self._dequeue_and_enqueue(key)

        return logits


class ModelWrapper(nn.Module):
    def __init__(self, base_model, projector_dim=128, output_dim=64, stop_at_layer='fc'):
        super(ModelWrapper, self).__init__()
        self.fc = base_model.fc
        self.backbone = nn.ModuleDict()
        
        for name, module in base_model.named_children():
            if name == stop_at_layer:
                break  # Stop if the layer matches stop_at_layer
            else:
                self.backbone[name] = module  # Keep (name, layer) pairs intact
        
        # Projection head (for contrastive learning)
        self.projector = nn.Sequential(
            nn.Linear(base_model.fc.in_features, projector_dim),
            nn.ReLU(inplace=False),
            nn.Linear(projector_dim, output_dim)
        )
    
    def parameters(self):
        return list(self.backbone.parameters()) + list(self.fc.parameters()) + list(self.projector.parameters())
    
    def recover(self):
        org_model = nn.Sequential(*list(self.backbone.children())[:-1], nn.Flatten(), self.fc)
        return org_model
    
    def forward(self, x, use_projection=False):
        inputs = x.clone()
        for _, layer in self.backbone.items():
            inputs = layer(inputs)

        inputs = inputs.view(inputs.size(0), -1)
        logits = self.fc(inputs)
        
        if use_projection:
            projected_features = F.normalize(self.projector(inputs), dim=1) 
            return logits, projected_features
        else:
            return logits
        
class RobustContrastMomentUnlearn(UnlearnBasic):
    __aug_pip__ = v2.Compose([
        # v2.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # v2.RandomRotation(degrees=15),
        # v2.GaussianBlur(kernel_size=(5, 5)),
        # nn.Identity(),
    ])
    
    __pos_aug_pip__ = v2.Compose([
        # v2.Pad(padding=4, fill=(125, 123, 113)),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(32, padding=4),
        v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
      
    augment_pipeline = v2.Compose([
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(0.4, 0.4, 0.4, 0.4),
        v2.RandomGrayscale(p=0.2),
        v2.GaussianBlur(kernel_size=3),
    ])
    
        
    def __init__(self, logpath: str, logname: str, out_dir: str='results', out_name: str='model'):
        super(RobustContrastMomentUnlearn, self).__init__(logpath, logname, out_dir, out_name)
        
    def __get_name__(self):
        return 'RoConMomentUnlearn'

    def _generate_augment_samples_(self, batch, augmentation_pipeline=None):
        if augmentation_pipeline is None:
            augmentation_pipeline = self.__aug_pip__
        return torch.stack([augmentation_pipeline(img) for img in batch])

    ## for the ResNet model unfreezing
    def unfreeze_layers(model, layer_group_to_unfreeze):
        """
        Unfreezes layers of the model based on the group number.

        Args:
            model: Model whose layers will be unfrozen.
            layer_group_to_unfreeze: Index of the layer group to unfreeze.
        """
        layers_to_unfreeze = [
            model.fc,      # Fully connected layer
            model.layer4,  # Block 4
            model.layer3,  # Block 3
            model.layer2,  # Block 2
            model.layer1,  # Block 1
            model.bn1,     # Batch normalization
            model.conv1    # Initial convolution layer
        ]

        if layer_group_to_unfreeze < len(layers_to_unfreeze):
            for param in layers_to_unfreeze[layer_group_to_unfreeze].parameters():
                param.requires_grad = True
    
    def update_momentum_encoder(self, momentum_encoder, model, momentum=0.999):
        """Update the momentum encoder by a weighted average of its own weights and the model's weights."""
        for param_m, param in zip(momentum_encoder.parameters(), model.parameters()):
            param_m.data = param_m.data * momentum + param.data * (1. - momentum)

    def progressive_unfreeze(self, model_layers_to_unfreeze, epoch, unfreeze_schedule):
        """
        Unfreeze layers progressively based on the current epoch and unfreeze schedule.
        
        Args:
            model: The model whose layers need to be unfrozen.
            layers_to_unfreeze: List of layers in the model (in the order of unfreezing).
            epoch: Current epoch number.
            unfreeze_schedule: List of epochs at which layers should be unfrozen.
        """
        # Check if it's time to unfreeze more layers
        if epoch in unfreeze_schedule:
            # Unfreeze the corresponding layer for the current epoch
            layer_idx = unfreeze_schedule.index(epoch)
            if layer_idx < len(model_layers_to_unfreeze):
                for param in model_layers_to_unfreeze[layer_idx].parameters():
                    param.requires_grad = True
                self.logger.info(f"Unfroze layer {layer_idx + 1}/{len(model_layers_to_unfreeze)}") # : {model_layers_to_unfreeze[layer_idx]}
            else:
                self.logger.info(f"All layers unfrozen by epoch {epoch}")

    def progressive_unfreeze_ratio(self, model, layers_to_unfreeze, current_epoch, total_epochs):
        """
        Unfreezes the model layers progressively, starting from the later layers.

        Parameters:
        - model: The model being fine-tuned.
        - layers_to_unfreeze: A list of layer names or modules to unfreeze progressively.
        - current_epoch: The current epoch number (0-indexed).
        - total_epochs: The total number of epochs for training.
        """
        # Calculate the unfreezing threshold based on the training progress
        unfreeze_threshold = len(layers_to_unfreeze) * (current_epoch / total_epochs)
        
        # Freeze/Unfreeze layers based on the threshold
        for i, layer_name in enumerate(layers_to_unfreeze):
            layer = getattr(model, layer_name, None)  # Get the layer/module by its name
            if layer is not None:
                if i <= unfreeze_threshold:
                    # Unfreeze this layer
                    for param in layer.parameters():
                        param.requires_grad = True
                else:
                    # Freeze this layer
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                self.logger.warning(f"Layer {layer_name} not found in the model.")

        # Log which layers are frozen/unfrozen for debugging
        for layer_name in layers_to_unfreeze:
            layer = getattr(model, layer_name, None)
            if layer:
                grad_status = "Unfrozen" if any(p.requires_grad for p in layer.parameters()) else "Frozen"
                self.logger.info(f"Layer {layer_name}: {grad_status}")

    ## unlearning integrated with progressive unfreezing, curriculum learning, and independent optimizing (unlearing and robust finetuning)
    def unlearn(self, model, momentum_encoder, moco_queue, layers_to_unfreeze, unfreeze_schedule,
                protocal="FT", patience=20, loss='ce', scheduler_option="", optimization='sgd', 
                device='cpu', checkpoint_path='.', **kwargs):
        self.logger.info(f"Unlearning with MoCo Augmentation | Optimizer: {optimization} |")

        model_type = kwargs.get('model_type', 'unknown')
        epochs = self.params['epochs']
        wt_f = kwargs.get('weight_forget', 5.0)
        wt_r = kwargs.get('weight_retain', 1.0)
        retain_augment = kwargs.get('augument_retain', False)
        tempreture = kwargs.get('temperature', 0.07)
        momentum = kwargs.get('moco_momentum', 0.8)
        
        ## data for model unlearning
        retain_data = self.dataloaders['retain']
        forget_data = self.dataloaders['forget']
        
        criterion = loss_picker(loss)
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optimizer_picker(optimization, params, lr=self.params['lr'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
        scheduler = scheduler_picker(optimizer, scheduler_option, lr=self.params['lr'],
                                     batch_size=self.params['batch_size'], num_epochs=epochs)

        ersp_fn = os.path.join(checkpoint_path, f'{self.__get_name__()}_{protocal}_{self._suffix_}_es_ckpt.pt')
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0, path=ersp_fn)
        
        best_epoch, best_model = -1, None
        best_acc, best_metric = 0.0, 0.0
        cum_time, best_time = 0.0, 0.0
        
        model = model.to(device)
        momentum_encoder = momentum_encoder.to(device)
        
        time_s = time.time()
        self.logger.info(f"Start the training process for {epochs} epochs")
        for epoch in tqdm(range(epochs)):
            l_cls, l_con = AverageMeter("loss_cls"), AverageMeter("loss_cont")
            self.logger.info(f"Epoch: {epoch+1}/{self.params['epochs']}")
            model.train()
            momentum_encoder.train()
            
            if layers_to_unfreeze and unfreeze_schedule: 
                self.progressive_unfreeze(layers_to_unfreeze, epoch, unfreeze_schedule)
                params = filter(lambda p: p.requires_grad, model.parameters())
                optimizer = optimizer_picker(optimization, params, lr=self.params['lr'], 
                            momentum=self.params['momentum'], weight_decay=self.params['weight_decay'])
            # if epoch in unfreeze_schedule:
                # self.unfreeze_layers(model, unfreeze_schedule[epoch])
                # self.logger.info(f"Unfreezing the layer group: {unfreeze_schedule[epoch]}")
            
            loss_run = 0.0
            for (batch_f, labs_f) in forget_data:
                batch_f = batch_f.to(device)
                
                ## construct negative samples (same size) from the retain dataset loader
                r_idx = np.random.choice(len(retain_data.dataset), size=batch_f.shape[0], replace=False)
                sampler = SubsetRandomSampler(r_idx)
                batch_r, labs_r = next(iter(DataLoader(retain_data.dataset, batch_size=batch_f.shape[0], sampler=sampler)))
                ### augmentation transform for the retain samples for the robust finetuning
                #if retain_augment:
                if np.random.random() > 0.8:
                    batch_r = self._generate_augment_samples_(batch_r, self.__pos_aug_pip__)
                batch_r, labs_r = batch_r.to(device), labs_r.to(device)

                ## forward pass with moco for forget set
                _, query = model(batch_f, use_projection=True)
                # key = model(batch_r)
                with torch.no_grad():
                    _, key = momentum_encoder(batch_r, use_projection=True)
                
                logits = moco_queue(query, key)
                labels = torch.zeros(logits.size(0), dtype=torch.long).to(device)
                loss_cont = criterion(logits / tempreture, labels)
                l_con.update(loss_cont.item(), batch_f.size(0))

                ## forward pass with moco for retain set
                output_r = model(batch_r, use_projection=False)
                loss_cls = criterion(output_r, labs_r)
                l_cls.update(loss_cls.item(), batch_r.size(0))

                total_loss = loss_cls * wt_r + loss_cont * wt_f
                loss_run += total_loss.item()
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()    
            
            self.update_momentum_encoder(momentum_encoder, model, momentum)
            
            if scheduler is not None:
               scheduler.step()
            
            loss_run = loss_run / len(forget_data)
            t_all = time.time() - time_s
            self.logger.info(f"Epoch: {epoch+1}/{self.params['epochs']} |  Time: {t_all:.3f} sec: Loss: {loss_run}")
            self.logger.info(f"Loss: [ Cls: {l_cls.avg:.4f} | Cont: {l_con.avg:.4f} ]")
            
            model_cur = model.recover()
            _, val_acc = test(model_cur, self.dataloaders['valid'], criterion, device)
            _, for_acc = test(model_cur, self.dataloaders['forget'], criterion, device)
            metric = val_acc - np.abs(for_acc - val_acc)
            
            if metric > best_metric:
                best_time = cum_time + t_all
                _, acc_test = test(model_cur, self.dataloaders['test'], criterion, device)
                true_met = acc_test - np.abs(for_acc - acc_test)
                self.logger.info('==> Best Validation [previous => current]:' + 
                            f' acc-[{best_acc :.4f} => {val_acc :.4f}], metric-[{best_metric :.4f} => {metric :.4f}] TRUE-metric: {true_met:.4f} | ' +
                            f' Epoch: {epoch+1} @ {best_time:.3f} sec')
                best_model = deepcopy(model_cur)
                best_epoch = epoch
                best_acc = val_acc
                best_metric = metric
            cum_time += time.time() - time_s
            
            for key_ in ['train', 'retain', 'valid', 'test', 'forget']:
                loader_ = self.dataloaders[key_]
                _, acc_ = test(model_cur, loader_, criterion, device)
                self.logger.info(f"Testing [ {key_:6} ] :  Acc: {acc_:.4f}")
            
            if self.params['save_checkpoints'] and (epoch % 10 == 0) and epoch > 0:
                outfn = f"{model_type}_{protocal}_augr-{retain_augment}_{self._suffix_}_ep-{epoch}.pt"
                out_path_ = os.path.join(self.path['model'], self.__get_name__(), outfn)
                save_model(model, out_path_)
            
            early_stopping(loss_run, model)
            if early_stopping.early_stop:
                self.logger.info("+++" * 10 + f"\t Early stopping at epoch: {epoch} \t" + "+++" * 10)
                break
            
        _, acc_test = test(best_model, self.dataloaders['test'], criterion, device)
        self.logger.info(f"Best model test accuracy: {acc_test:.4f} @  {best_epoch} epoch: {best_time:.3f} sec")
        
        checkpoint = {
            'epoch': -1,
            'state_dict': best_model.state_dict(),
            'best_acc': acc_test,
            'protocal': protocal,
        }
        return best_model, checkpoint
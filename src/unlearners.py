from os import path
import time
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import check_sparsity, kaiming_weights_init
# from logger import create_logger
from model_bases import DeepModels
from schema import BasicUnlearnSchema
from model_deep import reset_final_layers
import influence_functions as ifs 
from certified_removal import CRModel, CertifiedRemoval
from learner import fisher_hessian, generate_weight_mask, get_mean_var, loss_picker, optimizer_picker, sam_grad, vairational_hessian, woodfisher
from data_tool import ConcatDatasets, RelabeledDataset, flatten_subset

# REF: train model from scratch
class BuildLearn(BasicUnlearnSchema):
    def __init__(self, logpath, logname) -> None:
        super(BuildLearn, self).__init__(logpath, logname)

    def __get_name__(self) -> str:
        return "ORG"  # 'original'
    
    def model_unlearn(self, model:DeepModels, data, batch_size, device, ckpt_path, **kwargs):
        is_train = kwargs.get('is_train', False)
        self.logger.info(f" ###### Learn a model by training from scratch ######")
        model_path = kwargs.get('model_path', None)
        if is_train or model_path is None or len(model_path) <= 0 or not path.exists(model_path):
            self._train_(model, data['train'], data['valid'], data['test'], ckpt_path + self.__get_name__(), batch_size, device)
        else:
            if not path.exists(model_path):
                raise Exception(f"invalid model path: {model_path}")
            model.load(model_path)
            self.logger.info(f"==> Load model from {model_path}")
        self.logger.info(f"==> Model training done!")
        return model

## REF: gound truth: retrain from scratch over retain dataset
class UnlearnRetrain(BasicUnlearnSchema):
    def __init__(self,logpath, logname) -> None:
        super(UnlearnRetrain, self).__init__(logpath, logname)
    
    def __get_name__(self) -> str:
        return "RT"  # 'retrain'
    
    def model_unlearn(self, model:DeepModels, data, batch_size, device, ckpt_path, **kwargs):
        self.logger.info(f" ###### Re-train a model over retain data ######")
        if kwargs.get('reinit', False):
            model._model.apply(kaiming_weights_init)
        model_ = self._train_(model, data['retain'], data['valid'], data['test'], ckpt_path + self.__get_name__(), batch_size, device)
        self.logger.info(f"==> Model unlearning done!")
        return model_
        
## REF Finetune over the retain dataset
class UnlearnFinetune(BasicUnlearnSchema):
    def __init__(self, logpath, logname) -> None:
        super(UnlearnFinetune, self).__init__(logpath, logname)
    
    def __get_name__(self) -> str:
        return "FT"  #'finetune'
    
    def model_unlearn(self, model, data, batch_size, device, ckpt_path, **kwargs):
        self.logger.info(f" ###### Finetune the original model over retain data ######")
        model_ = self._train_(model, data['retain'], data['valid'], data['test'], ckpt_path + self.__get_name__(), batch_size, device)
        self.logger.info(f"==> Model unlearning done!")
        return model_


class UnlearnSparseL1(BasicUnlearnSchema):
    def __init__(self, logpath, logname) -> None:
        super(UnlearnSparseL1, self).__init__(logpath, logname)
    
    def __get_name__(self) -> str:
        return "L1_sparse"

    def model_unlearn(self, model, data, batch_size, device, ckpt_path, **kwargs):
        self.logger.info(f" ###### Unlearn the model using L1 sparse regularization ######")
        check_sparsity(model._model)
        model.params['regularization'] = 'l1'
        model.params['dynamic_regular'] = True
        model_ = self._train_(model, data['retain'], data['valid'], data['test'], ckpt_path + self.__get_name__(), batch_size, device)
        self.logger.info(f"==> Model unlearning done!")
        check_sparsity(model_._model)
        return model_
    

## REF: random label unlearning: Laura Graves, et. al. AAAI 2021
class UnlearnRandomLabels(BasicUnlearnSchema):
    def __init__(self, logpath, logname) -> None:
        super(UnlearnRandomLabels, self).__init__(logpath, logname)
    
    def __get_name__(self) -> str:
        return "RL"  # 'random_label'
    
    def __get_random_label__(self, true_labels, num_classes):
        cand_classes = set(np.arange(num_classes))
        random_labels = list()
        for i in range(len(true_labels)):
            random_labels.append(np.random.choice(list(cand_classes - set([true_labels[i]]))))
        return random_labels
        
    def model_unlearn(self, model, data, batch_size, device, ckpt_path, **kwargs):
        self.logger.info(f" ###### Tune the original model using forget data with replacement of random label ######")
        num_classes = kwargs.get('num_classes', len(np.unique(data['train'].dataset.targets)))
        flatten_data = flatten_subset(deepcopy(data['forget']))
        new_lables = self.__get_random_label__(flatten_data.targets, num_classes)
        relabel_data = RelabeledDataset(flatten_data.dataset, flatten_data.targets, new_lables)
        model_ = self._train_(model, relabel_data, data['valid'], data['test'], ckpt_path + self.__get_name__(), batch_size, device)
        self.logger.info(f"==> Model unlearning done!")
        return model_


# REF: SALUN: EMPOWERING MACHINE UNLEARNING VIA GRADIENT-BASED WEIGHT SALIENCY IN BOTH IMAGE CLASSIFICATION AND GENERATION  iclr-24
class UnlearnSalunRL(UnlearnRandomLabels):
    def __init__(self, logpath, logname) -> None:
        super(UnlearnSalunRL, self).__init__(logpath, logname)
    
    def __get_name__(self) -> str:
        return "SALUN-RL"  # 'salun'
        
    def model_unlearn(self, model:DeepModels, data, batch_size, device, ckpt_path, **kwargs):
        self.logger.info(f" ###### Saliency Unlearning (SALUN) for the model combined with RL ######")
        
        mask_threshold = kwargs.get('mask_threshold', 0.5)
        num_classes = kwargs.get('num_classes', len(np.unique(data['train'].dataset.targets)))
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optimizer_picker(model.params['optim'], model._model.parameters(), lr=model.params['maxlr'], 
                                     momentum=model.params['momentum'], weight_decay=model.params['weight_decay'])
        forget_loader = DataLoader(data['forget'], batch_size=batch_size, shuffle=True)        
        mask = generate_weight_mask(model._model, forget_loader, criterion, optimizer, mask_threshold, device)

        model.params['dynamic_regular'] = True
        flatten_data = flatten_subset(deepcopy(data['forget']))
        new_lables = self.__get_random_label__(flatten_data.targets, num_classes)
        relabel_data = RelabeledDataset(flatten_data.dataset, flatten_data.targets, new_lables)
        new_train_data = ConcatDatasets([data['retain'], relabel_data])
        
        model_ = self._train_(model, new_train_data, data['valid'], data['test'], ckpt_path + self.__get_name__(), 
                              batch_size, device, mask)
        self.logger.info(f"==> Model unlearning done!")
        return model_


## REF: negative gradient (gradient ascent) unlearning:  Goel et al.
class UnlearnGradientAscent(BasicUnlearnSchema):
    def __init__(self, logpath, logname) -> None:
        super(UnlearnGradientAscent, self).__init__(logpath, logname)
    
    def __get_name__(self) -> str:
        return "GA"  # 'gradient_ascent'
    
    def model_unlearn(self, model:DeepModels, data, batch_size, device, ckpt_path, **kwargs):
        assert model.params is not None, "Please set the parameter configuration first"
        self.logger.info(f" ###### Tune the original model using forget data with negative gradient ######")
        model.params['loss_sign'] = -1.0
        model_ = self._train_(model, data['forget'], data['valid'], data['test'], ckpt_path + self.__get_name__(), batch_size, device)
        model_.params['loss_sign'] = 1.0
        self.logger.info(f"==> Model unlearning done!")
        return model_

## REF: Weighted gradient ascent (WGA) following the idea in dataloader.py
class UnlearnWeightedGradientAscent(BasicUnlearnSchema):
    def __init__(self, logpath, logname) -> None:
        super(UnlearnWeightedGradientAscent, self).__init__(logpath, logname)
    
    def __get_name__(self) -> str:
        return "WGA"  # 'weighted_gradient_ascent'
    
    def model_unlearn(self, model:DeepModels, data, batch_size, device, ckpt_path, **kwargs):
        """
        Weighted GA loss: for each sample i with CE loss l_i, use weight w_i = exp(-l_i)^beta
        and optimize -(w_i * l_i) averaged over batch to ascend loss on forget set while focusing on easier samples.
        Args:
          beta: weighting exponent (default 5.0 to mirror dataloader.py)
        """
        assert model.params is not None, "Please set the parameter configuration first"
        beta = kwargs.get('beta', kwargs.get('wga_beta', 5.0))
        self.logger.info(f" ###### Weighted Gradient Ascent on forget data (beta={beta}) ######")
        # Save previous flags
        prev_use_wga = model.params.get('use_wga', False)
        prev_wga_beta = model.params.get('wga_beta', 5.0)
        prev_loss_sign = model.params.get('loss_sign', 1.0)

        # Enable WGA branch in the training loop (src/learner.py)
        model.params['use_wga'] = True
        model.params['wga_beta'] = beta
        # Keep loss_sign = +1.0 since WGA branch already negates CE internally
        model.params['loss_sign'] = 1.0

        model_ = self._train_(model, data['forget'], data['valid'], data['test'], ckpt_path + self.__get_name__(), batch_size, device)

        # Restore previous flags
        model_.params['use_wga'] = prev_use_wga
        model_.params['wga_beta'] = prev_wga_beta
        model_.params['loss_sign'] = prev_loss_sign
        self.logger.info(f"==> Model unlearning done!")
        return model_

## REF: retrain the last k layers:  Goel et al. arXiv 2022
class UnlearnLastKLayer(BasicUnlearnSchema):
    def __init__(self, logpath, logname) -> None:
        super(UnlearnLastKLayer, self).__init__(logpath, logname)

    def __get_name__(self) -> str:
        return "LKL"  # 'last_k_layer'
    
    def model_unlearn(self, model:DeepModels, data, batch_size, device, ckpt_path, **kwargs):
        self.logger.info(f" ###### Retrain the last-k layer of the original model over the retain data ######")
        last_k = kwargs.get('last_k', 1)
        re_init = kwargs.get('re_init', True)
        self.logger.info(f" ###### reset the final {last_k} layers of the model ######")
        model._model = reset_final_layers(model._model, last_k, self.logger, re_init)
        model_ = self._train_(model, data['retain'], data['valid'], data['test'], ckpt_path + self.__get_name__(), batch_size, device)
        self.logger.info(f"==> Model unlearning done!")
        return model_


## REF:  Fisher / variational approximation based unlearn. Golatkar et al.  CVPR 2021
class UnlearnFisherApprox(BasicUnlearnSchema):
    def __init__(self, logpath, logname) -> None:
        super(UnlearnFisherApprox, self).__init__(logpath, logname)
    
    def __get_name__(self) -> str:
        return "FARX"  # 'fisher_approx'
    
    def model_unlearn(self, model:DeepModels, data, batch_size, device, ckpt_path, **kwargs):
        lamb = kwargs.get('lamb', 1e-9)
        lamb = 1e-9 if lamb <= 0 else lamb
        loss_name  = kwargs.get('lossfn', 'ce')
        forget_class = kwargs.get('forget_classes', None)
        fisher_type = kwargs.get('fisher_type', 'golaker')
        is_class_forget = False if forget_class is None else True
        
        self.logger.info(f" ###### Tune the original model using forget data with {(fisher_type)} Fisher approximation ######")
        criterion = loss_picker(loss_name)
        num_classes = kwargs.get('num_classes', len(np.unique(data['train'].dataset.targets)))
        for param in model._model.parameters():
            param.data0 = deepcopy(param.data.clone())
        
        loader = DataLoader(data['retain'], batch_size=1, shuffle=True)
        start_time_ = time.time()
        if fisher_type == 'golaker':
            self.logger.info(" ######### golaker-type fisher-hession approx")
            model._model = fisher_hessian(model._model, loader, criterion, device)
        elif fisher_type == 'variate':
            self.logger.info(" ######### variational hession approx")
            # model._model = vairational_hessian(model._model, loader, criterion, device)
            raise NotImplementedError("Variational Hessian Approximation is not implemented yet!")
        
        self.logger.info(f" ##### add noise for model parameters #####")
        ## apply fisher noise for model parameters:
        for _, param in model._model.named_parameters():
            mu, var = get_mean_var(param, num_classes, is_class_forget, forget_class, lamb=lamb)
            param.data = mu + var.sqrt() * torch.empty_like(param.data).normal_()
        self.logger.info(f"==> Model unlearning done @ [{time.time() - start_time_}] s!")
        return model

# REF:  Singh, Sidak Pal, and Dan Alistarh. "Woodfisher: Efficient second-order approximation for neural network compression." 
#       Advances in Neural Information Processing Systems 33 (2020): 18098-18109.
class UnlearnWoodFisher(BasicUnlearnSchema):
    def __init__(self, logpath, logname) -> None:
        super(UnlearnWoodFisher, self).__init__(logpath, logname)
    
    def __get_name__(self) -> str:
        return "WFARX"
    
    def model_unlearn(self, model:DeepModels, data, batch_size, device, ckpt_path, **kwargs):
        loss_name  = kwargs.get('lossfn', 'ce')
        Ns = kwargs.get('Ns', len(data['retain']))
        alpha = kwargs.get('alpha', 0.2)  # unlearn noice, similar to the lambda in fisher approximation
        
        criterion = loss_picker(loss_name)
        retain_loader = DataLoader(data['retain'], batch_size=batch_size, shuffle=True)
        forget_loader = DataLoader(data['forget'], batch_size=batch_size, shuffle=False)
        
        params = []
        for param in model._model.parameters():
            params.append(param.view(-1))
        retain_grad = torch.zeros_like(torch.cat(params)).to(device)
        forget_grad = torch.zeros_like(torch.cat(params)).to(device)
        
        total_f, total_r = 0, 0
        self.logger.info(f"==> Compute the gradient difference between forget and retain data!")
        model._model.eval()
        for inputs, targets in forget_loader:
            model._model.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model._model(inputs)
            loss = model.criterion(outputs, targets)
            forget_grad += sam_grad(model._model, loss) * inputs.size(0)
            total_f += inputs.size(0)

        for inputs, targets in retain_loader:
            model._model.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model._model(inputs)
            loss = model.criterion(outputs, targets)
            retain_grad += sam_grad(model._model, loss) * inputs.size(0)
            total_r += inputs.size(0)
        
        forget_grad /= total_f + total_r
        retain_grad *= total_f / ((total_f + total_r) * total_r)
        
        self.logger.info(f"==> Compute the perturbation using woodfisher!")
        grad_diff = forget_grad - retain_grad
        perturb = woodfisher(model._model, retain_loader, criterion, grad_diff, Ns, device)
        
        curr = 0
        model_ = deepcopy(model._model)
        self.logger.info(f"==> Apply the perturbation!")
        with torch.no_grad():
            for param in model_.parameters():
                length = param.view(-1).shape[0]
                param += alpha * perturb[curr : curr + length].view(param.shape)
                curr += length
        self.logger.info(f"==> Model unlearning done!")
        
        model._model = model_
        return model
        

# REF:  certified removal with differential privacy: Chuan Guo et al.  ICML 2018
class UnlearnCertifiedRemoval(BasicUnlearnSchema):
    def __init__(self, logpath, logname) -> None:
        super(UnlearnCertifiedRemoval, self).__init__(logpath, logname)
    
    def __get_name__(self) -> str:
        return "CR"   # 'certified_removal'
    
    def model_unlearn(self, model:DeepModels, data, batch_size, device, ckpt_path, **kwargs):
        self.logger.info(f" ###### Tune the original model using certified removal with differential privacy ######")
        num_classes = kwargs.get('num_classes', len(np.unique(data['train'].targets)))
        
        train_loader = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
        forget_loader = DataLoader(data['forget'], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(data['test'], batch_size=batch_size, shuffle=False)
        
        model_name = kwargs.get('arch', 'resnet')
        lamb_reg = kwargs.get('lamb', 1e-5)
        train_mode = kwargs.get('train_mode', "ovr")
        train_seperate = kwargs.get('train_seperate', False)
        sample_rate = kwargs.get('sample_rate', 1.0)
        std = kwargs.get('std', 1.0)
        epochs = kwargs.get('epochs', 100)
        tol = kwargs.get('tol', 1e-5)
        
        start_time_ = time.time()
        self.logger.info(f" ##### certified removal training #####")
        cr_model = CRModel(num_classes, DeepModels.to_extractor(model, model_name, device=device))
        cr_model.train(train_loader, lamb_reg, train_mode, sample_rate, std, epochs, tol, train_seperate, device)
        self.logger.info(f"==> Model unlearning done @ [{time.time() - start_time_}] s!")
        
        acc_train = cr_model.test_acc(train_loader)
        auc_train = cr_model.test_auc(train_loader)
        acc_forget = cr_model.test_acc(forget_loader)
        auc_forget = cr_model.test_auc(forget_loader)
        acc_test = cr_model.test_acc(test_loader)
        auc_test = cr_model.test_auc(test_loader)
        self.logger.info("==> [ BEFORE REMOVAL ] [Remain, Forget, Test]" + 
                            f"ACC: {acc_train:.4f}, {acc_forget:.4f}, {acc_test:.4f} |" +
                            f"AUC: {auc_train:.4f}, {auc_forget:.4f}, {auc_test:.4f}")
        
        cr_remove_proc = CertifiedRemoval()
        cr_model_unlearn = cr_remove_proc.remove(cr_model, train_loader, forget_loader, lamb_reg, train_mode, batch_size, device)
        return cr_model_unlearn


# REF: Zachary Izzo, Mary Anne Smart, Kamalika Chaudhuri, and James Zou. Approximate data deletion from machine learning models. AIStats, 2021.
#      Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. ICML, 2017.
class UnlearnInfluence(BasicUnlearnSchema):
    def __init__(self, logpath, logname) -> None:
        super(UnlearnInfluence, self).__init__(logpath, logname)
    
    def __get_name__(self) -> str:
        return "UI"  # 'influence_unlearn'
    
    @staticmethod
    def freeze_all_params(model):
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_last_k_params(model, last_k):
        params = list(model.parameters())
        for param in params[-last_k:]:
            param.requires_grad = True
    
    def model_unlearn(self, model:DeepModels, data, batch_size, device, ckpt_path, **kwargs):
        self.logger.info(f" ###### Influence function based Unlearning for model weight update ######")
        
        damp = kwargs.get('damp', 0.01)
        scale = kwargs.get('scale', 25.0)
        recursion_depth = kwargs.get('recursion_depth', 1000)
        r_averaging = kwargs.get('r_averaging', 1)
        last_k = kwargs.get('last_k', -1)   ## the number of layers for parameter updating
        
        start_time_ = time.time()
        forget_loader = DataLoader(data['forget'], batch_size=batch_size, shuffle=False)
        train_loader = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
        
        count = 0
        delta_ = None
        model_ = deepcopy(model._model)
        
        if last_k > 0:
            self.freeze_all_params(model_)
            self.unfreeze_last_k_params(model_, last_k)
        
        model_ = model_.to(device)
        for e_i, (inputs, targets) in tqdm(enumerate(forget_loader)):
            ihvp_i = ifs.s_test_sample(model_, inputs, targets, train_loader, device, damp, scale, recursion_depth, r_averaging)
            # ihvp_i = ifs.calc_s_test_single(model_, inputs, targets, train_loader, device, damp, scale, recursion_depth, r_averaging)
            if delta_ is None:
                delta_ = ihvp_i
            else:
                for i, d in enumerate(ihvp_i):
                    delta_[i] += ihvp_i[i] * inputs.size(0)
            count += inputs.size(0)
        
        if last_k > 0:
            params = list(model_.parameters())
            for param, d in zip(params[-last_k:], delta_):
                param.data.add_(d / count)
        else:
            for param, d in zip(model_.parameters(), delta_):
                param.data.add_(d / count)
        self.logger.info(f"==> Influence Unlearning done @ [{time.time() - start_time_}] s!")
        
        model._model = model_
        return model
import numpy as np
from abc import abstractmethod
import random
import torch
from torch.utils.data import DataLoader
from learner import loss_picker
from torch.nn import functional as F


class ExtractBase(object):
    def __init__(self, stacked=False, include_posterior=False):
        self.stacked = stacked
        self.include_posterior = include_posterior
    
    @abstractmethod
    def extract_feature(self, model, dataset, device):
        raise NotImplementedError

    def _get_predict_proba_(self, model, dataset, batch_size, logits=True, device='cuda'):
        model.eval()
        model.to(device)
        
        res = []
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                output = model(inputs)
                if logits:
                    post_ = output
                else:
                    post_ = F.softmax(output, dim=1)
                res.append(post_.detach().cpu())
            res = torch.cat(res, dim=0)
        return np.array(res)
    
    def _feature_extent_(self, features_set, layers, proba):
        feature = None
        if self.stacked and len(layers) > 1:
            ## all the last k-th layer features are concatenated
            feature = np.concatenate(list(features_set.values()), axis=1)
        else:
            ## only select the last k-th layer feature
            feature = features_set[layers[-1]]
            
        if self.include_posterior:
            feature = np.concatenate([feature, proba], axis=1)
            # self.logger.info(f"attack model feature: add posterior")
        
        return feature

## Using the loss information of the model
class ExtractLoss(ExtractBase):
    def __init__(self, batch_size=32, lossfn='ce'):
        super(ExtractLoss, self).__init__(stacked=False, include_posterior=False)
        self.batch_size = batch_size
        self.lossfn = lossfn
        
    def extract_feature(self, model, dataset, device='cuda'):
        feature = None
        logits = self._get_predict_proba_(model, dataset, self.batch_size, device=device)
        criterion = loss_picker(self.lossfn).to(device)
        criterion.reduction = 'none'
        feature = criterion(torch.tensor(logits).to(device), torch.tensor(dataset.targets).to(device)).cpu().numpy()
        feature = feature.reshape((-1, 1))
        return feature


## Using the entropy for the prediction
class ExtractEntropy(ExtractBase):
    def __init__(self, batch_size=32):
        super(ExtractEntropy, self).__init__(stacked=False, include_posterior=False)
        self.batch_size = batch_size
    
    def extract_feature(self, model, dataset, device='cuda'):
        probs = self._get_predict_proba_(model, dataset, self.batch_size, logits=False, device=device)
        probs = torch.tensor(probs).to(device)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).cpu().numpy()
        entropy = entropy.reshape((-1, 1))
        return entropy


## Using the posterior information of the model
class ExtractPosterior(ExtractBase):
    def __init__(self, batch_size=32):
        super(ExtractPosterior, self).__init__(stacked=False, include_posterior=True)
        self.batch_size = batch_size
    
    def extract_feature(self, model, dataset, device='cuda'):
        # only return the posteriors of the model
        return self._get_predict_proba_(model, dataset, self.batch_size, device=device)

## Using the output features of the last k linear layers of the model
class ExtractLastLinears(ExtractBase):
    def __init__(self, last_k=1, batch_size=32, stacked=False, include_posterior=False):
        super(ExtractLastLinears, self).__init__(stacked, include_posterior)
        self.last_k = last_k
        self.batch_size = batch_size
        
    def extract_feature(self, model, dataset, device='cuda'):
        ## only return the linear features of the last few layers
        layer_outputs = {}
        def hook(name):
            def hook_fn(module, input, output):
                layer_outputs[name] = output.detach()
            return hook_fn
        
        count = 0
        hooked_layers = list()
        for name, layer in list(model.named_modules())[::-1]:
            if isinstance(layer, torch.nn.Linear) and not isinstance(layer, torch.nn.Dropout):
                if self.stacked:
                    hooked_layers.append(name)
                    layer.register_forward_hook(hook(name))
                count += 1
            if count == self.last_k:
                if not self.stacked:
                    hooked_layers.append(name)
                    layer.register_forward_hook(hook(name))
                break
        assert len(hooked_layers) > 0, "No linear layer is found."
        
        model.eval()
        _feature_set_ = {k_name :list()  for k_name in hooked_layers}
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for input, _ in loader:
                input = input.to(device)
                logits = model(input)
                
                for name, output in layer_outputs.items():
                    _feature_set_[name].append(output.cpu().numpy().squeeze())
                
        for k_name in hooked_layers:
            _feature_set_[k_name] = np.concatenate(_feature_set_[k_name], axis=0)
            shape_ = _feature_set_[k_name].shape
            if len(shape_) > 2:
                _feature_set_[k_name] = _feature_set_[k_name].reshape((shape_[0], -1))
        
        proba = self._get_predict_proba_(model, dataset, self.batch_size, device=device)
        feature = self._feature_extent_(_feature_set_, hooked_layers, proba)
        return feature

## Using the output features of the last k non-linear layers of the model
class ExtractNonLinears(ExtractBase):
    def __init__(self, last_k=1, batch_size=32, stacked=False, include_posterior=False):
        super(ExtractNonLinears, self).__init__(stacked, include_posterior)
        self.last_k = last_k
        self.batch_size = batch_size
    
    def extract_feature(self, model, dataset, device='cuda'):
        ## only return the non-linear features of the middle layers of a DNN model
        layer_outputs = {}
        def hook(name):
            def hook_fn(module, input, output):
                layer_outputs[name] = output.detach()
            return hook_fn
        
        count = 0
        hooked_layers = list()
        for name, layer in list(model.named_modules())[::-1]:
            if not isinstance(layer, torch.nn.Linear) and not isinstance(layer, torch.nn.Dropout):
                if self.stacked:
                    hooked_layers.append(name)
                    layer.register_forward_hook(hook(name))
                count += 1
            if count == self.last_k:
                if not self.stacked:
                    hooked_layers.append(name)
                    layer.register_forward_hook(hook(name))
                break
        
        model.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        _feature_set_ = {k_name :list()  for k_name in hooked_layers}
        with torch.no_grad():
            for input, _ in loader:
                input = input.to(device)
                logits = model(input)

                for name, output in layer_outputs.items():
                    _feature_set_[name].append(output.cpu().numpy().squeeze())
                
        for k_name in hooked_layers:
            _feature_set_[k_name] = np.concatenate(_feature_set_[k_name], axis=0)
            shape_ = _feature_set_[k_name].shape
            if len(shape_) > 2:
                _feature_set_[k_name] = _feature_set_[k_name].reshape((shape_[0], -1))
        
        proba = self._get_predict_proba_(model, dataset, self.batch_size, device=device)
        feature = self._feature_extent_(_feature_set_, hooked_layers, proba)
        return feature

## Using the output features of the last k layers (linear and non-linear) of the model
class ExtractMixLayers(ExtractBase):
    def __init__(self, last_k, batch_size=32, stacked=False, include_posterior=False):
        super(ExtractMixLayers, self).__init__(stacked, include_posterior)
        self.last_k = last_k
        self.batch_size = batch_size
    
    def extract_feature(self, model, dataset, device='cuda'):
        ## only return the last k linear features and the non-linear features of the middle layers of a DNN model
        layer_outputs = {}
        def hook(name):
            def hook_fn(module, input, output):
                layer_outputs[name] = output.detach()
            return hook_fn

        count = 0
        hooked_layers = list()
        for name, layer in list(model.named_modules())[::-1]:
            if not isinstance(layer, torch.nn.Dropout):
                hooked_layers.append(name)
                layer.register_forward_hook(hook(name))
                count += 1
            if count == self.last_k:
                break
        assert len(hooked_layers) > 0, "No layer is found."

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        model.eval()
        
        _feature_set_ = {k_name :list()  for k_name in hooked_layers}
        with torch.no_grad():
            for input, _ in loader:
                input = input.to(device)
                logits = model(input)

                for name, output in layer_outputs.items():
                    _feature_set_[name].append(output.cpu().numpy().squeeze())
                
        for k_name in hooked_layers:
            _feature_set_[k_name] = np.concatenate(_feature_set_[k_name], axis=0)
            shape_ = _feature_set_[k_name].shape
            if len(shape_) > 2:
                _feature_set_[k_name] = _feature_set_[k_name].reshape((shape_[0], -1))
        
        proba = self._get_predict_proba_(model, dataset, self.batch_size, device=device)
        feature = self._feature_extent_(_feature_set_, hooked_layers, proba)        
        return feature

## Using the norm of gradients of the model w.r.t. the input
class ExtractGradientNorm(ExtractBase):
    def __init__(self, lossfn='ce', batch_size=32):
        super(ExtractGradientNorm, self).__init__(stacked=False, include_posterior=False)
        self.lossfn = lossfn
        self.batch_size = batch_size
    
    def extract_feature(self, model, dataset, device='cuda'):
        criterion = loss_picker(self.lossfn)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        logits = []
        gradient_norms = []
        for inputs, targets in loader:
            model.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad_(True)
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            
            input_gradients = inputs.grad.detach()
            grad_norms = torch.norm(input_gradients.view(input_gradients.size(0), -1), dim=1)
            gradient_norms.append(grad_norms.cpu().numpy())
            
            logit_ = F.softmax(output, dim=1)
            logits.append(logit_.detach().cpu())
            
            # grad_norm = torch.norm(inputs.grad.data)
            # gradient_norms.append(grad_norm.item())
            # # Reset the gradients in the batch data for the next sample
            # inputs.grad.data.zero_()
        features = np.concatenate(gradient_norms)
        
        if self.include_posterior:
            logits = np.concatenate(logits)
            features = np.concatenate([features.reshape(-1, 1), logits], axis=1)
        
        return features

## Using the gradient features of the last k layers (linear and non-linear) of the model
class ExtractGradients(ExtractBase):
    def __init__(self, lossfn='ce', batch_size=32, last_k=1, stacked=False, include_posterior=False):
        super(ExtractGradients, self).__init__(stacked, include_posterior)
        self.lossfn = lossfn
        self.batch_size = batch_size
        self.last_k = last_k
    
    def register_hooks(self, model, layer_indices):
        """
        Register hooks on the specified layers to store their outputs and gradients.
        """
        grads = {}
        
        def save_grad(layer, grad_in, grad_out):
            grads[layer] = grad_out[0]

        hooks = []
        for idx in layer_indices:
            layer = list(model.children())[idx]
            hooks.append(layer.register_backward_hook(save_grad))
        
        return hooks, grads
    
    def extract_feature(self, model, dataset, device='cuda'):
        # only return the gradients of the last k linear features and the non-linear features of the middle layers of a DNN model
        # is_stacked = self.option.stacked
        # include_posterior = self.option.include_posterior
        # batch_size = self.option.batch_size
        # loss_func = self.option.lossfn
        
        # last_k = self.option.last_k_attack
        # model = model._model
        # model = model.to(model.device)
        # device = model.device
        
        # layer_outputs = {}
        # def hook(name):
        #     def hook_fn(module, grad_input, grad_output):
        #         layer_outputs[name] = grad_output[0].detach()
        #     return hook_fn

        # count = 0
        # hooked_layers = list()        
        # for name, layer in list(model.named_modules())[::-1]:
        #     if not isinstance(layer, torch.nn.Dropout):
        #         if self.stacked:
        #             hooked_layers.append(name)
        #             layer.register_forward_hook(hook(name))
        #         count += 1
        #     if count == self.last_k:
        #         if not self.stacked:
        #             hooked_layers.append(name)
        #             layer.register_forward_hook(hook(name))
        #         break
        
        # criterion = loss_picker(self.lossfn).to(device)
        # loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # _feature_set_ = {k_name :list()  for k_name in hooked_layers}
        # for input, target in loader:
        #     input, target = input.to(device), target.to(device)
        #     output = model(input)
        #     loss = criterion(output, target)
        #     loss.backward()

        #     for name, output in layer_outputs.items():
        #         _feature_set_[name].append(output.cpu().numpy().squeeze())

        # for k_name in hooked_layers:
        #     _feature_set_[k_name] = np.concatenate(_feature_set_[k_name], axis=0)
        #     shape_ = _feature_set_[k_name].shape
        #     if len(shape_) > 2:
        #         _feature_set_[k_name] = _feature_set_[k_name].reshape((shape_[0], -1))
        
        
        model_childs = list(model.children())
        if self.stacked:
            layer_indices = list(range(len(model_childs)))[-self.last_k:]
        else:
            layer_indices = [list(range(len(model_childs)))[-self.last_k]]
        hooks, grads = self.register_hooks(model, layer_indices)
    
        model.eval()
        criterion = loss_picker(self.lossfn).to(device)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        logits = list()
        grads_all = list()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.requires_grad_(True)
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()

            logit_ = F.softmax(output, dim=1)
            logits.append(logit_.detach().cpu())
        
            for b in range(inputs.size(0)):
                concatenated_grads = []
                for idx in layer_indices:
                    mod_layer_ = model_childs[idx]
                    if mod_layer_ in grads:
                        concatenated_grads.append(grads[mod_layer_][b].detach().cpu().numpy().flatten())
                grads_all.append(np.concatenate(concatenated_grads))
            
        features = np.stack(grads_all)
        if self.include_posterior:
            logits = np.concatenate(logits)
            features = np.concatenate([features, logits], axis=1)
                
        return features


def get_extractor(attack_feature, **kwargs):
    if attack_feature == 'entropy':
        batch_size = kwargs.get('batch_size', 32)
        return ExtractEntropy(batch_size)
    if attack_feature == 'loss':
        batch_size = kwargs.get('batch_size', 32)
        lossfn = kwargs.get('lossfn', 'ce')
        return ExtractLoss(batch_size, lossfn)
    elif attack_feature == 'posterior':
        batch_size = kwargs.get('batch_size', 32)
        return ExtractPosterior(batch_size)
    elif attack_feature == 'linear':
        last_k = kwargs.get('last_k', 1)
        batch_size = kwargs.get('batch_size', 32)
        stacked = kwargs.get('stacked', False)
        include_posterior = kwargs.get('include_posterior', False)
        return ExtractLastLinears(last_k, batch_size, stacked, include_posterior)
    elif attack_feature == 'nonlinear':
        last_k = kwargs.get('last_k', 1)
        batch_size = kwargs.get('batch_size', 32)
        stacked = kwargs.get('stacked', False)
        include_posterior = kwargs.get('include_posterior', False)
        return ExtractNonLinears(last_k, batch_size, stacked, include_posterior)
    elif attack_feature == 'mixlayer':
        last_k = kwargs.get('last_k', 1)
        batch_size = kwargs.get('batch_size', 32)
        stacked = kwargs.get('stacked', False)
        include_posterior = kwargs.get('include_posterior', False)
        return ExtractMixLayers(last_k, batch_size, stacked, include_posterior)
    elif attack_feature == 'gradient':
        lossfn = kwargs.get('lossfn', 'ce')
        last_k = kwargs.get('last_k', 1)
        batch_size = kwargs.get('batch_size', 32)
        stacked = kwargs.get('stacked', False)
        include_posterior = kwargs.get('include_posterior', False)
        return ExtractGradients(lossfn, batch_size, last_k, stacked, include_posterior)
    elif attack_feature == 'gradientnorm':
        lossfn = kwargs.get('lossfn', 'ce')
        batch_size = kwargs.get('batch_size', 32)
        return ExtractGradientNorm(lossfn, batch_size)
    elif attack_feature == 'lira':
        return LIRAFExtractor(
            batch_size=kwargs.get('batch_size', 256),
            lossfn=kwargs.get('lossfn', 'ce'),
            last_k=kwargs.get('last_k', 1),
            stacked=kwargs.get('stacked', False),
            include_posterior=kwargs.get('include_posterior', True),
            shadow_models=kwargs.get('shadow_models', 5),
            augmentations=kwargs.get('augmentations', 10),
            shift=kwargs.get('shift', 4)
        )
    else:
        raise ValueError(f"Unknown extractor name: {attack_feature}")
class LIRAFExtractor(ExtractBase):
    """LiRA特征提取器（支持数据增强）"""
    
    def __init__(self, batch_size=256, lossfn='ce', last_k=1, 
                 stacked=False, include_posterior=True,
                 shadow_models=5, augmentations=10, shift=4):
        super().__init__(stacked=stacked, include_posterior=include_posterior)
        self.batch_size = batch_size
        self.lossfn = lossfn
        self.last_k = last_k
        self.shadow_models = shadow_models
        self.augmentations = augmentations
        self.shift = shift
        
    def extract_feature(self, model, dataset, device):
        """提取LiRA特征（包含数据增强）"""
        model.eval()
        all_features = []
        
        # 创建数据加载器
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                
                # 对每个样本进行多次增强
                batch_features = []
                for i in range(len(data)):
                    sample = data[i:i+1]
                    sample_target = target[i:i+1]
                    sample_features = self._get_sample_augmentation_features(
                        model, sample, sample_target, device)
                    batch_features.append(sample_features)
                
                all_features.extend(batch_features)
                
        # 转换为numpy数组
        features_array = np.array(all_features)
        return features_array
    
    def _get_sample_augmentation_features(self, model, sample, sample_target, device):
        """对单个样本进行多次增强并收集特征"""
        augmentation_features = []
        
        # 基础预测（无增强）
        with torch.no_grad():
            output = model(sample)
            confidence = torch.softmax(output, dim=1).cpu().numpy()[0]
            augmentation_features.extend(confidence)
        
        # 数据增强版本
        for aug_idx in range(self.augmentations - 1):
            # 随机增强
            augmented_sample = self._random_augment(sample.clone())
            
            with torch.no_grad():
                aug_output = model(augmented_sample.to(device))
                aug_confidence = torch.softmax(aug_output, dim=1).cpu().numpy()[0]
                augmentation_features.extend(aug_confidence)
        
        return augmentation_features
    
    def _random_augment(self, sample):
        """应用随机数据增强（基于LiRA论文）"""
        # 随机水平翻转
        if random.random() > 0.5:
            sample = torch.flip(sample, [3])
        
        # 随机平移（使用反射填充）
        shift_x = random.randint(-self.shift, self.shift)
        shift_y = random.randint(-self.shift, self.shift)
        
        if shift_x != 0 or shift_y != 0:
            # 使用反射填充来处理边界
            pad_x = abs(shift_x)
            pad_y = abs(shift_y)
            sample_padded = F.pad(sample, (pad_y, pad_y, pad_x, pad_x), mode='reflect')
            
            # 计算裁剪区域
            start_x = pad_x + shift_x
            start_y = pad_y + shift_y
            sample = sample_padded[:, :, start_x:start_x+sample.shape[2], 
                                  start_y:start_y+sample.shape[3]]
            
        # 添加轻微噪声
        noise = torch.randn_like(sample) * 0.01
        sample = sample + noise
        
        return sample
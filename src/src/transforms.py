
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from timm import data as timm_data
from torchvision import transforms
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from torchtoolbox.transform import Cutout

class Transforms(nn.Module):
    def __init__(self, mean, std, size=224):
        self.mean = mean
        self.std = std
        self.size = size
    
    def get_transform(self, type='normal', num_classes=10):
        h_params = {"translate_const": 100, "img_mean": (124, 116, 104)}
        transform = None
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        resize_trans = transforms.Resize(size=(self.size, self.size))
        if type in ['normal', 'test']:   ## __call__(input)    'fgsm', 'vat'            
            transform = transforms.Compose([
                transforms.Resize(size=(self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        elif type == 'random':    ## __call__(input)
            transform = transforms.Compose([
                transforms.Pad(padding=4, fill=(125, 123, 113)),
                transforms.RandomCrop(self.size, padding=0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        elif type == 'augafn':    ## __call__(input)            
            transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=30,               # Range of degrees to select from for rotation  # 0
                    translate=(0.1, 0.1),     # Tuple representing the fraction of translation
                    scale=(0.9, 1.1),         # Scaling factors interval
                    shear=(10, 10)            # Shearing factor  # (0.8, 1.2)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        elif type == 'baseaug':   ## __call__(input)            
            transform = transforms.Compose([
                transforms.Resize(size = (256, 256)),
                transforms.RandomCrop(size = (self.size, self.size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        elif type == 'cutout':    ## __call__(input)
            transform = transforms.Compose([
                transforms.RandomCrop(self.size),
                Cutout(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        elif type == 'randerase':    ## __call__(input)
            transform = transforms.Compose([
                transforms.Resize(size=(self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
                transforms.RandomErasing(),
                # timm_data.random_erasing.RandomErasing(
                #     probability=0.5, 
                #     min_area=0.02,
                #     max_area=1 / 3,
                #     min_aspect=0.3,
                #     max_aspect=None,
                #     mode="pixel", #"const",
                #     min_count=1,
                #     max_count=None,
                #     num_splits=0)
                ])
        elif type == 'mixup':     ## __call__(x, target)
            # mixup active if mixup_alpha > 0
            # cutmix active if cutmix_alpha > 0
            transform = timm_data.mixup.Mixup(
                mixup_alpha=1.0,
                cutmix_alpha=0.0,
                cutmix_minmax=None,
                prob=1.0,
                switch_prob=0.0,
                mode="batch",
                correct_lam=True,
                label_smoothing=0.1,
                num_classes=num_classes,
            )
        elif type == 'cutmix':    ## __call__(x, target)
            transform = timm_data.mixup.Mixup(
                mixup_alpha=0.0,
                cutmix_alpha=1.0,
                cutmix_minmax=None,
                prob=0.8,
                switch_prob=0.0,
                mode="batch",
                correct_lam=True,
                label_smoothing=0.1,
                num_classes=num_classes,
            )
        elif type == 'autoaug':   ## __call__(img)
            transform = timm_data.auto_augment.auto_augment_transform(
                config_str="original-mstd0.5", hparams=h_params
            )
        elif type == 'augmix':    ## __call__(img)
            transform = timm_data.auto_augment.augment_and_mix_transform(
                config_str="augmix-m5-w4-d2", hparams=h_params
            )
        elif type == 'randaug':   ## __call__(img)
            transform = timm_data.auto_augment.rand_augment_transform(
                config_str="rand-m3-n2-mstd0.5", hparams=h_params
            )
        else:
            transform = transforms.ToTensor()
            
        if type in ['randaug', 'augmix', 'autoaug']:  # wrapper the tranform with resize and normalization
            transform = transforms.Compose( [
                    resize_trans, transform, transforms.ToTensor(), normalize 
                ])
        return transform
    
    def forward(self, type, x, target=None, num_classes=10):
        transform = self.get_transform(type, num_classes)
        
        x_, target_ = x, target
        if type in ['mixup', 'cutmix']:
            assert target is not None, "Invalid target for mixup or cutmix"
            x_, target_ = transform(x, target)
        else:
            x_ = transform(x)

        return x_, target_
        
    def __call__(self, type, x, target, num_classes=10):
        return self.forward(type, x, target, num_classes)


class OnlineTransforms(nn.Module):
    def __init__(self, mean, std, size=224):
        super(OnlineTransforms, self).__init__()
        self.mean = mean
        self.std = std
        self.size = size
    
    def _get_transform_(self, type='normal', num_classes=None):
        h_params = {"translate_const": 100, "img_mean": (124, 116, 104)}
        transform = None
        normalize = v2.Normalize(mean=self.mean, std=self.std)
        resize_trans = v2.Resize(size=(self.size, self.size))
        
        if type in ['normal', 'test']:
            transform = v2.Compose([
                v2.Resize(size=(self.size, self.size)),
                normalize
            ])
            
            # def transform_batch(x_batch):
            #     res = []
            #     for img in x_batch:
            #         img_pil = TF.to_pil_image(img)
            #         img_ = resize_trans(img_pil)
            #         img_ = TF.to_tensor(img_)
            #         res.append(normalize(img_))
            #     return torch.stack(res)

            # transform = transforms.Compose([
            #     transforms.Lambda(transform_batch)
            # ])
        elif type == 'random':
            # transform = v2.Compose([
            #     # v2.Pad(padding=4, fill=(125, 123, 113)),
            #     v2.RandomCrop(size=self.size, padding=0),
            #     # v2.RandomHorizontalFlip(),
            #     v2.RandomRotation(degrees=15),
            #     normalize
            # ])
            
            transform = v2.Compose([
                v2.RandomResizedCrop(size=self.size, scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                v2.RandomRotation(degrees=15),
                v2.GaussianBlur(kernel_size=(5, 5)),
            ])
            
            # def transform_batch(x_batch):
            #     res = []
            #     for img in x_batch:
            #         img_pil = TF.to_pil_image(img)
            #         img_ = transforms.Pad(padding=4, fill=(125, 123, 113))(img_pil)
            #         img_ = transforms.RandomCrop(self.size, padding=0)(img_)
            #         img_ = transforms.RandomHorizontalFlip()(img_)
            #         img_ = TF.to_tensor(img_)
            #         img_ = normalize(img_)
            #         res.append(normalize(img_))
            #     return torch.stack(res)
            
            # transform = transforms.Compose([
            #     transforms.Lambda(transform_batch)
            # ])
        elif type == 'augafn':
            transform = v2.Compose([
                v2.RandomAffine(
                    degrees=30,               # Range of degrees to select from for rotation  # 0
                    translate=(0.1, 0.1),     # Tuple representing the fraction of translation
                    scale=(0.9, 1.1),         # Scaling factors interval
                    shear=(10, 10)            # Shearing factor  # (0.8, 1.2)
                ),
                v2.RandomHorizontalFlip(),
                normalize
            ])
            # _affine_ = transforms.RandomAffine(
            #     degrees=30,               # Range of degrees to select from for rotation  # 0
            #     translate=(0.1, 0.1),     # Tuple representing the fraction of translation
            #     scale=(0.9, 1.1),         # Scaling factors interval
            #     shear=(10, 10)            # Shearing factor  # (0.8, 1.2)
            # )
            # def transform_batch(x_batch):
            #     res = []
            #     for img in x_batch:
            #         img_pil = TF.to_pil_image(img)
            #         img_ = _affine_(img_pil)
            #         img_ = transforms.RandomHorizontalFlip()(img_)
            #         img_ = TF.to_tensor(img_)
            #         img_ = normalize(img_)
            #         res.append(img_)
            #     return torch.stack(res)
            
            # transform = transforms.Compose([
            #     transforms.Lambda(transform_batch)
            # ])
        elif type == 'baseaug':
            transform = v2.Compose([
                v2.Resize(size = (256, 256)),
                v2.RandomCrop(size = (self.size, self.size)),
                v2.RandomHorizontalFlip(),
                normalize
            ])
            
            # def transform_batch(x_batch):
            #     res = []
            #     for img in x_batch:
            #         img_pil = TF.to_pil_image(img)
            #         img_ = transforms.Resize(size = (256, 256))(img_pil)
            #         img_ = transforms.RandomCrop(size = (self.size, self.size))(img_)
            #         img_ = transforms.RandomHorizontalFlip()(img_)
            #         img_ = TF.to_tensor(img_)
            #         res.append(normalize(img_))
            #     return torch.stack(res)

            # transform = transforms.Compose([
            #     transforms.Lambda(transform_batch)
            # ])
        elif type == 'cutout':
            transform = timm_data.random_erasing.RandomErasing(
                probability=0.5, 
                min_area=0.02,
                max_area=1 / 3,
                min_aspect=0.3,
                max_aspect=None,
                mode="const",
                min_count=1,
                max_count=None,
                num_splits=0 
            )
        elif type == 'autoaug':
            transform = timm_data.auto_augment.auto_augment_transform(
                config_str="original-mstd0.5", hparams=h_params
            )
        elif type == 'augmix':
            transform = timm_data.auto_augment.augment_and_mix_transform(
                config_str="augmix-m5-w4-d2", hparams=h_params
            )
        elif type == 'randaug':
            transform = timm_data.auto_augment.rand_augment_transform(
                config_str="rand-m3-n2-mstd0.5", hparams=h_params
            )
        else:
            transform = nn.Identity()

        if type in ['randaug', 'augmix', 'autoaug']:
            transform = v2.Compose([
                resize_trans, transform, normalize
            ])
            # pre_trans = deepcopy(transform)
            # def transform_batch_aug(x_batch):
            #     res = []                
            #     for img in x_batch:
            #         img_pil = TF.to_pil_image(img)
            #         img_ = resize_trans(img_pil)
            #         img_ = pre_trans(img_)
            #         img_ = TF.to_tensor(img_)
            #         img_ = normalize(img_)
            #         res.append(img_)
            #     return torch.stack(res)
            
            # transform = transforms.Compose( [ transforms.Lambda(transform_batch_aug) ])
        
        return transform
    
    def forward(self, type, x, target=None, num_classes=10):
        transform = self._get_transform_(type, num_classes)
        
        x_, target_ = x, target
        if type in ['mixup', 'cutmix']:
            assert target is not None, "Invalid target for mixup or cutmix"
            x_, target_ = transform(x, target)
        else:
            x_ = transform(x)

        return x_, target_
        
    def __call__(self, type, x, target, num_classes=10):
        return self.forward(type, x, target, num_classes)


class ApproxAugment():
    def __init__(self, feature_avg=True, regularization=False):
        self.feature_avg = feature_avg
        self.regularization = regularization

        self._avg_features = None
        self._centered_features = None
        
    
    def regularization_2nd_order(self, output, reduce=True, device='cpu'):
        """Compute regularization term from output instead of from loss.
        Fast implementation by evaluating the Jacobian directly instead of relying on 2nd order differentiation.
        """
        p = F.softmax(output, dim=-1)
        # Using autograd.grad(output[:, i]) is slower since it creates new node in graph.
        # ones = torch.ones_like(output[:, 0])
        # W = torch.stack([autograd.grad(output[:, i], self._avg_features, grad_outputs=ones, create_graph=True)[0]
        #                  for i in range(10)], dim=1)
        eye = torch.eye(output.size(1), device=device)
        eye = eye[None, :].expand(output.size(0), -1, -1)
        W = torch.stack([autograd.grad(output, self._avg_features, grad_outputs=eye[:, i], create_graph=True)[0]
                         for i in range(10)], dim=1)
        # t = (W[:, None] * self._centered_features[:, :, None]).view(W.size(0), self._centered_features.size(1), W.size(1), -1).sum(dim=-1)
        t = (W.view(W.size(0), 1, W.size(1), -1) @ self._centered_features.view(*self._centered_features.shape[:2], -1, 1)).squeeze(-1)
        term_1 = (t**2 * p[:, None]).sum(dim=-1).mean(dim=-1)
        # term_1 = (t**2 @ p[:, :, None]).squeeze(2).mean(dim=-1)
        term_2 = ((t * p[:, None]).sum(dim=-1)**2).mean(dim=-1)
        # term_2 = ((t @ p[:, :, None]).squeeze(2)**2).mean(dim=-1)
        reg = (term_1 - term_2) / 2
        return reg.mean() if reduce else reg
    
    def regularization_2nd_order_linear(self, output, reduce=True):
        """Variance regularization (2nd order) term when the model is linear.
        Fastest implementations since it doesn't rely on pytorch's autograd.
        Equal to E[(W phi - W psi)^T (diag(p) - p p^T) (W phi - W psi)] / 2,
        where W is the weight matrix, phi is the feature, psi is the average
        feature, and p is the softmax probability.
        In this case @output is W phi + bias, but the bias will be subtracted away.
        """
        p = F.softmax(output, dim=-1)
        unreduced_output = self.output_from_features(self._centered_features + self._avg_features[:, None])
        reduced_output = self.output_from_features(self._avg_features)
        centered_output = unreduced_output - reduced_output[:, None]
        term_1 = (centered_output**2 * p[:, None]).sum(dim=-1).mean(dim=-1)
        term_2 = ((centered_output * p[:, None]).sum(dim=-1)**2).mean(dim=-1)
        reg = (term_1 - term_2) / 2
        return reg.mean() if reduce else reg

    def regularization_2nd_order_slow(self, output, reduce=True):
        """Compute regularization term from output, but uses pytorch's 2nd order differentiation.
        Slow implementation, only faster than @regularization_2nd_order_from_loss.
        """
        p = F.softmax(output, dim=-1)
        g, = autograd.grad(output, self._avg_features, grad_outputs=p, create_graph=True)
        term_1 = []
        for i in range(self._centered_features.size(1)):
            gg, = autograd.grad(g, p, grad_outputs=self._centered_features[:, i], create_graph=True)
            term_1.append((gg**2 * p).sum(dim=-1))
        term_1 = torch.stack(term_1, dim=-1).mean(dim=-1)
        term_2 = ((g[:, None] * self._centered_features).view(*self._centered_features.shape[:2], -1).sum(dim=-1)**2).mean(dim=-1)
        reg = (term_1 - term_2) / 2
        return reg.mean() if reduce else reg

    def regularization_2nd_order_from_loss(self, loss, reduce=True):
        """Variance regularization (2nd order) term.
        Computed from loss, using Pytorch's 2nd order differentiation.
        This is much slower but more likely to be correct. Used to check other implementations.
        """
        g, = autograd.grad(loss * self._avg_features.size(0), self._avg_features, create_graph=True)
        reg = []
        for i in range(self._centered_features.size(1)):
            gg, = autograd.grad(g, self._avg_features, grad_outputs=self._centered_features[:, i], create_graph=True)
            reg.append((gg * self._centered_features[:, i]).view(gg.size(0), -1).sum(dim=-1))
        reg = torch.stack(reg, dim=-1).mean(dim=-1) / 2
        return reg.mean() if reduce else reg    
    
    @staticmethod
    def combine_transformed_dimension(input):
        """Combine the minibatch and the transformation dimensions.
        Parameter:
            input: Tensor of shape B x T x ..., where B is the batch size and T is
                the number of transformations.
        Return:
            output: Same tensor, now of shape (B * T) x ....
        """
        return input.view(-1, *input.shape[2:])

    @staticmethod
    def split_transformed_dimension(input, n_transforms):
        """Split the minibatch and the transformation dimensions.
        Parameter:
            input: Tensor of shape (B * T) x ..., where B is the batch size and T is
                the number of transformations.
        Return:
            output: Same tensor, now of shape B x T x ....
        """
        return input.view(-1, n_transforms, *input.shape[1:])
    
    def forwards(self, x, model, device='cpu'):
        augmented = x.dim() > 4
        if augmented:
            n_transforms = x.size(1)
            x = self.combine_transformed_dimension(x)
        
        
        
        model.eval()
        with torch.no_grad():
            x = x.to(device)
            output = model(x)
            features = model.features(x)
            avg_features = features.mean(dim=0)
            centered_features = features - avg_features[None]
            return output, features, avg_features, centered_features
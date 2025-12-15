from os import path
import math
from copy import copy
from sklearn.metrics import roc_auc_score

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import mkdir


class CRModel(object):
    def __init__(self, num_classes, extractor):
        self.num_classes = num_classes
        self._extractor_ = extractor
        self._clf_w = None
        self._clf_b = None
        self._weight = None
    
    def save_model(self, out_path):
        mkdir(path.dirname(out_path))
        torch.save({'w': self._clf_w, 'b': self._clf_b, 'weight': self._weight}, out_path)

    def load_model(self, in_path):
        model = torch.load(in_path)
        self._clf_w = model['w']
        self._clf_b = model['b']
        self._weight = model['weight']
    
    def _tranform_y(self, y):
        y_ = copy(y)
        all_class = set(y)
        if len(all_class) == 2:
            if all_class == {-1, 1}:
                y_ = 2 * y_ - 1
        else:
            y_ = -torch.ones(y.size(0), int(y.max().item()) + 1).float()
            y_.scatter_(1, y.long().unsqueeze(1), 1)
        return y_

    def extract_data(self, loader, device='cuda'):
        train_x, train_y = list(), list()
        for inputs, labels in loader:
            inputs = inputs.to(device)
            x_ = self._extractor_(inputs)
            train_x.append(x_.detach().cpu())
            train_y.append(labels.detach().cpu())
        
        train_x = torch.cat(train_x, dim=0).squeeze()
        train_y = torch.cat(train_y, dim=0).squeeze()
        
        return train_x, train_y
    
    def train(self, loader, lam_reg,  train_mode, sample_rate=1.0, std=1.0, num_steps=100, 
                tolerance=1e-5, train_seperate=False, device='cuda'):
        train_x = list()
        for inputs, _ in loader:
            inputs = inputs.to(device)
            x_ = self._extractor_(inputs)
            train_x.append(x_.cpu())        
        train_x = torch.cat(train_x, dim=0).squeeze()
        train_y = torch.from_numpy(loader.dataset.targets)
        train_y_onehot = self._tranform_y(train_y)

        if sample_rate < 1.0:
            idx = torch.rand(train_y.size()).lt(sample_rate).float()
            self._clf_w = (idx + train_y.ge(0).float()).gt(0).float()
            self._clf_w = self._clf_w / self._clf_w.sum(0).unsqueeze(0)
            self._clf_w = self._clf_w.to(device)
        
        train_x = train_x.float().to(device)
        train_y = train_y.float().to(device)
        train_y_onehot = train_y_onehot.float().to(device)
        
        if train_mode == 'ovr':
            self.__train_ovr__(train_x, train_y_onehot, lam_reg, num_steps, train_seperate, device)
        else:
            self._clf_b = std * torch.randn(train_x.size(1)).float()
            self._clf_w = self.__lr_optimize__(train_x, train_y, lam_reg, self._clf_b, num_steps, tolerance, device)
        
    def predict_proba(self, loader, device='cuda'):
        self._extractor_.eval()
        self._extractor_.to(device)
        self._clf_w = self._clf_w.to(device)
        self._clf_b = self._clf_b.to(device)
        
        logit_pred, y_true = list(), list()        
        for inputs, labels in loader:
            inputs = inputs.to(device)
            test_x = self._extractor_(inputs).squeeze()
            pred = test_x.mm(self._clf_w)
            pred = F.sigmoid(pred)
            logit_pred += list(pred.cpu())
            y_true += list(labels.cpu())

        logit_pred = torch.stack(logit_pred, dim=0)
        y_true = torch.stack(y_true, dim=0)
        
        return logit_pred, y_true

    def test_acc(self, test_loader, device='cuda'):
        logits, y_true = self.predict_proba(test_loader, device)
        preds = torch.max(logits, dim=1)
        acc = preds.eq(y_true).float().mean()
        
        return acc.item()

    def test_auc(self, test_loader, device='cuda'):
        logits, y_true = self.predict_proba(test_loader, device)
        preds = torch.max(logits, dim=1)
        score = roc_auc_score(y_true, preds)
        return score
    
    def __batch_multiply(self, A, B, batch_size=256, device='cuda'):
        if A.is_cuda:
            if len(B.size()) == 1:
                return A.mv(B)
            else:
                return A.mm(B)
        else:
            out = []
            num_batch = int(math.ceil(A.size(0) / float(batch_size)))
            with torch.no_grad():
                for i in range(num_batch):
                    lower = i * batch_size
                    upper = min((i+1) * batch_size, A.size(0))
                    A_sub = A[lower:upper]
                    A_sub = A_sub.to(device)
                    if len(B.size()) == 1:
                        out.append(A_sub.mv(B).cpu())
                    else:
                        out.append(A_sub.mm(B).cpu())
            
            res = torch.cat(out, dim=0)
            res = res.to(device)
            return res

    def __lr_loss(self, w, X, y, lam):
        return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2

    def __lr_optimize__(self, X, y, lam, b=None, num_steps=100, tol=1e-10, device='cuda'):
        w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)
        def closure():
            loss = self.__lr_loss(w, X, y, lam)
            if b is not None:
                loss += b.dot(w) / X.size(0)
            return loss
        optimizer = optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-20)
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = self.__lr_loss(w, X, y, lam)
            if b is not None:
                loss += b.dot(w) / X.size(0)
            loss.backward()
            # self.logger.info('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
            optimizer.step(closure)
        return w.data

    def __ovr_lr_loss(self, w, X, y, lam, weight=None):
        z = self.__batch_multiply(X, w).mul_(y)
        loss = 0.0
        if weight is None:
            loss += -F.logsigmoid(z).mean() + lam * w.pow(2).sum() / 2
        else:
            loss += -F.logsigmoid(z).mul_(weight).sum() + lam * w.pow(2).sum() / 2
        return loss
    
    def __ovr_lr_oprimizer(self, X, y, lam, weight=None, b=None, num_steps=100, tol=1e-10, device='cuda'):
        T = torch.zeros(X.size(1), y.size(1)).float().to(device)
        w = torch.autograd.Variable(T, requires_grad=True)
        def closure():
            ## loss of the ovr LR
            loss = self.__ovr_lr_loss(w, X, y, lam, weight)
            if b is None:
                loss += (b * w).sum() / X.size(0)
            return loss
            
        optimizer = optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-10)
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = self.__ovr_lr_loss(w, X, y, lam, weight)
            if b is not None:
                if weight is None:
                    loss += (b * w).sum() / X.size(0)
                else:
                    loss += ((b * w).sum(0) * weight.max(0)[0]).sum()
            loss.backward()
            # self.logger.info('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
            optimizer.step(closure)
        return w.data
    
    def __train_ovr__(self, train_x, train_y_onehot, lam, std=1.0, num_steps=100, train_seperate=False, device='cuda'):
        num_classes = train_y_onehot.size(1)
        self._clf_b = std * torch.randn(train_x.size(1), num_classes).float().to(device)
        if train_seperate:
            ## train K binary LR classifiers seperately
            self._clf_w = torch.zeros(self._clf_b.size()).to(device)
            for i in range(num_classes):
                # self.logger.info('Training classifier for class %d' % i)
                if self._weight is None:
                    self._clf_w[:, i] = self.__lr_optimize__(train_x, train_y_onehot[:, i], lam, 
                                                           b=self._clf_b[:, i], num_steps=num_steps, device=device)
                else:
                    idx = self._weight[:, i].gt(0)
                    self._clf_w[:, i], self._clf_b[i] = self.__lr_optimize__(train_x[idx], train_y_onehot[idx, i], lam, 
                                                                        b=self._clf_b[:, i], num_steps=num_steps, device=device)
        else:
            ## train K binary LR classifiers jointly
            self._clf_w = self.__ovr_lr_oprimizer(train_x, train_y_onehot, lam, weight=self._weight, 
                                                  b=self._clf_b, num_steps=num_steps, device=device)


class CertifiedRemoval(object):
    LIPSCHITZ_CONST = 1/4.0
    def __init__(self):
        pass

    def __spectral_norm__(self, A, num_iters=20, device='cuda'):
        x = torch.randn(A.size(0)).float().to(device)
        norm = 1
        for i in range(num_iters):
            x = A.mv(x)
            norm = x.norm()
            x /= norm
        return math.sqrt(norm)
    
    @staticmethod
    def lr_eval(w, X, y):
        return X.mv(w).sign().eq(y).float().mean()

    @staticmethod
    def lr_grad(w, X, y, lam):
        z = torch.sigmoid(y * X.mv(w))
        return X.t().mv((z-1) * y) + lam * X.size(0) * w
    
    def __lr_hessian_inv__(self, w, X, y, lam, batch_size=256, device='cuda'):
        z = torch.sigmoid(X.mv(w).mul_(y))
        D = z * (1 - z)
        H = None
        num_batch = int(math.ceil(X.size(0) / batch_size))
        for i in range(num_batch):
            lower = i * batch_size
            upper = min((i+1) * batch_size, X.size(0))
            X_i = X[lower:upper]
            if H is None:
                H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
            else:
                H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)).inverse()


    def remove(self, model:CRModel, train_loader, forget_loader, lam_reg, train_mode, 
               batch_size=256, device='cuda'):
        train_x, train_y = model.extract_data(train_loader, device)
        forget_x, forget_y = model.extract_data(forget_loader, device)
        num_forget = len(forget_x)
        
        train_one_hot = model._tranform_y(train_y)
        if train_mode == 'ovr':
            train_y = train_one_hot
        
        train_x, train_y = train_x.to(device), train_y.to(device)
        forget_x, forget_y = forget_x.to(device), forget_y.to(device)
        
        grad_norm_approx = torch.zeros(num_forget).float().to(device)
        w_approx = model._clf_w.clone()
        
        if model._weight is None:
            K = train_x.t().mm(train_x)
        else:
            Ks = list()
            for i in range(train_one_hot.size(0)):
                x_sub = train_x.cpu()[model._weight[:, i].gt(0).cpu()]
                Ks.append(x_sub.t().mm(x_sub).to(device))

        for i in range(num_forget):
            x_rm, y_rm = forget_x[i], forget_y[i]
            if train_mode == 'ovr':
                # remove from all one-vs-rest models (classifiers)
                for k in range(train_one_hot.size(1)):
                    if model._weight is None or model._weight[i, k] > 0:
                        train_x = train_x
                        train_y_label = train_y[:, k]
                        if model._weight is not None:
                            idx = model._weight[:, k].gt(0)
                            train_x = train_x[idx]
                            train_y_label = train_y_label[idx]
                            
                        H_inv = self.__lr_hessian_inv__(w_approx[:, k], train_x, train_y_label, lam_reg, batch_size=batch_size, device=device)
                        grad_i = self.lr_grad(w_approx[:, k], x_rm.unsqueeze(0), y_rm[k].unsqueeze(0), lam_reg)
                        
                        if model._weight is None:
                            K = K - torch.ger(x_rm, x_rm)
                            spec_norm = self.__spectral_norm__(K, device)
                        else:
                            Ks[k] = Ks[k] - torch.ger(x_rm, x_rm)
                            spec_norm = self.__spectral_norm__(Ks[k], device)
                        
                        Delta = H_inv.mv(grad_i)
                        Delta_p = train_x.mv(Delta)
                        w_approx[:, k] += Delta
                        grad_norm_approx[i] += (Delta.norm() + spec_norm * Delta_p.norm() * self.LIPSCHITZ_CONST).cpu()
            else:
                H_inv = self.__lr_hessian_inv__(w_approx, train_x, train_y, lam_reg, batch_size=batch_size, device=device)
                grad_i = self.lr_grad(w_approx, x_rm.unsquezee(0), train_y.unsqueeze(0), lam_reg)
                K = K - torch.ger(x_rm, x_rm)
                spec_norm = self.__spectral_norm__(K, device)
                Delta = H_inv.mv(grad_i)
                Delta_p = train_x.mv(Delta)
                w_approx += Delta
                grad_norm_approx[i] += (Delta.norm() + spec_norm * Delta_p.norm() * self.LIPSCHITZ_CONST).cpu()

        model._clf_w = w_approx        
        return model

    
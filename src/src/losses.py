import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.modules.loss import _WeightedLoss
from geomloss import SamplesLoss

from utils import union_of_vectors
from learner import _disable_tracking_bn_stats, normalize_l2, optimize_linear


######################### triplet_loss #########################
class TripletLoss(nn.Module):
    def __init__(self, format='diff', class_wise=False, generalize=False, device='cpu'):
        super(TripletLoss, self).__init__()
        self.class_wise = class_wise
        self.generalize = generalize
        self.device = device
        if format == 'diff':
            self.operator = lambda x, y: torch.abs(x - y)
        elif format == 'add':
            self.operator = lambda x, y: x + y
        elif format =='ratio':
            self.operator = lambda x, y: x / (y + 1e-6)
        else:
            raise NotImplementedError
    
    def forward(self, retain_embeddings, retain_labels, valid_embeddings, valid_labels, forget_embeddings, forget_labels):
        NotImplementedError
    
    def __call__(self, retain_embeddings, retain_labels, valid_embeddings, valid_labels, forget_embeddings, forget_labels):
        if self.class_wise:
            assert (not (retain_labels is None)) and (not (valid_labels is None)) and (not (forget_labels is None)), \
                    "Class-wise triplet loss requires all labels to be provided."
        return self.forward(retain_embeddings, retain_labels, valid_embeddings, valid_labels, forget_embeddings, forget_labels)

class ContrastiveLoss(TripletLoss):
    def __init__(self, temperature=0.1, margin=0.1, class_wise=True, generalize=False, device='cpu', alpha=1.0):
        super(ContrastiveLoss, self).__init__('add', class_wise, generalize, device)
        self.temperature = temperature
        self.margin = margin
    
    def __str__(self):
        return "TriContLoss"
    
    def forward(self, retain_embeddings, retain_labels, valid_embeddings, valid_labels, forget_embeddings, forget_labels):
        device = self.device
        retain_embs = F.normalize(retain_embeddings, p=2, dim=1).to(device)
        forget_embs = F.normalize(forget_embeddings, p=2, dim=1).to(device)
        
        # if self.class_wise:
        #     # Compute class-conditioned logits
        #     num_classes = torch.max(forget_labels).item() + 1
        #     class_mask = torch.eye(num_classes).to(device)
        #     class_logits = torch.matmul(forget_norm * class_mask[forget_labels], 
        #                                 valid_norm.transpose(0, 1) @ class_mask[valid_labels]) / self.temperature

        #     # Compute class-conditioned negative logits
        #     negative_class_logits = torch.matmul(forget_norm * class_mask[forget_labels], 
        #                                             retain_norm.transpose(0, 1) @ class_mask[retain_labels]) / self.temperature
        #     # Combine positive and negative logits
        #     logits = torch.cat([class_logits, negative_class_logits], dim=1)
        # else:
        #     # Compute normal logits
        #     logits = torch.matmul(forget_norm, valid_norm.transpose(0, 1)) / self.temperature
        #     # Compute negative logits
        #     negative_logits = torch.matmul(forget_norm, retain_norm.transpose(0, 1)) / self.temperature
        #     # Combine positive and negative logits
        #     logits = torch.cat([logits, negative_logits], dim=1)

        # # Compute labels
        # labels = torch.zeros(logits.size(0), device=device, dtype=torch.long)

        # # Compute contrastive loss
        # loss_ = F.cross_entropy(logits, labels)
        # return loss_
        
        loss_ = 0.0
        random_forget = len(np.intersect1d(torch.unique(forget_labels), torch.unique(retain_labels))) > 0
        if self.class_wise:
            ## prossess the forget samples
            fv_mask = torch.eq(forget_labels.unsqueeze(1), valid_labels.unsqueeze(0)).to(device)
            f_v_dist = torch.cdist(forget_embs, valid_embs, p=2).to(device)
            
            f_v_dist = fv_mask * f_v_dist + (~fv_mask) * self.margin
            loss_ += f_v_dist.mean()
            
            if random_forget:  ## random sample forget
                f_r_mask = torch.eq(forget_labels.unsqueeze(1), retain_labels.unsqueeze(0)).to(device)
                f_r_dist = torch.cdist(forget_embs, retain_embs, p=2).to(device)
                
                f_r_dist = f_r_mask * f_r_dist + (~f_r_mask) * 0
                loss_ += F.relu(self.margin - f_r_dist).mean()
            else:       ## class forget
                ## process the non-forget samples
                if self.generalize:  ## add the generalization term for those samples
                    vf_mk_ = torch.eq(valid_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1).to(device)
                    rf_mk_ = torch.eq(retain_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1).to(device)
                    
                    mask = torch.eq(valid_labels[~vf_mk_].unsqueeze(1), retain_labels[~rf_mk_].unsqueeze(0)).to(device)
                    dist = torch.cdist(valid_embs[~vf_mk_], retain_embs[~rf_mk_], p=2).to(device)
                    vr_dist = dist * mask + (~mask) * self.margin
                    loss_ += vr_dist.mean()
        else:
            if random_forget:  ## random sample forget
                ## prossess the forget samples
                vf_mask = torch.eq(valid_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1).to(device)
                vf_dist = torch.cdist(forget_embs, valid_embs[vf_mask], p=2).to(device)
                
                rf_mask = torch.eq(retain_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1).to(device)
                rf_dist = torch.cdist(forget_embs, retain_embs[rf_mask], p=2).to(device)
                
                loss_ += vf_dist.mean() + (~vf_mask).sum() * self.margin
                loss_ += F.relu(self.margin - rf_dist).mean()
                
                if self.generalize and torch.any(~rf_mask) and torch.any(~vf_mask):
                    rv_dist = torch.cdist(retain_embs[~rf_mask], valid_embs[~vf_mask], p=2).to(device)
                    loss_ += rv_dist.mean()
            else:      ## class forget
                vf_mask = torch.eq(valid_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1).to(device)
                vf_dist = torch.cdist(valid_embs[vf_mask], forget_embs, p=2).to(device)
                loss_ += vf_dist.mean()
                if self.generalize and torch.any(~vf_mask):
                    rv_dist = torch.cdist(retain_embs, valid_embs[~vf_mask], p=2).to(device)
                    loss_ += rv_dist.mean()
        
        return loss_

class TriSoftNearestNeighborsLoss(TripletLoss):
    def __init__(self, temperature=0.1, generalize=False, device='cpu'):
        super(TriSoftNearestNeighborsLoss, self).__init__('ratio', class_wise=True, generalize=generalize, device=device)
        self.temperature = temperature
    
    def __str__(self):
        return "TriSNNLoss"
    
    def _snn_(self, X1, X2, Y1, Y2):
        dist = torch.cdist(X1, X2, p=2)   ## l2 distance
        wts = F.softmax(-dist / self.temperature, dim=1)
        mask_ = torch.eq(Y1.unsqueeze(1), Y2.unsqueeze(0)).float().to(self.device)
        probs = torch.sum(wts * mask_, dim=1) / wts.sum(dim=1) - wts.diag()
        return - torch.log(probs).mean()

    def forward(self, retain_embeddings, retain_labels, valid_embeddings, valid_labels, forget_embeddings, forget_labels):
        device = self.device
        # Normalize embeddings to use cosine similarity
        retain_embs = F.normalize(retain_embeddings, p=2, dim=1).to(device)
        valid_embs = F.normalize(valid_embeddings, p=2, dim=1).to(device)
        forget_embs = F.normalize(forget_embeddings, p=2, dim=1).to(device)
        
        top_k = 5
        loss_ = 0.0
        f_cls = torch.unique(forget_labels)
        v_cls = torch.unique(valid_labels)
        random_forget = len(np.intersect1d(torch.unique(forget_labels), torch.unique(retain_labels))) > 0
        if random_forget:   ## random sample forget
            ##TODO: version v1
            vf_mask = torch.eq(valid_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1)
            rf_mask = torch.eq(retain_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1)
            # fv_ = self._snn_(forget_embs, valid_embs[vf_mask], forget_labels, valid_labels[vf_mask])
            # fr_ = self._snn_(forget_embs, retain_embs[rf_mask], forget_labels, retain_labels[rf_mask])
            # loss_ += self.operator(fv_, fr_)
            
            ## version v2
            for c_i in f_cls:
                f_mask = forget_labels == c_i
                v_mask = valid_labels == c_i
                r_mask = retain_labels == c_i
                # dist_1 = torch.topk(torch.cdist(forget_embs[f_mask], valid_embs[v_mask], p=2), k=top_k, largest=False, dim=1)[0]
                # dist_2 = torch.topk(torch.cdist(forget_embs[f_mask], retain_embs[r_mask], p=2), k=top_k, largest=False, dim=1)[0]
                dist_1 = torch.cdist(forget_embs[f_mask], valid_embs[v_mask], p=2)
                dist_2 = torch.cdist(forget_embs[f_mask], retain_embs[r_mask], p=2)
                w_1 = torch.sum(torch.exp(-dist_1 / self.temperature), dim=1)
                w_2 = torch.sum(torch.exp(-dist_2 / self.temperature), dim=1)
                loss_ += - torch.log(w_1 / (w_1 + w_2)).mean()
            
            if self.generalize and torch.any(~rf_mask) and torch.any(~vf_mask):
                rv_ = self._snn_(retain_embs[~rf_mask], valid_embs[~vf_mask], retain_labels[~rf_mask], valid_labels[~vf_mask])
                loss_ += rv_
            
        else:                   ## class forget
            vf_mask = torch.eq(valid_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1).to(device)
            fv_ = self._snn_(valid_embs[vf_mask], forget_embs, valid_labels[vf_mask], forget_labels)
            loss_ += fv_
            if self.generalize and torch.any(~vf_mask):
                rv_ = self._snn_(retain_embs, valid_embs[~vf_mask], retain_labels, valid_labels[~vf_mask])
                loss_ += rv_
        
        return loss_

class TriMMDLoss(TripletLoss):
    def __init__(self, format='diff', kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, class_wise=True, generalize=False, device='cpu'):
        assert kernel_type in ['linear', 'rbf'], "Unknown kernel type."
        super(TriMMDLoss, self).__init__(format, class_wise, generalize, device)
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
            
    def __str__(self):
        return "TriMMDLoss"
    
    @staticmethod
    def linear_mmd2(S, T):
        loss = 0.0
        delta = S.float().mean(0) - T.float().mean(0)
        loss = delta.dot(delta.T)
        return loss
    
    @staticmethod
    def _gaussian_kernel_(S, T, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        Computes the Gaussian kernel between two tensors.

        Args:
            S (torch.Tensor): First tensor.
            T (torch.Tensor): Second tensor.
            kernel_mul (float, optional): Multiplier for the squared distance. Default is 2.0.
            kernel_num (int, optional): Number of Gaussian kernels. Default is 5.
            fix_sigma (float or None, optional): Fixed value for the standard deviation of the Gaussian kernels.
                If None, the standard deviation is computed based on the data. Default is None.

        Returns:
            torch.Tensor: Gaussian kernel matrix between x and y.
        """
        n_samples = int(T.size(0)) + int(S.size(0))
        total = torch.cat([S, T])
        
        tot_d0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        tot_d1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((tot_d0 - tot_d1) ** 2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) + 1e-6 for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
        return sum(kernel_val)
    
    def _mmd_(self, S, T):
        if len(S) <= 0 or len(T) <= 0:
            return 0
        
        if self.kernel_type == 'linear':
            return self.linear_mmd2(S, T)
        elif self.kernel_type == 'rbf':
            kernels = self._gaussian_kernel_(S, T, self.kernel_mul, self.kernel_num, self.fix_sigma)
            bs = int(S.size()[0])
            XX = kernels[:bs, :bs]
            YY = kernels[bs:, bs:]
            XY = kernels[:bs, bs:]
            YX = kernels[bs:, :bs]
            loss = XX.mean() + YY.mean() - XY.mean() - YX.mean()
            return loss
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def forward(self, retain_embeddings, retain_labels, valid_embeddings, valid_labels, forget_embeddings, forget_labels):
        ## maximize the mmd loss between the forget and retain set while minimize the mmd loss between the forget and valid set
        device = self.device
        retain_embs = F.normalize(retain_embeddings, p=2, dim=1).to(device)
        valid_embs = F.normalize(valid_embeddings, p=2, dim=1).to(device)
        forget_embs = F.normalize(forget_embeddings, p=2, dim=1).to(device)
        
        loss_ = 0.0
        f_cls = set(torch.unique(forget_labels))
        r_cls = set(torch.unique(retain_labels))
        random_forget = len(np.intersect1d(f_cls, r_cls)) > 0
        if self.class_wise:
            # Class-wise MMD computation
            if random_forget:
                all_cls = union_of_vectors(retain_labels, valid_labels, forget_labels)
                for c_i in all_cls:
                    v_mask = valid_labels == c_i
                    r_mask = retain_labels == c_i
                    if c_i in f_cls:
                        f_mask = forget_labels == c_i
                        
                        dist_pos = 0
                        dist_neg = self._mmd_(forget_embs(f_mask), retain_embs[r_mask])
                        if torch.any(v_mask):
                            dist_pos = self._mmd_(forget_embs(f_mask), valid_embs[v_mask])
                        loss_ += self.operator(dist_pos, dist_neg)
                    else:
                        ## minize the distance between valiadation and retain
                        if self.generalize and torch.any(v_mask) and torch.any(r_mask):
                            loss_ += self._mmd_(valid_embs[v_mask], retain_embs[r_mask])
            else:  ## class forget
                for c_i in f_cls:   ## forget classes
                    vf_mask = valid_labels == c_i
                    loss_ += self._mmd_(forget_embs, valid_embs[vf_mask])
                if self.generalize:   ## retained classes
                    for c_i in r_cls:
                        v_mask = valid_labels == c_i
                        r_mask = retain_labels == c_i
                        if torch.any(v_mask) and torch.any(r_mask):
                            loss_ += self._mmd_(valid_embs[v_mask], retain_embs[r_mask])
        else:
            # Global MMD computation
            vf_mask = torch.eq(valid_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1).to(device)
            if random_forget:
                rf_mask = torch.eq(retain_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1).to(device)
                dist_pos = self._mmd_(forget_embs, valid_embs[vf_mask])
                dist_neg = self._mmd_(forget_embs, retain_embs[rf_mask])
                loss_ = self.operator(dist_pos, dist_neg)
                
                if self.generalize and torch.any(~rf_mask) and torch.any(~vf_mask):
                    rv_ = self._mmd_(retain_embs[~rf_mask], valid_embs[~vf_mask])
                    loss_ += rv_
            else:  ## class forget
                loss_ = self._mmd_(forget_embs, valid_embs[vf_mask]) 
                
                if self.generalize and torch.any(~vf_mask):
                    rv_ = self._mmd_(retain_embs, valid_embs[~vf_mask])
                    loss_ += rv_

        return loss_

class TriWassersteinLoss(TripletLoss):
    def __init__(self, format='diff', class_wise=True, generalize=False, device='cpu'):
        super(TriWassersteinLoss, self).__init__(format, class_wise, generalize, device)
    
    def __str__(self):
        return "TriWassersteinLoss"
    
    def forward(self, retain_embeddings, retain_labels, valid_embeddings, valid_labels, forget_embeddings, forget_labels):
        if self.class_wise:
            f_cls = set(torch.unique(forget_labels).numpy())
            r_cls = set(torch.unique(retain_labels).numpy())
            v_cls = set(torch.unique(valid_labels).numpy())
            assert f_cls.issubset(r_cls) and f_cls.issubset(v_cls), "Forget class(es) must be a subset of retain and valid classes."
            
        device = self.device
        retain_embs = F.normalize(retain_embeddings, p=2, dim=1).to(device)
        valid_embs = F.normalize(valid_embeddings, p=2, dim=1).to(device)
        forget_embs = F.normalize(forget_embeddings, p=2, dim=1).to(device)

        # Sinkhorn loss setup with default parameters
        loss_ = 0.0
        f_cls = torch.unique(forget_labels)
        random_forget = len(np.intersect1d(torch.unique(forget_labels), torch.unique(retain_labels))) > 0
        sinkhorn_dist = SamplesLoss(loss="sinkhorn", p=2, blur=0.1, scaling=0.7, backend="online")
        if self.class_wise:
            if random_forget:
                all_cls = union_of_vectors(retain_labels, valid_labels, forget_labels)
                for c_i in all_cls:
                    v_mask = valid_labels == c_i
                    r_mask = retain_labels == c_i
                    
                    if c_i in f_cls:
                        d_pos = 0.0
                        f_mask = forget_labels == c_i
                        d_neg = sinkhorn_dist(forget_embs[f_mask], retain_embs[r_mask]) / (f_mask.sum() * r_mask.sum())
                        if torch.any(v_mask):
                            d_pos += sinkhorn_dist(forget_embs[f_mask], valid_embs[v_mask]) / (f_mask.sum() * v_mask.sum())
                        loss_ += self.operator(d_pos, d_neg)
                    else:
                        if self.generalize and torch.any(v_mask) and torch.any(r_mask):
                            loss_ += sinkhorn_dist(retain_embs[r_mask], valid_embs[v_mask]) / (r_mask.sum() * v_mask.sum())
            else:
                for c_i in f_cls:
                    vf_mask = valid_labels == c_i
                    if torch.any(vf_mask):
                        loss_ += sinkhorn_dist(forget_embs, valid_embs[vf_mask]) / (forget_embs.size(0) * v_mask.sum())
                if self.generalize:
                    for c_i in r_cls:
                        v_mask = valid_labels == c_i
                        r_mask = retain_labels == c_i
                        if torch.any(r_mask) and torch.any(v_mask):
                            loss_ += sinkhorn_dist(retain_embs[r_mask], valid_embs[v_mask]) / (r_mask.sum() * v_mask.sum())
        else:
            # Compute global OT loss without class conditioning
            # minimize the distance between the forget and the valid while maximizing the distance between the forget and the retain
            vf_mask = torch.eq(valid_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1).to(device)
            if random_forget:
                rf_mask = torch.eq(retain_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1).to(device)
                d_pos = sinkhorn_dist(forget_embs, valid_embs[vf_mask]) / (forget_embs.size(0) * vf_mask.sum())
                d_neg = sinkhorn_dist(forget_embs, retain_embs[rf_mask]) / (forget_embs.size(0) * rf_mask.sum())
                loss_ = self.operator(d_pos, d_neg)

                if self.generalize and torch.any(~rf_mask) and torch.any(~vf_mask):
                    rv_ = sinkhorn_dist(retain_embs[~rf_mask], valid_embs[~vf_mask])
                    rv_ /= (~rf_mask).sum() * (~vf_mask).sum()
                    loss_ += rv_
                
            else:      # class forget
                loss_ = sinkhorn_dist(forget_embs, valid_embs[vf_mask]) 
                if self.generalize and torch.any(~vf_mask):
                    rv_ = sinkhorn_dist(retain_embs, valid_embs[~vf_mask])
                    rv_ /= (~vf_mask).sum() * retain_embs.size(0)
                    loss_ += rv_
        
        return loss_

class TriEntropyLoss(TripletLoss):
    def __init__(self, protocal='dist', class_wise=True, generalize=False, device='cpu'):
        format='diff'
        super(TriEntropyLoss, self).__init__(format, class_wise, generalize, device)
        ## here protocla also indicates that that using the distance of entropy distribution rather than the entropy mean
        self.protocal = protocal
        
    def __str__(self):
        return "TriEntropyLoss"
    
    @staticmethod
    def gaussian_kde(data, grid, bandwidth=1.0):
        """Compute the Gaussian KDE for a set of data points on a specified grid."""
        data = data.unsqueeze(-1)  # Shape (n, 1)
        grid = grid.unsqueeze(0)  # Shape (1, m)
        diff = data - grid  # Shape (n, m)
        weights = torch.exp(-0.5 * (diff / bandwidth) ** 2)
        kde = weights.mean(dim=0)
        return kde / kde.sum()  # Normalize to make it a probability distribution
    
    @staticmethod
    def sinkhorn_distance(p, q, cost_matrix, reg, num_iter=100):
        """Compute the Sinkhorn distance between two distributions."""
        K = torch.exp(-cost_matrix / reg)
        u = torch.ones_like(p)
        v = torch.ones_like(q)
        
        for _ in range(num_iter):
            u = p / (K @ v)
            v = q / (K.t() @ u)
        
        transport_plan = torch.diag(u) @ K @ torch.diag(v)
        return torch.sum(transport_plan * cost_matrix)

    def _calc_gauskde_sinkhorn_(self, logent_1, logent_2, bandwidth = 0.1, reg=0.01):
        min_val = min(logent_1.min().item(), logent_2.min().item())
        max_val = max(logent_1.max().item(), logent_2.max().item())
        
        grid = torch.linspace(min_val, max_val, 100).to(logent_1.device)

        # Compute KDEs
        kde1 = self.gaussian_kde(logent_1, grid, bandwidth=bandwidth)
        kde2 = self.gaussian_kde(logent_2, grid, bandwidth=bandwidth)

        # Define the cost matrix (L1 distance between grid points)
        cost_matrix = torch.abs(grid.unsqueeze(1) - grid.unsqueeze(0))
        sinkhorn_dist = self.sinkhorn_distance(kde1, kde2, cost_matrix, reg)
        return sinkhorn_dist

    def forward(self, retain_embeddings, retain_labels, valid_embeddings, valid_labels, forget_embeddings, forget_labels):
        device = self.device
        retain_norm_embs = F.normalize(retain_embeddings, p=2, dim=1).to(device)
        valid_norm_embs = F.normalize(valid_embeddings, p=2, dim=1).to(device)
        forget_norm_embs = F.normalize(forget_embeddings, p=2, dim=1).to(device)
        
        # Compute entropy loss
        forget_probs = F.softmax(forget_norm_embs, dim=1)
        valid_probs = F.softmax(valid_norm_embs, dim=1)
        retain_probs = F.softmax(retain_norm_embs, dim=1)
        
        ## log entropy
        forget_logent = torch.log(-torch.sum(forget_probs * torch.log(forget_probs + 1e-10), dim=-1) + 1e-10)
        valid_logent = torch.log(-torch.sum(valid_probs * torch.log(valid_probs + 1e-10), dim=-1) + 1e-10)
        retain_logent = torch.log(-torch.sum(retain_probs * torch.log(retain_probs + 1e-10), dim=-1) + 1e-10)
        
        ## INSTRUCTION:  for the log entropy of the prediction
        ##      1.   [class forget case]  maximize the forget entropy and minimize the ot distance between retian and valid entropy distribution
        ##      2. [random sample forget] maximize the ot distance between forget and valid, and between forget and retain entropy distribution
        ## choose use what format of the info.:  basic descriptive statistic mean or vetor difference (distribution)
        loss_ = 0.0
        # return  forget_logent.mean()
        # loss_ += 1.0 * forget_logent.mean()  ### maximize the log entropy of forget samples
        f_cls = torch.unique(forget_labels)
        all_cls = union_of_vectors(retain_labels, valid_labels, forget_labels)
        random_forget = len(np.intersect1d(f_cls, torch.unique(retain_labels))) > 0
        if self.protocal == 'mean':
            if random_forget:
                if self.class_wise:
                    for c_i in all_cls:
                        v_mask = valid_labels == c_i
                        r_mask = retain_labels == c_i
                        if c_i in f_cls:
                            f_mask = forget_labels == c_i
                            f_e_i = forget_logent[f_mask].mean()
                            r_e_i = retain_logent[r_mask].mean()
                            d_neg = self.operator(f_e_i, r_e_i)
                            if torch.any(v_mask):
                                v_e_i = retain_logent[v_mask].mean()
                                d_neg += self.operator(f_e_i, v_e_i)
                            loss_ += -1.0 * d_neg
                        else:
                            if self.generalize and torch.any(v_mask) and torch.any(r_mask):
                                r_e_i = retain_logent[r_mask].mean()
                                v_e_i = valid_logent[v_mask].mean()
                                loss_ += (r_e_i - v_e_i)**2
                else:
                    rf_mask = torch.eq(retain_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1)
                    vf_mask = torch.eq(valid_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1)
                    loss_ = -1.0 * (forget_logent.mean() - valid_logent[vf_mask].mean())**2
                    loss_ += -1.0 * (forget_logent.mean() - retain_logent[rf_mask].mean())**2
                    
                    if self.generalize and torch.any(~rf_mask) and torch.any(~vf_mask):
                        loss_ += (valid_logent[~vf_mask].mean() - retain_logent[~rf_mask].mean())**2
            else:  # class forget
                if self.class_wise:
                    for c_i in f_cls:
                        f_mask = forget_labels == c_i
                        v_mask = valid_labels == c_i
                        f_e_i = forget_logent[f_mask].mean()
                        v_e_i = valid_logent[v_mask].mean()
                        loss_ += -1.0 * self.operator(f_e_i, v_e_i)
                    
                    if self.generalize:
                        r_cls = torch.unique(retain_labels)
                        for c_i in r_cls:
                            r_mask = retain_labels == c_i
                            v_mask = valid_labels == c_i
                            if torch.any(r_mask) and torch.any(v_mask):
                                r_e_i = retain_logent[r_mask].mean()
                                v_e_i = valid_logent[v_mask].mean()
                                loss_ += (r_e_i - v_e_i)**2
                else:
                    vf_mask = torch.eq(valid_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1)
                    loss_ += -1.0 * (forget_logent.mean() - valid_logent[vf_mask].mean())**2
                    
                    if self.generalize and torch.any(~vf_mask):
                        loss_ += (valid_logent[~vf_mask].mean() - retain_logent.mean())**2
        elif self.protocal == 'dist':
            if random_forget:
                if self.class_wise:
                    for c_i in all_cls:
                        v_mask = valid_labels == c_i
                        r_mask = retain_labels == c_i
                        if c_i in f_cls:
                            f_mask = forget_labels == c_i
                            d_neg = self._calc_gauskde_sinkhorn_(forget_logent[f_mask], retain_logent[r_mask])
                            if torch.any(v_mask):
                                d_neg += self._calc_gauskde_sinkhorn_(forget_logent[f_mask], valid_logent[v_mask])
                            loss_ += -1.0 * d_neg
                        else:
                            if self.generalize_loss and torch.any(v_mask) and torch.any(r_mask):
                                loss_ += self._calc_gauskde_sinkhorn_(retain_logent[r_mask], valid_logent[v_mask])
                else:
                    rf_mask = torch.eq(retain_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1).to(device)
                    vf_mask = torch.eq(valid_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1).to(device)
                    loss_ = -1.0 * self._calc_gauskde_sinkhorn_(forget_logent, valid_logent[vf_mask])
                    loss_ += -1.0 * self._calc_gauskde_sinkhorn_(forget_logent, retain_logent[rf_mask])
                    
                    if self.generalize and torch.any(~rf_mask) and torch.any(~vf_mask):
                        loss_ += self._calc_gauskde_sinkhorn_(valid_logent[~vf_mask], retain_logent[~rf_mask])
            else:   ## class forget
                if self.class_wise:
                    for c_i in f_cls:
                        f_mask = forget_labels == c_i
                        v_mask = valid_labels == c_i
                        loss_ += -1.0 * self._calc_gauskde_sinkhorn_(forget_logent[f_mask], valid_logent[v_mask])
                        
                    if self.generalize:
                        r_cls = torch.unique(retain_labels)
                        for c_i in r_cls:
                            r_mask = retain_labels == c_i
                            v_mask = valid_labels == c_i
                            if torch.any(r_mask) and torch.any(v_mask):
                                loss_ += self._calc_gauskde_sinkhorn_(retain_logent[r_mask], valid_logent[v_mask])
                else:
                    vf_mask = torch.eq(valid_labels.unsqueeze(1), forget_labels.unsqueeze(0)).any(dim=1).to(device)
                    loss_ += -1.0 * self._calc_gauskde_sinkhorn_(forget_logent, valid_logent[vf_mask])
                    
                    if self.generalize and torch.any(~vf_mask):
                        loss_ += self._calc_gauskde_sinkhorn_(valid_logent[~vf_mask], retain_logent)
        
        
        return -1.0 * loss_ + forget_logent.mean()



##################### adversarial loss ############################

class AdvLoss(nn.Module):
    def __init__(self, eps, clip_min=None, clip_max=None, sanity_checks=False):
        """
        Initialize the adversarial loss module.
        
        Args:
            eps (float): Maximum perturbation amount.
            clip_min (float): Minimum input component value.
            clip_max (float): Maximum input component value.
            sanity_checks (bool): If True, include asserts.
        """
        super(AdvLoss, self).__init__()
        assert eps >= 0, "eps must be greater than or equal to 0."
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.sanity_checks = sanity_checks
        
        if clip_min is not None or clip_max is not None:
            assert clip_min is not None and clip_max is not None, "Both clip_min and clip_max must be set."
            assert clip_min <= clip_max, "It must be the case that clip_min <= clip_max."
    
    def __sanity_checking__(self, x):
        asserts = []
        # If a data range was specified, check that the input was in that range
        if self.clip_min is not None:
            assert_ge = torch.all(
                torch.ge(x, torch.tensor(self.clip_min, device=x.device, dtype=x.dtype))
            )
            asserts.append(assert_ge)

        if self.clip_max is not None:
            assert_le = torch.all(
                torch.le(x, torch.tensor(self.clip_max, device=x.device, dtype=x.dtype))
            )
            asserts.append(assert_le)

        if self.sanity_checks:
            assert all(asserts)

    def forward(self, model, x, y):
        raise NotImplementedError

    def __call__(self, model, x, y):
        return self.forward(model, x, y)

class VATAdvLoss(AdvLoss):
    def __init__(self, eps=1.0, step_size=10.0, ip=1, sanity_checks=False):
        """VAT loss
        :param eps: hyperparameter of VAT (default: 1.0)  1e-5
        :param xi: hyperparameter of VAT (default: 10.0)  1e-6
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATAdvLoss, self).__init__(eps, None, None, sanity_checks)
        self.step_size = step_size
        self.ip = ip

    def __str__(self):
        return "VATAdvLoss"
    
    def forward(self, model, x, y=None):
        with torch.autograd.set_grad_enabled(False):
            pred = F.softmax(model(x), dim=1)

        # Initialize random unit tensor for power iteration
        # prepare random unit tensor
        x.requires_grad = True
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = normalize_l2(d)
        model.eval()
        # Power iteration to approximate the first singular vector
        with _disable_tracking_bn_stats(model):
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.eps * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_dist = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_dist.backward(retain_graph=True)
                # rnd_grad = torch.autograd.grad(adv_dist, d, retain_graph=True)[0]
                # d = torch.nn.functional.normalize(rnd_grad, p=2, dim=1)
                # d.requires_grad = True
                d = normalize_l2(d.grad)
                model.zero_grad()
    
            # Compute virtual adversarial perturbation
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            loss_ = F.kl_div(logp_hat, pred, reduction='batchmean')

        return loss_

## more adversarial loss: cw, deepfool, onepixel, spsa, 
## refer to:  https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks
class FGSMAdvLoss(AdvLoss):
    def __init__(self, eps=8.0/255, norm=np.inf, clip_min=None, clip_max=None, targeted=False, sanity_checks=False):
        """Adveratial loss based on the Fast Gradient Sign Method (FGSM).
        :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
        :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
        :param clip_min: (optional) float. Minimum float value for adversarial example components.
        :param clip_max: (optional) float. Maximum float value for adversarial example components.
        :param targeted: (optional) bool. Is the attack targeted or untargeted? 
                            Untargeted, the default, will try to make the label incorrect.
                            Untargeted if False. Defaults to False.
        :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
                    memory or for unit tests that intentionally pass strange input)
        """
        super(FGSMAdvLoss, self).__init__(eps, clip_min, clip_max, sanity_checks)
        assert norm in [np.inf, 1, 2], "Norm order must be either np.inf, 1, or 2."
        self.norm = norm
        self.targeted = targeted
    
    def __str__(self):
        return "FGSMAdvLoss"
    
    def forward(self, model, x, y=None):
        """
        Compute the adversarial loss using the Fast Gradient Sign Method (FGSM).
        Args:
            model (torch.nn.Module): The model to attack.
            x (torch.Tensor): Input images.
            y (torch.Tensor): True labels for x.
        """
        # x needs to be a leaf variable, of floating point type and have requires_grad being True for
        # its grad to be computed and stored properly in a backward call
        self.__sanity_checking__(x)
        
        if y is None:
            y = torch.argmax(model(x), dim=1)
        
        x_adv = x.clone()
        criterion = nn.CrossEntropyLoss()
        if self.eps > 0:
            x_ = x.clone().to(x.device)
            x_.requires_grad = True
            loss = criterion(model(x_), y)
            # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
            if self.targeted: 
                loss = -1.0 * loss

            model.zero_grad()
            loss.backward()
            
            # ## simple version
            # epsilon=0.03
            # grad = x.grad.data
            # # L2-norm constraint
            # grad_normalized = grad / torch.norm(grad, p=2)
            # x_adv = x + epsilon * grad_normalized
            
            opt_perturb = optimize_linear(x_.grad, self.eps, self.norm)
            # Add perturbation to original example to obtain adversarial example
            x_adv = x_ + opt_perturb
            if self.clip_min is not None and self.clip_max is not None:
                x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        # compute the adversarial loss
        outputs_adv = model(x_adv)
        adv_loss = criterion(outputs_adv, y)
        
        return adv_loss

## refer to: https://github.com/mpagli/Uncertainty-Driven-Perturbations/blob/59a915b/src/utils.py
class TradesAdvLoss(AdvLoss):
    def __init__(self, eps=0.1, step_size=0.03, ip=10, clip_min=None, clip_max=None, 
                    rs=False, use_alpha_scheduler=False, sanity_checks=False):
        """
        Initialize the Trades Loss module.
        
        Args:
            eps (float): Maximum perturbation amount.
            alpha (float): Step size for perturbation.
            ip (int): Number of iterations to perform perturbation.
            rs (bool): Whether to initialize perturbation randomly.
        """
        super(TradesAdvLoss, self).__init__(eps, clip_min, clip_max, sanity_checks)
        self.step_size = step_size
        self.ip = ip
        self.rs = rs
        self.alpha_scheduler = use_alpha_scheduler
    
    def __str__(self):
        return "TradesAdvLoss"
        
    def forward(self, model, x, y=None):        
        self.__sanity_checking__(x)
        
        delta = torch.zeros_like(x).to(x.device)
        if self.rs:
            delta.uniform_(-self.eps, self.eps)
        
        if self.alpha_scheduler:
            alpha_scheduler = lambda t: np.interp([t], [0, self.ip // 2, self.ip], 
                                    [self.step_size, max(self.eps/5, self.step_size), self.step_size])[0]
        
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)
        
        delta.requires_grad = True
        for itr in range(self.ip):
            x_ = x + delta
            if self.clip_min is not None and self.clip_max is not None:
                x_ = torch.clamp(x_, self.clip_min, self.clip_max)

            output = model(x_)
            probs_ = torch.softmax(output, dim=1)            
            loss = - (probs * (probs_ + 1e-8).log()).sum(dim=1).mean()
            loss.backward()
            
            grad_sign = delta.grad.detach().sign()
            if self.alpha_scheduler:
                delta.data = delta + alpha_scheduler(itr + 1) * grad_sign
            else:
                delta.data = delta + self.step_size * grad_sign
                
            if self.clip_min is not None and self.clip_max is not None:
                delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x
                
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.grad.zero_()

        x_adv = x + delta
        logits_x = model(x)
        p_x = torch.softmax(logits_x, dim=1)
        logits_adv = model(x_adv)
        p_adv = torch.softmax(logits_adv, dim=1)
        adv_loss = -1.0 * (p_x * (p_adv + 1e-8).log()).sum(dim=1).mean()
        return adv_loss


class UDPAdvLoss(AdvLoss):
    def __init__(self, eps=0.1, step_size=0.03, ip=10, clip_min=None, clip_max=None, 
                    rs=False, use_alpha_scheduler=False, sample_iters="none", sanity_checks=False):
        """
        Initialize the AdvUDP Loss module.
        
        Args:
            eps (float): Maximum perturbation amount.
            step_size (float): Step size for perturbation.
            ip (int): Number of iterations to perform perturbation.
            rs (bool): Whether to initialize perturbation randomly.
        """
        super(UDPAdvLoss, self).__init__(eps, clip_min, clip_max, sanity_checks)
        self.step_size = step_size
        self.ip = ip
        self.rs = rs
        self.use_alpha_scheduler = use_alpha_scheduler
        self.sample_iters = sample_iters

    def __str__(self):
        return "UDPAdvLoss"
    
    def __perturbation__(self, model, x, device):
        delta = torch.zeros_like(x).to(device)
        if self.rs:
            delta.uniform_(-self.eps, self.eps)
        if self.use_alpha_scheduler:
            alpha_scheduler = lambda t: np.interp([t], [0, self.ip // 2, self.ip], 
                                    [self.step_size, max(self.eps/2, self.step_size), self.step_size])[0]
        
        if self.sample_iters == 'uniform':
            shape = [delta.shape[0]] + [1] * (len(delta.shape)-1)
            sampled_iters = torch.randint(1,self.ip + 1,shape).expand_as(delta).to(device)
            
        delta.requires_grad = True
        for itr in range(self.ip):
            if self.clip_min is not None and self.clip_max is not None:
                x_ = torch.clamp(x + delta, self.clip_min, self.clip_max)
            else:
                x_ = x + delta

            output = model(x_)
            probs = torch.softmax(output, dim=1) + 1e-9
            loss = - (probs * probs.log()).sum(dim=1).mean()
            loss.backward()
            
            grad = delta.grad.detach().sign()
            if self.sample_iters != 'none':
                grad[sampled_iters <= itr] = 0.0
            
            if self.use_alpha_scheduler:
                delta.data = delta + alpha_scheduler(itr + 1) * grad
            else:
                delta.data = delta + self.step_size * grad
                
            if self.clip_min is not None and self.clip_max is not None:
                delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x
                
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.grad.zero_()

        return delta.detach()
    
    def forward(self, model, x, y=None):
        """
        Compute the uncertainty-driven adversarial loss.

        Args:
            x (torch.Tensor): Input images.
            y (torch.Tensor): True labels for x. Not used here 
        Returns:
            torch.Tensor: The perturbed images after uncertainty-driven perturbations.
        """
        self.__sanity_checking__(x)
        
        model.eval()
        delta = self.__perturbation__(model, x, x.device)

        # Create perturbed examples
        x_adv = x + delta
        # compute the adversarial loss
        outputs_adv = model(x_adv)
        criterion = nn.CrossEntropyLoss()
        loss_ = criterion(outputs_adv, y)
        
        return loss_

## Linear PGD 
class PGDAdvLoss(AdvLoss):
    def __init__(self, eps=8.0/255, step_size=2.0/255, ip=10, grad_sign=True, sanity_checks=False):
        """
        Initialize the AdvPGD Loss module.
        
        Args:
            eps (float): Maximum perturbation amount.
            alpha (float): Step size for perturbation.
            ip (int): Number of iterations to perform perturbation.
            norm (int): Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
        """
        super(PGDAdvLoss, self).__init__(eps, 0, 1, sanity_checks)
        self.step_size = step_size
        self.grad_sign = grad_sign
        self.ip = ip


    def __str__(self):
        return "PGDAdvLoss"
    
    @staticmethod
    def normalize_l2(x):
        """
            Expects x.shape == [N, C, H, W]
        """
        norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
        norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x / norm
    
    def forward(self, model, x, y):
        """
        Compute the adversarial loss using the Projected Gradient Descent (PGD) method.
        :param model: the classifier's forward method
        :param x: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        :param y: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        :return: perturbed batch of images
        """
        # unnormalize
        self.__sanity_checking__(x)
        
        x_ = (x + 1) / 2.0
        x_adv = (x_).detach()  # unnormalize
        x_adv += torch.zeros_like(x_adv).uniform_(-self.eps, self.eps)
        
        criterion = nn.CrossEntropyLoss()
        for i_ in range(self.ip):
            x_adv.to(x.device)
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(2 * x_adv - 1)
                loss = F.cross_entropy(logits, y, reduction='sum')
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            
            if self.grad_sign:
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = self.normalize_l2(grad.detach())
                x_adv = x_adv.detach() + self.step_size * grad
            
            x_adv = torch.min(torch.max(x_adv, x_ - self.eps), x_ + self.eps).clamp(self.clip_min, self.clip_max)
            
        x_adv = (x_adv * 2) - 1  # renormalize

        # compute the adversarial loss
        outs = model(x_adv)
        adv_loss = criterion(outs, y)
        
        return adv_loss

# refer to: https://github.com/cg563/simple-blackbox-attack/tree/master
class SimBAAdvLoss(AdvLoss):
    def __init__(self, eps=0.2, ip=1000, targeted=False, sanity_checks=False):
        """
        Initialize the AdvSimBA Loss module.
        
        Args:
            eps (float): Maximum perturbation amount.
            ip (int): Number of iterations to perform perturbation (adv. noise).
            clip_min (float): Minimum input component value.
            clip_max (float): Maximum input component value.
        """
        super(SimBAAdvLoss, self).__init__(eps, 0, 1, sanity_checks)
        self.ip = ip
        self.targeted = targeted
    
    def __str__(self):
        return "SimBAAdvLoss"
    
    def get_probs(self, model, x, y):
        outs = model(x)
        probs = torch.index_select(F.softmax(outs, dim=-1).data, 1, y)
        return torch.diag(probs)

    def forward(self, model, x, y):
        """
        Compute the adversarial loss using the SimBA method.
        :param model: the classifier's forward method
        :param x: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        :param y: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        :return: perturbed batch of images
        """
        pass
    
    
    ## for a single image
    def forward_single(self, model, x, y):
        # unnormalize
        self.__sanity_checking__(x)
        
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        last_prob = self.get_probs(model, x, y)
        
        x_adv = x.clone()
        for i_ in range(self.ip):
            diff = torch.zeros(n_dims)
            diff[perm[i_]] = self.eps
            x_i = x_adv - diff.view(x.size()).clamp(self.clip_min, self.clip_max)
            x_i.to(x.device)
            left_prob = self.get_probs(model, x_i, y)
            if self.targeted != (left_prob < last_prob):
                x_adv = x_i
                last_prob = left_prob
            else:
                x_i = x_adv + diff.view(x.size()).clamp(self.clip_min, self.clip_max)
                x_i.to(x.device)
                right_prob = self.get_probs(model, x_i, y)
                if self.targeted != (right_prob < last_prob):
                    x_adv = x_i
                    last_prob = right_prob
            
            if self.sanity_checks and i_ % 10 == 0:
                print(f"Step: {i_}, Prob: {last_prob}")
        
        outs = model(x_adv)
        criterion = nn.CrossEntropyLoss()
        adv_loss = criterion(outs, y)
        
        return adv_loss

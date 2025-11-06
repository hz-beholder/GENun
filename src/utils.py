# pyright: reportGeneralTypeIssues=false
import os
import errno
import random
import argparse
import shutil

import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist

from const import Summary


def pdb():
    import pdb
    pdb.set_trace

def get_project_root():
    return Path(__file__).parent.parent


def seed_everything(seed):
    '''
    Fixes the class-to-task assignments and most other sources of randomness, except CUDA training aspects.
    '''
    # Avoid all sorts of randomness for better replication
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def class_mask(data, targets, forget_classes):
    ## remove the `forget class` from the data 
    data_copy = data.clone()
    targets_copy = targets.clone()
    mask = ~torch.any(targets.unsqueeze(1) == forget_classes.unsqueeze(0), dim=1)
    data_copy = data_copy[mask]
    targets_copy = targets[mask]
    return (data_copy, targets_copy)


def trainable_params_(m):
    return [p for p in m.parameters() if p.requires_grad]


def check_sparsity(model):
    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    if zero_sum:
        remain_weight_ratie = 100 * (1 - zero_sum / sum_list)
        print("* remain weight ratio = ", 100 * (1 - zero_sum / sum_list), "%")
    else:
        print("no weight for calculating sparsity")
        remain_weight_ratie = None

    return remain_weight_ratie


def kaiming_weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def parameter_count(logger, model):
    count = 0
    for p in model.parameters():
        count += np.prod(np.array(p.shape))
    logger.debug(f'Total Number of Parameters: {count}')

def print_param_shape(logger, model):
    for k, p in model.named_parameters():
        logger.debug(k, p.shape)

def print_model_params(model, path):
    with open(path, 'w') as f:
        for name, param in model.named_parameters():
            print(name, param, file=f)
    
def set_batchnorm_mode(model, train=True):
    if isinstance(model, torch.nn.BatchNorm1d) or isinstance(model, torch.nn.BatchNorm2d):
        if train:
            model.train()
        else:
            model.eval()
    for l in model.children():
        set_batchnorm_mode(l, train=train)
    
def mkdir(directory):
    '''Make directory and all parents, if needed.
    Does not raise and error if directory already exists.
    '''
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def create_folder(folder):
    if not os.path.exists(folder):
        try:
            # self.logger.info("checking directory %s", folder)
            # self.logger.info("new directory %s created", folder)
            mkdir(folder)
        except OSError as error:
            # self.logger.info("deleting old and creating new empty %s", folder)
            # os.rmdir(folder)
            # self.logger.info("new empty directory %s created", folder)
            shutil.rmtree(folder)
            mkdir(folder)
    else:
        # self.logger.info("folder %s exists, do not need to create again.", folder)
        print(f"folder {folder} exists, do not need to create again.")
            

def enum_contains(enum_type, value):
    try:
        enum_type(value)
    except ValueError:
        return False
    return True


def union_of_vectors(*vectors):
    # Convert each tensor to a set
    sets = [set(v.tolist()) for v in vectors]
    
    # Compute the union of these sets
    union_set = set.union(*sets)
    
    # Convert the union set back to a tensor
    union_tensor = torch.tensor(list(union_set), dtype=vectors[0].dtype)
    
    return union_tensor


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    # Randomly select one image in the batch to mix with others
    assert alpha > 0
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(device)
    
    # Randomly choose a region to cut and paste
    h, w = x.size()[2], x.size()[3]
    cx = np.random.uniform(0, w)
    cy = np.random.uniform(0, h)
    w_cut = w * np.sqrt(1 - lam)
    h_cut = h * np.sqrt(1 - lam)
    
    x1 = np.clip(cx - w_cut // 2, 0, w)
    x2 = np.clip(cx + w_cut // 2, 0, w)
    y1 = np.clip(cy - h_cut // 2, 0, h)
    y2 = np.clip(cy + h_cut // 2, 0, h)

    # Apply CutMix
    x[:, :, int(y1):int(y2), int(x1):int(x2)] = x[rand_index, :, int(y1):int(y2), int(x1):int(x2)]
    # Adjust labels
    y_a, y_b = y, y[rand_index]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    return x, y_a, y_b, lam


def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Compute the mixup data. Return mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def list_of_ints(param):
    if param.find(',') != -1:
        return list(map(int, param.split(',')))
    else:
        if param.lower() == 'none':
            return None
        else:
            return [int(param)]

def argparse2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def shuffle_in_union(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b



def dict_to_str(d, precision=4):
    str_ = ""
    for k in sorted(d.keys()):
        value = d[k]
        if isinstance(value, float) and precision is not None:
            formatted_value = f'{value:.{precision}f}'  # Format float with specified precision
        else:
            formatted_value = value
        str_ += f'{k}: {formatted_value}, '
    return str_.rstrip(', ') 


def save_model(model, out_path):
    '''
    Used for saving the pretrained model, not for intermediate breaks in running the code.
    '''
    folder = os.path.dirname(out_path)
    if not os.path.exists(folder):
        mkdir(folder)
    
    if not (out_path.endswith('.pt') or out_path.endswith('.pth')):
         out_path += '.pt'
    torch.save(model, out_path)
    
    # state = {'state_dict': model.state_dict()}
    # torch.save(state, out_path)

def load_model(in_path):
    '''
    Used for loading the pretrained model, not for intermediate breaks in running the code.
    '''
    assert os.path.isfile(in_path), "no checkpoint found at '{}'".format(in_path)
    # checkpoint = torch.load(in_path, map_location=torch.device('cuda'))
    # model.load_state_dict(checkpoint['state_dict'])
    # return model
    
    model = torch.load(in_path, map_location=torch.device('cpu'), weights_only=False)
    return model

class AverageMeter(object):
    # Sourced from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """Computes and stores the average and current value"""
    def __init__(self, name="meter", fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum * 1.0 / self.count * 1.0

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# Modify based on need for confusion
def confusion_matrix(num_class, conf_type, num_change, exch_classes=None):
    mat = []
    for i in range(num_class):
        mat.append([])
        for j in range(num_class):
            mat[i].append(0)

    if conf_type == 'noise':
        for i in range(num_class):
            for j in range(num_class):
                if i != j:
                    mat[i][j] = int(num_change / (num_class - 1))
            leftover = num_change % num_class
            c_t = [j for j in range(num_class) if j != i]
            for j in random.sample(c_t, leftover):
                mat[i][j] += 1

    if conf_type == 'exchange':
        assert(exch_classes is not None)
        for i in exch_classes:
            for j in exch_classes:
                if i != j:  mat[i][j] = num_change
    return mat


def get_accuracy(output, target):
    if output.shape[1] > 1:
        pred = output.argmax(dim=1, keepdim=True)
        return pred.eq(target.view_as(pred)).float().mean().item()
    else:
        pred = output.clone()
        pred[pred>0]=1
        pred[pred<=0]=-1
        return pred.eq(target.view_as(pred)).float().mean().item()


def plt_entropy_distribution(X_tr, Y_tr, X_forget, title="Entropy Distribution"):
    plt.figure()
    ax = plt.gca()
    sns.distplot(np.log(X_tr[Y_tr == 1]).reshape(-1), kde=False, norm_hist=True, rug=False, label='retain', ax=ax)
    sns.distplot(np.log(X_tr[Y_tr == 0]).reshape(-1), kde=False, norm_hist=True, rug=False, label='test', ax=ax)
    sns.distplot(np.log(X_forget).reshape(-1), kde=False, norm_hist=True, rug=False, label='forget', ax=ax)
    ax.legend(prop={'size': 14})
    ax.tick_params(labelsize=12)
    ax.set_title(title,size=18)
    ax.set_xlabel('Log of Entropy',size=14)
    ax.set_ylim(0,0.4)
    ax.set_xlim(-35,2)
    

def freeze_layers_for_lp(model,use_head=False):
    for name, param in model.named_parameters():
        if use_head:
            if "fc" not in name and "head" not in name: 
                param.requires_grad = False
        else:
            if "fc" not in name:
                param.requires_grad = False
    return model


def unfreeze_layers_for_ft(model,use_head=False):
    for param in model.parameters():
        param.requires_grad = True
    return model


def unfreeze_layers_for_lpfrz_ft(model,use_head=False):
    for name, param in model.named_parameters():
        if use_head:
            if "fc" not in name and "head" not in name: 
                param.requires_grad = False
        else:
            if "fc" not in name:
                param.requires_grad = False
        
    return model


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def make_functional(model):
    orig_params = tuple(model.parameters())
    # Remove all the parameters in the model
    names = []

    for name, p in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append(name)

    return orig_params, names

def load_weights(model, names, params, as_params=False):
    for name, p in zip(names, params):
        if not as_params:
            set_attr(model, name.split("."), p)
        else:
            set_attr(model, name.split("."), torch.nn.Parameter(p))
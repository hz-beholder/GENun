from os import path
import numpy as np
from copy import deepcopy

from learner import *
from logger import create_logger
from model_deep import get_model as get_deep_model
from model_shallow import get_model as get_shallow_model
from utils import load_model

__SHALLOWMODELSET__ = ['lr', 'dt', 'svm', 'mlp', 'rf']
__DEEPMODELSET__ = ['mlps', 'logistic', 'simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'densenet', 'vgg', 'vit','ALLCNN']


class ShallowModels(object):
    def __init__(self, model_name, log_path, logname):
        assert model_name in __SHALLOWMODELSET__, "invalid model type, which should be one of %s" % __SHALLOWMODELSET__
        self.model_name = model_name
        self.log_path = log_path
        self.logger = create_logger(log_path, '{}_{}_{}'.format(logname, model_name, 'shallow'))
        self._model = self.get_model(model_name)
    
    @classmethod
    def is_valid(cls, model_name):
        return model_name in __SHALLOWMODELSET__
    
    def get_model(self, model_name, **kwargs):
        self.logger.info("determine_model for %s" % model_name)
        return get_shallow_model(model_name, **kwargs)
        # if model_name == 'lr':
        #     return LR()
        # elif model_name == 'dt':
        #     return DT()
        # elif model_name == 'svm':
        #     return SVM()
        # elif model_name == 'mlp':
        #     maxlr = kwargs.get('maxlr', 0.01)
        #     hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (100,))
        #     return MLP(lr=maxlr, hidden_layer_sizes=hidden_layer_sizes)
        # elif model_name == 'rf':
        #     return RF()
        # else:
        #     raise Exception("invalid model name")
    
    def train_(self, train_x, train_y, save_path=".", suffix=""):
        self.logger.info("Training model over dataset")
        if suffix != "":
            suffix = "_" + suffix
        save_name = path.join(save_path, f"{self.model_name}{suffix}.pkl")
        return self._model.train(train_x, train_y, save_name)

    def predict_proba(self, test_x):
        return self._model.predict_proba(test_x)

    def test_acc(self, test_x, test_y):
        return self._model.test_acc(test_x, test_y)

    def test_auc(self, test_x, test_y):
        return self._model.test_auc(test_x, test_y)

    def load(self, save_path, suffix=""):
        if suffix != "":
            suffix = "_" + suffix
        save_name = path.join(save_path, f"{self.model_name}{suffix}.pkl")
        self.logger.info("=> loading model from %s" % save_name)
        self._model.load(save_name)


class DeepModels(object):
    def __init__(self, model_name, feature_dimension, num_classes, log_path, logname, pretrained=True):
        assert model_name in __DEEPMODELSET__, "invalid model type, which should be one of %s" % __DEEPMODELSET__
        self.model_name = model_name
        self.log_path = log_path
        self.logger = create_logger(log_path, '{}_{}_{}'.format(logname, model_name, 'deep'))
        self._model = get_deep_model(model_name, feature_dimension, num_classes, pretrained)
        self.params = None
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def is_valid(cls, model_name):
        return model_name in __DEEPMODELSET__
    
    def parameter_config(self, **kwargs):
        self.params = {}
        self.params['device'] = kwargs.get('device', 'cuda')
        self.params['name'] = kwargs.get('name', f'{self.model_name}')
        self.params['num_classes'] = kwargs.get('num_classes', 10)
        self.params['procedure'] = kwargs.get('procedure', 'none')
        self.params['lossfn'] = kwargs.get('lossfn', 'ce')
        self.params['epochs'] = kwargs.get('epochs', 40)
        self.params['lamb_reg'] = kwargs.get('lamb', 0.0)
        self.params['no_reg_epochs'] = kwargs.get('no_reg_epochs', 0)
        self.params['patience'] = kwargs.get('patience', 20)
        self.params['optim'] = kwargs.get('optim', 'adam')
        self.params['minlr'] = kwargs.get('minlr', 0.0001)
        self.params['maxlr'] = kwargs.get('maxlr', 0.001)
        self.params['momentum'] = kwargs.get('momentum', 0.9)
        self.params['weight_decay'] = kwargs.get('weight_decay', 0.0)
        self.params['scheduler'] = kwargs.get('scheduler', 'none')
        self.params['resume'] = kwargs.get('resume', None)
        self.params['is_dp_defense'] = kwargs.get('is_dp_defense', False)
        self.params['delta'] = kwargs.get('delta', 1e-5)
        self.params['epsilon'] = kwargs.get('epsilon', 1.0)
        self.params['clip'] = kwargs.get('clip', 1.0)
        self.params['batch_size'] = kwargs.get('batch_size', 128)
        self.params['mixup'] = kwargs.get('mixup', 'none')
        self.params['mixup_alpha'] = kwargs.get('mixup_alpha', 0.1)
        self.params['mixup_prob'] = kwargs.get('mixup_prob', 0.5)
        self.params['loss_sign'] = kwargs.get('loss_sign', 1)
        self.params['disable_batchnorm'] = kwargs.get('disable_batchnorm', False)
        self.params['regularization'] = kwargs.get('regularization', 'none')
        self.params['dynamic_regular'] = kwargs.get('dynamic_regular', False)
    
    def load(self, save_path):
        self.logger.info("=> loading checkpoint '{}'".format(save_path))
        self._model = load_model(save_path)
        
    def resume(self, resume_path):
        self.logger.info(f"Model resumed from {resume_path}")
        self.load(path.join(resume_path))
    
    def train_(self, train_loader, test_loader, valid_loader, ckpt_path, device='cuda', mask=None):
        assert self.params is not None, "Please set the parameter configuration first"
        
        num_classes = len(np.unique(train_loader.dataset.targets))  # self.params['num_classes']  #
        num_train, num_test = len(train_loader.dataset), len(test_loader.dataset)
        num_val = len(valid_loader.dataset) if valid_loader is not None else 0
        
        # console_logger = create_logger(self.log_path, 'deep_train_%s' % self.model_name)
        self.logger.info(f"Checkpoint name: {self.params['name']}_{self.params['procedure']}")
        self.logger.info(f"Number of Classes: input data-{num_classes}, model-{self.params['num_classes']}")
        self.logger.info(f"Train set size: {num_train} | Valid set size: {num_val} | Test set size: {num_test}")

        print("training device: ", device)
        self._model.to(device)
        # model_outpath = path.join(save_path, f"{self.params['name']}_{self.params['procedure']}")
        best_model, train_time = train(self._model, train_loader, valid_loader, test_loader, self.params, self.logger, 
                                            device=device, # out_model_name=self.params['name'], 
                                            lossfn=self.params['lossfn'], epochs=self.params['epochs'], patience=self.params['patience'],
                                            optim_type=self.params['optim'], scheduler_type=self.params['scheduler'],
                                            momentum=self.params['momentum'], maxlr=self.params['maxlr'], minlr=self.params['minlr'],
                                            weight_decay=self.params['weight_decay'], checkpoint_dir=ckpt_path, mask=mask)
        self.logger.info(f"Training time: {train_time} \n\n")
        self._model = best_model
        return best_model
        # return  #best_model
    
    def predict_proba(self, test_loader, logits=True):
        self._model.eval()
        self._model = self._model.to(self.device)
        res = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                output = self._model(inputs)
                if logits:
                    post_ = output
                else:
                    post_ = F.softmax(output, dim=1)
                res.append(post_.detach().cpu())
            res = torch.cat(res, dim=0)
        
        return res
        
    def test_acc(self, test_loader, lossfn='ce', device='cuda'):
        self._model.eval()
        criterion = loss_picker(lossfn)
        loss_, acc_ = test(self._model, test_loader, criterion, device)
        return acc_
    
    def test_auc(self, test_loader, device='cuda'):
        self._model.eval()
        self._model = self._model.to(device)
        num_classes = len(np.unique(test_loader.dataset.targets))
        
        with torch.no_grad():
            logits_, labels_ = [], []
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = self._model(inputs)
                outputs = F.softmax(outputs, dim=1)
                logits_.append(outputs.detach().cpu().numpy())
                labels_.append(labels.detach().cpu().numpy())
            logits_ = np.concatenate(logits_, axis=0)
            labels_ = np.concatenate(labels_, axis=0)

            #TODO: check this for multiclass case
            if num_classes > 2:
                return roc_auc_score(labels_, logits_, multi_class='ovr')
            else:
                return roc_auc_score(labels_, logits_[:, 1])

    
    @staticmethod
    def to_extractor(model, model_name, freeze=True, device='cuda'):
        extractor = None
        if model_name != 'logistic':
            extractor = deepcopy(model)
            extractor = extractor.to(device)
            extractor = extractor.eval()
            extractor = nn.Sequential(*list(extractor.children())[:-1])
            if freeze:
                for param in extractor.parameters():
                    param.requires_grad = False
        
        return extractor
    

def distance(logger, model_src, model_tar):
    distance = 0
    normalization = 0
    for (k, p), (k0, p0) in zip(model_src.named_parameters(), model_tar.named_parameters()):
        space='  ' if 'bias' in k else ''
        current_dist = (p.data0-p0.data0).pow(2).sum().item()
        current_norm = p.data0.pow(2).sum().item()
        distance += current_dist
        normalization += current_norm
    logger.info(f'Distance: {np.sqrt(distance)}')
    logger.info(f'Normalized Distance: {1.0 * np.sqrt(distance / normalization)}')
    return 1.0 * np.sqrt(distance / normalization)
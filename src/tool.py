import os
import argparse
import config

class PathGenerator:
    def __init__(self):
        pass
        # self.args = args
        # self._save_name_ = "_".join((args.dataset, args.arch, str(args.unlearn_size),))
        #TODO: actual save_name parameters
        # if args.is_dp_defense:
        #     self._save_name_ += "_DP"
    
    # def get_log_path(self):
    #     return os.path.join(self.args.log_dir, self.args.exp_name)
    
    # def get_split_path(self, suffix=''):
    #     if len(suffix) > 0:
    #         suffix = '_' + suffix
    #     return os.path.join(config.SPLIT_DATA_PATH, self._save_name_ + f'{suffix}.split')
    
    @staticmethod
    def get_target_model_path(unlearn_method, save_path=None):
        if save_path is None:
            save_path = config.MODEL_PATH
        return os.path.join(save_path, unlearn_method, '')
    
    @staticmethod
    def get_checkpoint_path(subpath='', save_path=None):
        if save_path is None:
            save_path = config.CHECKPOINT_PATH
        return os.path.join(save_path, subpath, '')
    
    @staticmethod
    def get_attack_model_path(subpath='', save_path=None):
        if save_path is None:
            save_path = config.ATTACK_MODEL_PATH
        return os.path.join(save_path, subpath,  '')
    
    @staticmethod
    def get_attack_data_path(subpath='', save_path=None):
        if save_path is None:
            save_path = config.ATTACK_DATA_PATH
        return os.path.join(save_path, subpath, '')
    
    @staticmethod
    def get_attack_result_path():
        return os.path.join(config.ATTACK_RESULT_PATH, "")
    
    @staticmethod
    def get_eval_result_path():
        return os.path.join(config.EVAL_RESULT_PATH, "")


def complete_parameters(args_dict):
    argumentlist = ['log_dir', 'logname', 'exp_name', 'procedure',  'dataset', 'seed', #'name',
                    'model', 'pretrained', 'num_classes', 'lossfn', 'epochs', 'patience',
                    'optim', 'minlr', 'maxlr', 'momentum', 'weight_decay', 'scheduler', 
                    'resume', 'is_dp_defense', 'delta', 'epsilon', 'clip',
                    'batch_size', 'mixup', 'mixup_prob', 'mixup_alpha',
                    'lamb', 'regularization', 'dynamic_regular',
                    'loss_sign', 'disable_batchnorm', 'forget_class', 'num_to_forget', 'device', 'cuda']
    default_param_set = argparse.Namespace(
        log_dir = '.',
        logname = 'train',
        exp_name = 'trial_unlearn',
        procedure = 'original',
        dataset = 'cifar10',
        seed = int(1),
        model_name = 'resnet',
        pretrained = True,
        num_classes = None,
        lossfn = 'ce',
        epochs = 50,
        patience = 20,
        optim = 'adam',
        minlr = 0.0001,
        maxlr = 0.001,
        momentum = 0.9,
        weight_decay = 5e-5,
        scheduler = 'CosineAnnealingWarmRestarts',
        resume = None,
        is_dp_defense = False,
        delta = 1e-5,
        epsilon = 1.0,
        clip = 1.0,
        batch_size = int(64),
        mixup = 'none',  # the type of mixup: 'none', 'mixup', 'cutmix', 'augmix'
        mixup_alpha = 1.0,
        mixup_prob = 0.5,
        loss_sign = 1,
        disable_batchnorm = False,
        num_to_forget = 0,
        forget_class = None,
        device = 'cuda',
        lamb = 0.0,
        regularization = 'none',
        dynamic_regular = False,
        cuda = None
        # validsplit = 0.2,
        # name = None,    # Name to be used for saving the model and initialized later 
    )
    
    dict_params = vars(default_param_set)
    for k_i in argumentlist:
        if k_i in args_dict:
            dict_params[k_i] = args_dict[k_i]
    
    # Map mixup_type to mixup parameter
    if 'mixup_type' in args_dict:
        dict_params['mixup'] = args_dict['mixup_type']
    
    dict_params['model_name'] = args_dict['arch']
    
    # if dict_params['cuda'] == '':
    #     dict_params['device'] = 'cpu'
    # else:
    #     gpus = dict_params['cuda'].split(',')
    #     if len(gpus) > 1:
    #         dict_params['cuda'] = [int(gpu) for gpu in gpus]
    #     else:
    #         dict_params['cuda'] = int(gpus[0])

    # default_param_set = argparse.Namespace(**dict_params)
    # return default_param_set
    return dict_params

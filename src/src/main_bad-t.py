import argparse
from copy import deepcopy
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from logger import create_logger
from model_deep import get_model
from const import SingletonString
from utils import load_model, mkdir, save_model, seed_everything
from bad_teacher_unlearning import blindspot_unlearner, UNSIR_create_noisy_loader, UNSIR_noise_train, UNSIR_noise, fit_one_unlearning_cycle



def data_model_test(model, data, batch_size, num_workers, device, transform=None):
    from learner import complete_test
    test_loader = DataLoader(data, batch_size, False, num_workers=num_workers)
    acc, auc = complete_test(model, test_loader, device, transform)
    return acc, auc



if __name__ == '__main__':    
    args = argparse.Namespace()
    args.cuda = 1
    args.arch = "vit" #"resnet18" #"simple_cnn"  #
    args.dataset = "cifar100"   #'cifar10' #"svhn"  #
    args.preproc_train_transform = 'normal'
    args.preproc_test_transform = 'test'
    args.select_classes = None
    args.batch_size = 256
    args.num_workers = 32
    args.lr = 0.00001
    args.epochs = 2
    
    args.optim = 'adam'
    args.unlearn_method = 'BadT'  # 'UNSIR' # 
    
    args.num_samples = -1
    args.forget_classes = []  # [2]  #
    if args.forget_classes is None or len(args.forget_classes) == 0:
        args.num_to_forget = 5000
        args.valid_size = 5000
        # args.num_to_forget = 7000
        # args.valid_size = 7000
    else:
        args.num_to_forget = 10000
        args.valid_size = 10000
    args.log_dir = 'logs'
    
    
    outfix = ""
    if args.forget_classes is not None and len(args.forget_classes) > 0:
        outfix = f"{outfix}_FGCLS-{args.forget_classes[0]}"
    
    args.out_dir = f'./outs/{args.dataset}_{args.num_samples}{outfix}'
    args.exp_name = f'{args.dataset}_{args.num_samples}{outfix}'
    
    
    if args.forget_classes is None or len(args.forget_classes) == 0:
        ## for all data and selective unlearn (random selected samples)
        # args.model_path = "./outs/cifar10_-1/model_bases/cifar10_resnet18/ORG/" + \
        #     "cifar10_resnet18_seed-3407_Nf-5000_ep-20_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-testregular_0.0_sched-None.pth"
        # args.model_path = "./outs/svhn_-1/model_bases/svhn_simple_cnn/ORG/" + \
        #       "svhn_simple_cnn_seed-0_Nf-7000_ep-40_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"
        # args.model_path = "./outs/cifar100_-1/model_bases/cifar100_vit/ORG/"+\
        #     "cifar100_vit_seed-0_Nf-5000_ep-40_bs-256_lr-[0_001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"
        args.model_path = "./outs/cifar100_-1/model_bases/cifar100_vit/ORG/"+\
            "cifar100_vit_seed-0_Nf-5000_ep-20_bs-256_lr-[0_0001-0_0002]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"
    else:
        ## for class unlearn 
        args.model_path = "./outs/cifar10_-1_FGCLS-2/model_bases/cifar10_resnet18/ORG/" + \
            "cifar10_resnet18_seed-0_Nf-10000_Cf-[2]_ep-40_bs-256_lr-[0_001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_0.0_sched-None.pth"

    str_seed_sidx = args.model_path.find('seed-')
    str_seed_eidx = args.model_path.find('_Nf')
    args.seed = int(args.model_path[str_seed_sidx+5:str_seed_eidx])
    
    str_fgt_sidx = args.model_path.find('Cf-')
    str_fgt_eidx = args.model_path.find('_ep')
    if str_fgt_sidx > 0:
        args.forget_classes = [int(x) for x in args.model_path[str_fgt_sidx+4:str_fgt_eidx-1].split(',')]
    
    
    outpath = SingletonString()
    outpath.content = args.out_dir
    
    # from model_deep import get_model
    from transforms import Transforms
    from tool import PathGenerator as pather
    
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    logpath = os.path.join(args.log_dir, args.exp_name)
    logger = create_logger(logpath, f'trail_unlearning_{args.unlearn_method}')
    logger.info(args)
    
    from data_tool import DataLoaderTool, DataStore, construct_data
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(args.dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(args.dataset)
    if args.arch in ['resnet50', 'densenet', 'vit']:
        dt_size = 224
        feature_dims = [3, 224, 224]
    
    processor = Transforms(dt_mean, dt_std, dt_size)
    train_transform = processor.get_transform(args.preproc_train_transform, num_classes)
    test_transform = processor.get_transform(args.preproc_test_transform, num_classes)
    transform = None
    
    dataset_conf = { "num_samples": args.num_samples, "select_classes": args.select_classes}
    train_data, test_data = DataLoaderTool.load_dataset(args.dataset, train_transform, test_transform, **dataset_conf)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    data, data_unlearn, data_retain = construct_data(train_data, test_data, train_size=args.num_samples, valid_size=args.valid_size, 
                                                     forget_size=args.num_to_forget, forget_classes=args.forget_classes)

    ## load original model and test
    logger.info(f"Load original model from @ {args.model_path}")
    model_org = load_model(args.model_path).to(device)
    unlearning_teacher = get_model(args.arch, feature_dims, num_classes, pretrained=False).to(device)
    student_model = deepcopy(model_org)
    
    # perform scrub unlearning
    start_time = time.time()
    if args.unlearn_method == 'BadT':
        KL_temperature = 1
        optimizer = torch.optim.Adam(student_model.parameters(), lr = args.lr)
        blindspot_unlearner(model=student_model, unlearning_teacher = unlearning_teacher, full_trained_teacher = model_org, 
                            retain_data = data['retain'], forget_data = data['forget'], 
                            epochs = args.epochs, optimizer = optimizer, lr = args.lr, batch_size = args.batch_size, 
                            num_workers = args.num_workers, device = device, KL_temperature = KL_temperature)
        out_fn = f"bad_teacher_unlearn_epochs-{args.epochs}_optim-Adam_bs-{args.batch_size}_lr-{args.lr}_T-{KL_temperature}.pth"
        
    elif args.unlearn_method == 'UNSIR':
        assert args.forget_classes is not None and len(args.forget_classes) > 0, "Forget classes are not supported in UNSIR"
        noise_batch_size = args.batch_size
        learn_epochs = 150
        noise = UNSIR_noise(args.batch_size, feature_dims[-1], feature_dims[0], feature_dims[1]).to(device)
        noise = UNSIR_noise_train(noise, student_model, args.forget_classes[0], learn_epochs, noise_batch_size, device=device)
        noise_loder = UNSIR_create_noisy_loader(noise, args.forget_classes[0], data['retain'], args.batch_size, device=device)
        
        test_loader = DataLoader(data['test'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        history = fit_one_unlearning_cycle(args.epochs, student_model, noise_loder, test_loader, lr=args.lr, device=device)
        out_fn = f"UNSIR_unlearn_epochs-{args.epochs}_bs-{args.batch_size}_lr-{args.lr}.pth"
        
    logger.info(f" +++++++++++ Time taken: {time.time()-start_time:.4f} sec +++++++++++ ")
    
    
    out_path = pather.get_target_model_path(f"{args.dataset}_{args.arch}/{args.unlearn_method}/")
    model_outpath = os.path.join(out_path, out_fn)
    logger.info(f"Save the model to @ {model_outpath}")
    save_model(student_model, model_outpath)
    
    ## EVALUATION: using the test data to evaluate the model
    if args.forget_classes is None or len(args.forget_classes) == 0:
        logger.info(" +++++++++++ Unlearned Model Test +++++++++++ ")
        train_acc, train_auc =  data_model_test(student_model, data['train'], args.batch_size, args.num_workers, device, transform)
        train_retain_acc, train_retain_auc = data_model_test(student_model, data['retain'], args.batch_size, args.num_workers, device, transform)
        valid_acc, valid_auc =  data_model_test(student_model, data['valid'], args.batch_size, args.num_workers, device, transform)
        test_acc, test_auc =  data_model_test(student_model, data['test'], args.batch_size, args.num_workers, device, transform)
        forget_acc, forget_auc = data_model_test(student_model, data['forget'], args.batch_size, args.num_workers, device, transform)
        retain_acc, retain_auc = data_model_test(student_model, data_retain, args.batch_size, args.num_workers, device, transform)
        logger.info(
            " ###### [Train, Train_rem, Valid, Test, Forget, Remain-Data] "
            + f"  Acc: {train_acc:.4f}, {train_retain_acc:.4f}, {valid_acc:.4f}, {test_acc:.4f}, {forget_acc:.4f}, {retain_acc:.4f} ######"
        )
        #     + f"| Auc: {train_auc:.4f}, {valid_auc:.4f}, {test_auc:.4f}, {forget_auc:.4f}, {retain_auc:.4f}, {retain_auc:.4f}###### ")
    else:
        logger.info(f" +++++++++++ Forget classes: {args.forget_classes} +++++++++++ ")
        train_retain_acc, _ =  data_model_test(student_model, data['retain'], args.batch_size, args.num_workers, device, transform)
        train_forget_acc, _ =  data_model_test(student_model, data['forget'], args.batch_size, args.num_workers, device, transform)
        test_retain_acc, _ = data_model_test(student_model, data['test'], args.batch_size, args.num_workers, device, transform)
        test_forget_acc, _ = data_model_test(student_model, data_unlearn['test'], args.batch_size, args.num_workers, device, transform)
        valid_retain_acc, _ = data_model_test(student_model, data['valid'], args.batch_size, args.num_workers, device, transform)
        valid_forget_acc, _ = data_model_test(student_model, data_unlearn['valid'], args.batch_size, args.num_workers, device, transform)
        
        logger.info(" ###### Remain [Train, Valid, Test]:"
                            + f"  Acc: {train_retain_acc:.4f}, {valid_retain_acc:.4f}, {test_retain_acc:.4f} ######")
        logger.info(" ###### Forget [Train, Valid, Test]:"
                            + f"  Acc: {train_forget_acc:.4f}, {valid_forget_acc:.4f}, {test_forget_acc:.4f} ######")

    logger.info("********************************* \n\n")
    print("Done!")
    
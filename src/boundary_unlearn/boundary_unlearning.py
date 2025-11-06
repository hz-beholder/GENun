import copy
import utils
import tqdm
import time
import numpy as np
import torch
from torch import nn
from .models import init_params as w_init
from .adv_generator import LinfPGD, inf_generator, FGSM
from .trainer import eval, loss_picker, optimizer_picker
from .utils import get_forget_loader, save_model


def boundary_shrink(ori_model, train_forget_loader, device, #dt, dv, test_loader, evaluate,
                    bound=0.1, step=2 / 255, iter=5, poison_epoch=10):  #, forget_class=0, path='./', **kwargs):
    # start = time.time()
    norm = True  # None#True if data_name != "mnist" else False
    random_start = True  # False if attack != "pgd" else True

    test_model = copy.deepcopy(ori_model).to(device)
    unlearn_model = copy.deepcopy(ori_model).to(device)
    start_time = time.time()
    adv = LinfPGD(test_model, bound, step, iter, norm, random_start, device)
    # adv = FGSM(test_model, bound, norm, random_start, device)
    forget_data_gen = inf_generator(train_forget_loader)
    batches_per_epoch = len(train_forget_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.00001, momentum=0.9)
    # optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.000001, momentum=0.9)

    num_hits = 0
    num_sum = 0
    nearest_label = []

    for itr in tqdm.tqdm(range(poison_epoch * batches_per_epoch)):
        x, y = forget_data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        test_model.eval()
        x_adv = adv.perturb(x, y, target_y=None, model=test_model, device=device)
        adv_logits = test_model(x_adv)
        pred_label = torch.argmax(adv_logits, dim=1)
        if itr >= (poison_epoch - 1) * batches_per_epoch:
            nearest_label.append(pred_label.tolist())
        num_hits += (y != pred_label).float().sum()
        num_sum += y.shape[0]

        # adv_train
        unlearn_model.train()
        unlearn_model.zero_grad()
        optimizer.zero_grad()

        ori_logits = unlearn_model(x)
        ori_loss = criterion(ori_logits, pred_label)

        loss = ori_loss  # - KL_div
        loss.backward()
        optimizer.step()

    print('attack success ratio:', (num_hits / num_sum).float())
    print('boundary shrink time:', (time.time() - start_time))
    # print(nearest_label)
    # np.save('nearest_label', nearest_label)
    
    return unlearn_model
    
    # prefix = f'{kwargs.get("model", "")}'
    # if len(prefix) > 0:
    #     prefix += '_'
    # suffix = f"bound_{bound}_step_{step}_iter_{iter}_poison_epoch_{poison_epoch}"
    # save_model(unlearn_model, f'{path}{prefix}boundary_shrink_unlearn_model_{suffix}.pth')

    
    # test_forget_loader, test_remain_loader = get_forget_loader(dv, forget_class)
    # _, train_remain_loader = get_forget_loader(dt, forget_class)

    # mode = 'pruned' if evaluate else ''
    # _, test_acc = eval(model=unlearn_model, data_loader=test_loader, mode=mode, print_perform=evaluate, device=device,
    #                    name='test set all class')
    # _, forget_acc = eval(model=unlearn_model, data_loader=test_forget_loader, mode=mode, print_perform=evaluate,
    #                      device=device, name='test set forget class')
    # _, remain_acc = eval(model=unlearn_model, data_loader=test_remain_loader, mode=mode, print_perform=evaluate,
    #                      device=device, name='test set remain class')
    # _, train_forget_acc = eval(model=unlearn_model, data_loader=train_forget_loader, mode=mode, print_perform=evaluate,
    #                            device=device, name='train set forget class')
    # _, train_remain_acc = eval(model=unlearn_model, data_loader=train_remain_loader, mode=mode, print_perform=evaluate,
    #                            device=device, name='train set remain class')
    # print('test acc:{:.2%}, forget acc:{:.2%}, remain acc:{:.2%}, train forget acc:{:.2%}, train remain acc:{:.2%}'
    #       .format(test_acc, forget_acc, remain_acc, train_forget_acc, train_remain_acc))
    # end = time.time()
    # print('Time Consuming:', end - start, 'secs')
    # return unlearn_model


# def boundary_expanding(feature_extractor, classifier, learning_rate, epochs,
#                        train_forget_loader, test_loader, test_forget_loader, test_remain_loader,
#                        train_remain_loader, optimization, device, evaluate, path='./'):
def boundary_expanding(ori_model, learning_rate, epochs, train_forget_loader, 
                       # test_loader, test_forget_loader, test_remain_loader,train_remain_loader, 
                            optimization, device):  #, evaluate, path='./', **kwargs):
    start_time = time.time()
    # num_classes = 10
    # n_filter2 = int(192 * 0.5)
    # narrow_model = copy.deepcopy(ori_model).to(device)
    # feature_extractor = narrow_model.features
    # classifier = narrow_model.classifier
    # cls_in_dims = classifier.in_features
    # num_classes = classifier.out_features
    classifier = copy.deepcopy(ori_model.fc)
    feature_dims = classifier.in_features
    num_classes = classifier.out_features
    
    widen_classifier = nn.Linear(feature_dims, num_classes + 1)
    w_init(widen_classifier)
    widen_model = copy.copy(ori_model)
    widen_model.fc = widen_classifier
    # widen_model = nn.Sequential(feature_extractor, nn.AvgPool1d(), widen_classifier)    
    widen_model = widen_model.to(device)

    # dict = widen_classifier.state_dict()
    for name, params in classifier.named_parameters():
        # print(name, params.data)
        if 'weight' in name:
            widen_classifier.state_dict()['weight'][0:10, ] = classifier.state_dict()[name][:, ]
        elif 'bias' in name:
            widen_classifier.state_dict()['bias'][0:10, ] = classifier.state_dict()[name][:, ]

    forget_data_gen = inf_generator(train_forget_loader)
    batches_per_epoch = len(train_forget_loader)
    # finetune_epochs = 30

    criterion = loss_picker('cross')
    optimizer = optimizer_picker(optimization, widen_model.parameters(), lr=learning_rate, momentum=0.9)
    # centr_optimizer = optimizer_picker(optimization, widen_model.parameters(), lr=0.00001, momentum=0.9)
    # adv_optimizer = optimizer_picker(optimization, adv_model.parameters(), lr=0.001, momentum=0.9)

    for itr in tqdm.tqdm(range(epochs * batches_per_epoch)):
        x, y = forget_data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        widen_logits = widen_model(x)

        # target label
        target_label = torch.ones_like(y, device=device)
        target_label *= num_classes

        # adv_train
        widen_model.train()
        widen_model.zero_grad()
        optimizer.zero_grad()
        widen_loss = criterion(widen_logits, target_label)
        widen_loss.backward()
        optimizer.step()

    widen_classifier = widen_model.fc
    pruned_classifier = nn.Linear(feature_dims, num_classes)
    for name, params in widen_classifier.named_parameters():
        # print(name)
        if 'weight' in name:
            pruned_classifier.state_dict()['weight'][:, ] = widen_classifier.state_dict()[name][0:10, ]
        elif 'bias' in name:
            pruned_classifier.state_dict()['bias'][:, ] = widen_classifier.state_dict()[name][0:10, ]

    # pruned_model = nn.Sequential(feature_extractor, nn.Flatten(), pruned_classifier)
    pruned_model = copy.copy(widen_model)
    pruned_model.fc = pruned_classifier
    pruned_model = pruned_model.to(device)
    print('boundary expanding time:', (time.time() - start_time))
    
    return pruned_model

    # mode = 'pruned' if evaluate else ''
    # _, test_acc = eval(model=pruned_model, data_loader=test_loader, mode=mode, print_perform=evaluate, device=device,
    #                    name='test set all class')
    # _, forget_acc = eval(model=pruned_model, data_loader=test_forget_loader, mode=mode, print_perform=evaluate,
    #                      device=device, name='test set forget class')
    # _, remain_acc = eval(model=pruned_model, data_loader=test_remain_loader, mode=mode, print_perform=evaluate,
    #                      device=device, name='test set remain class')
    # _, train_forget_acc = eval(model=pruned_model, data_loader=train_forget_loader, mode=mode, print_perform=evaluate,
    #                            device=device, name='train set forget class')
    # _, train_remain_acc = eval(model=pruned_model, data_loader=train_remain_loader, mode=mode, print_perform=evaluate,
    #                            device=device, name='train set remain class')
    # print('test acc:{:.2%}, forget acc:{:.2%}, remain acc:{:.2%}, train forget acc:{:.2%}, train remain acc:{:.2%}'
    #       .format(test_acc, forget_acc, remain_acc, train_forget_acc, train_remain_acc))
    # end = time.time()
    # print('Time Consuming:', end - start, 'secs')
    
    # prefix = f'{kwargs.get("model", "")}'
    # if len(prefix) > 0:
    #     prefix += '_'
    # suffix = f'lr_{learning_rate}_epochs_{epochs}'
    # save_model(widen_model, f'{path}{prefix}boundary_expand_widen_model_{suffix}.pth')
    # save_model(pruned_model, f'{path}{prefix}boundary_expand_pruned_model_{suffix}.pth')
    # return pruned_model

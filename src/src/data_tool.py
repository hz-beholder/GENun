import pickle
from os import path
from collections import OrderedDict

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset # TensorDataset
from torch.utils.data import Subset, SubsetRandomSampler, random_split
from torchvision import datasets
from torchvision.datasets import SVHN

import config
from utils import create_folder
from dataset.small_CIFFAR10 import Small_Binary_CIFAR10, Small_CIFAR10, Small_CIFAR5

CATEGORICAL_DATASETS = ['adult', 'accident', 'location']

IMAGE_DATASETS = ['imagenet', 'svhn', 'stl10', 'mnist', #'small_mnist', 'small_mnist_2', 
                'smbincifar10', 'smcifar5', 'smcifar10', 'cifar10', 'cifar100', ]

class ListToDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input, label = self.samples[idx]
        return torch.tensor(input).float(), torch.tensor(label).long()

class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.targets = [self.dataset.targets[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        img, label = self.dataset[data_idx]
        return img, label


def flatten_subset(subset):
    if subset is None or not isinstance(subset, Subset):
        return subset
    
    indices = subset.indices
    while isinstance(subset.dataset, Subset):
        subset = subset.dataset
        indices = [subset.indices[i] for i in indices]
    return SubsetDataset(subset.dataset, indices)

# Custom dataset class
class RelabeledDataset(Dataset):
    def __init__(self, dataset, indices, new_labels):
        self.dataset = dataset
        self.indices = indices
        self.targets = new_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        img, _ = self.dataset[data_idx]
        new_label = self.targets[idx]
        return img, new_label

class CustomerDataset(Dataset):
    def __init__(self, inputs, ture_labels, logits_pred):
        self.inputs = inputs
        self.labels = ture_labels
        self.predictions = logits_pred

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.predictions[idx]


class ConcatDatasets(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        # Combine the targets from all datasets
        self.targets = torch.cat([torch.tensor(dataset.targets) for dataset in datasets])

class CustomSVHN(SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(CustomSVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.targets = self.labels  # Make `targets` an alias for `labels`



class DataLoaderTool:
    @staticmethod
    def __mnist_transform__():
        def expand_to_two_channels(img):
            """ Convert a single channel image to a two-channel image by replicating the channel """
            return img.expand(2, -1, -1)  # Expands the tensor to 2 channels

        mean, std, crop_size = dataset_normalizer('mnist')
        augment_transform = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=4),  # Randomly crop the image
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine transformation
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.ToTensor(),  # Convert images to tensor
            transforms.Lambda(expand_to_two_channels),  # Convert to two channels
            transforms.Normalize(mean, std)  # Normalize using MNIST mean and std for each channel
        ])

        # Define transformations for testing - No augmentation, just conversion and normalization
        normal_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensor
            transforms.Lambda(expand_to_two_channels),  # Convert to two channels
            transforms.Normalize(mean, std)  # Normalize using MNIST mean and std for each channel
        ])

        return augment_transform, normal_transform
    
    @staticmethod
    def __cifar_transform__():
        mean, std, crop_size = dataset_normalizer('cifar')
        augment_transform = transforms.Compose([
            transforms.Pad(padding=4, fill=(125, 123, 113)),
            transforms.RandomCrop(crop_size, padding=0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        normal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return augment_transform, normal_transform

    @staticmethod
    def __stl_transform__():
        mean, std, crop_size = dataset_normalizer('stl')
        augment_transform = transforms.Compose([
            transforms.Pad(padding=4, fill=(125, 123, 113)),
            transforms.RandomCrop(crop_size, padding=0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        normal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return augment_transform, normal_transform

    @staticmethod
    def __imagenet_transform__():
        mean, std, crop_size = dataset_normalizer('imagenet')
        augment_transform = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        normal_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return augment_transform, normal_transform

    def __svhn_transform__():
        mean, std, crop_size = dataset_normalizer('svhn')
        augment_transform = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        normal_transform = transforms.Compose([
                    transforms.Scale(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
        ])
        return augment_transform, normal_transform

    
    def default_transform(dataset_name):
        if dataset_name == 'mnist':
            return DataLoaderTool.__mnist_transform__()
        elif 'cifar' in dataset_name:
            return DataLoaderTool.__cifar_transform__()
        elif dataset_name == 'stl10':
            return DataLoaderTool.__stl_transform__()
        elif dataset_name == 'imagenet':
            return DataLoaderTool.__imagenet_transform__()
        elif dataset_name == 'svhn':
            return DataLoaderTool.__svhn_transform__()
        else:
            raise Exception("invalid dataset name")
    
    @staticmethod
    def load_dataset(dataset_name, train_transform=None, test_transform=None, augment=False, use_default_transform=False, **kwargs):
        raw_path = config.ORIGINAL_DATASET_PATH
        if use_default_transform:
            aug_transform, test_transform = DataLoaderTool.default_transform(dataset_name)
            if augment:
                train_transform = aug_transform
            else:
                train_transform = test_transform
        else:
            if train_transform is None:
                train_transform = transforms.Compose([transforms.ToTensor()])
            elif test_transform is None:
                test_transform = transforms.Compose([transforms.ToTensor()])
        
        ## load different dataset based on the daataset name and the above transforms
        if dataset_name == 'mnist':
            train_dataset = datasets.MNIST(raw_path + 'mnist', train=True, download=True, transform=train_transform)
            test_dataset = datasets.MNIST(raw_path + 'mnist', train=False, download=True, transform=test_transform)
        elif dataset_name == 'smbincifar10':
            n_sams = kwargs.get('num_samples')
            cls_ = kwargs.get('select_classes')
            target_remap = dict(zip(cls_, range(len(cls_))))
            target_transform = transforms.Lambda(lambda x: target_remap.get(x, -1))
            subset_name = "smcifar2-" + "_".join([str(i) for i in cls_])
            train_dataset = Small_Binary_CIFAR10(root=config.ORIGINAL_DATASET_PATH + subset_name, select_classes=cls_, 
                                num_samples=n_sams, train=True, transform=train_transform, target_transform=target_transform)
            test_dataset = Small_Binary_CIFAR10(root=config.ORIGINAL_DATASET_PATH + subset_name, select_classes=cls_, 
                                num_samples=n_sams, train=False, transform=test_transform, target_transform=target_transform)
        elif dataset_name == 'smcifar5':
            n_sams = kwargs.get('num_samples')
            cls_ = kwargs.get('select_classes')
            target_remap = dict(zip(cls_, range(len(cls_))))
            target_transform = transforms.Lambda(lambda x: target_remap.get(x, -1))
            subset_name = "smcifar5-" + "_".join([str(i) for i in cls_])
            train_dataset = Small_CIFAR5(root=config.ORIGINAL_DATASET_PATH + subset_name, select_classes=cls_, 
                                num_samples=n_sams, train=True, transform=train_transform, target_transform=target_transform)
            test_dataset = Small_CIFAR5(root=config.ORIGINAL_DATASET_PATH + subset_name, select_classes=cls_, 
                                num_samples=n_sams, train=False, transform=test_transform, target_transform=target_transform)
        elif dataset_name == 'smcifar10':
            n_sams = kwargs.get('num_samples')
            subset_name = "small_cifar10_" + str(n_sams)
            train_dataset = Small_CIFAR10(root=config.ORIGINAL_DATASET_PATH + subset_name, num_samples=n_sams, 
                                            train=True, transform=train_transform)
            test_dataset = Small_CIFAR10(root=config.ORIGINAL_DATASET_PATH + subset_name, num_samples=n_sams, 
                                            train=False, transform=test_transform)
        elif dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(raw_path + 'cifar10', train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(raw_path + 'cifar10', train=False, download=True, transform=test_transform)
        elif dataset_name == 'cifar100':
            train_dataset = datasets.CIFAR100(raw_path + 'cifar100', train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR100(raw_path + 'cifar100', train=False, download=True, transform=test_transform)
        elif dataset_name == 'stl10':
            train_dataset = datasets.STL10(raw_path + 'stl10', split='train', download=True, transform=train_transform)
            test_dataset = datasets.STL10(raw_path + 'stl10', split='test', download=True, transform=test_transform)
        elif dataset_name == 'svhn':
            train_dataset = CustomSVHN(raw_path + 'svnh', split='train', download=True, transform=train_transform)
            test_dataset = CustomSVHN(raw_path + 'svnh', split='test', download=True, transform=test_transform)
        elif dataset_name == 'imagenet':
            train_dataset = datasets.ImageFolder(raw_path + 'imagenet', split='train', transform=train_transform)
            test_dataset = datasets.ImageFolder(raw_path + 'imagenet', split='test', transform=test_transform)
        else:
            raise Exception("invalid dataset name")

        train_dataset.targets = np.array(train_dataset.targets)
        test_dataset.targets = np.array(test_dataset.targets)
        
        return train_dataset, test_dataset

class DataStore:
    @staticmethod
    def create_basic_folders():
        folder_list = [config.ROOT_PATH, config.MODEL_PATH, config.ATTACK_DATA_PATH, 
                    config.ATTACK_MODEL_PATH, config.ATTACK_RESULT_PATH, config.EVAL_RESULT_PATH]  #config.SPLIT_DATA_PATH, 
        for folder in folder_list:
            create_folder(folder)
    
    @staticmethod
    def get_dataset_info(dataset):
        features_dims = {
            # "adult": [14],
            # "accident": [29],
            # "location": [168],
            
            "small_mnist": [28, 28],
            "small_mnist_2": [28, 28],
            "mnist": [28, 28],
            
            "stl10": [96, 96, 3],
            "small_cifar2": [32, 32, 3],
            "small_cifar5": [32, 32, 3],
            "small_cifar10": [32, 32, 3],
            "cifar10": [32, 32, 3],
            "cifar100": [32, 32, 3],
            "imagenet": [224, 224, 3],
            "svhn": [32, 32, 3],
        }
        
        crop_size = {
            "cifar10": 32,
            "cifar100": 32,
            "mnist": 28,
            "stl10": 96,
            "imagenet": 224,
            "svhn": 32,
        }
        
        num_classes = {
            # "adult": 2,
            # "accident": 3,
            # "location": 9,
            "small_mnist": 10,
            "small_mnist_2": 2,
            "mnist": 10,
            "small_cifar2": 2,
            "small_cifar5": 5,
            "small_cifar10": 10,
            "cifar10": 10,
            "cifar100": 100,
            "stl10": 10,
            "imagenet": 1000,
            "svhn": 10,
        }
        
        return features_dims[dataset], num_classes[dataset]

    @staticmethod
    def get_normalizer(dataset):
        return dataset_normalizer(dataset)

    @staticmethod
    def save_record_split(record_split, path):
        pickle.dump(record_split, open(path, 'wb'))

    @staticmethod
    def load_record_split(path):
        record_split = pickle.load(open(path, 'rb'))
        return record_split
    
    @staticmethod
    def save_attack_data(attack_data, outfn):
        if not path.exists(outfn):
            create_folder(outfn)
        pickle.dump((attack_data), open(outfn, 'wb'))

    @staticmethod
    def load_attack_data(infn):
        attack_train_data = pickle.load(open(infn, 'rb'))
        return attack_train_data


def dataset_normalizer(dataset):
    if 'cifar' in dataset:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        crop_size = 32
    elif 'imagenet' in dataset:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_size = 224
    elif 'mnist' in dataset:
        mean = (0.1307,)
        std = (0.3081,)
        crop_size = 28
    elif 'svhn' in dataset:
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980,0.2010,0.1970)
        crop_size = 32
    elif 'stl10' in dataset:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_size = 96
    else:
        raise ValueError("Dataset not found")

    return mean, std, crop_size

def split_class_data(dataset, forget_classes, num_to_forget):
    forget_index = []
    class_retain_index = []
    retain_index = []
    sum = 0
    for i, (_, target) in enumerate(dataset):
        if target in forget_classes and sum < num_to_forget:
            forget_index.append(i)
            sum += 1
        elif target in forget_classes and sum >= num_to_forget:
            class_retain_index.append(i)
            retain_index.append(i)
            sum += 1
        else:
            retain_index.append(i)
    return forget_index, retain_index, class_retain_index

def get_forget_loader(dt, forget_classes, batch_size):
    idx = []
    # count = 0
    els_idx = []
    for i in range(len(dt)):
        _, lbl = dt[i]
        if lbl in forget_classes:
            # if forget:
            #     count += 1
            #     if count > forget_num:
            #         continue
            idx.append(i)
        else:
            els_idx.append(i)
    forget_loader = DataLoader(dt, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(idx)) # , drop_last=True
    retain_loader = DataLoader(dt, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(els_idx)) # , drop_last=True
    return forget_loader, retain_loader

def construct_data(train_data, test_data, train_size=-1, 
                   valid_size=5000, forget_size=2000, forget_classes=None):
    if train_size == -1:
        train_size = len(train_data) - valid_size
    if train_size + valid_size > len(train_data):
        print("Warning: sum of train and validation sizes exceeds dataset size.")
        train_size = len(train_data) - valid_size
    
    retain_size = len(train_data) - train_size - valid_size
    train_data, valid_data, retain_data = random_split(train_data, [train_size, valid_size, retain_size])
    
    data = OrderedDict()
    data_unlearn = OrderedDict()
    data_retain = None
    data['train'] = flatten_subset(train_data)
    # 增加边界检查，防止 forget_size 超过 train 样本数
    available_forget = len(train_data)
    if forget_size > available_forget:
        print(f"Warning: forget_size({forget_size}) > train_data({available_forget}), 自动调整为 {available_forget}")
        forget_size = available_forget
    if forget_classes is None or len(forget_classes) == 0:
        train_retain_set, forget_set = random_split(train_data, [len(train_data) - forget_size, forget_size])
        data['retain'] = flatten_subset(train_retain_set)
        data['valid'] = flatten_subset(valid_data)
        data['test'] = flatten_subset(test_data)
        data['forget'] = flatten_subset(forget_set)
        data_unlearn['train'] = flatten_subset(forget_set)
        data_retain = flatten_subset(retain_data)
    else:
        train_forget_index, train_retain_index, class_retain_index = split_class_data(train_data, forget_classes, num_to_forget=forget_size)
        valid_forget_index, valid_retain_index, _ = split_class_data(valid_data, forget_classes, num_to_forget=len(valid_data))
        test_forget_index, test_retain_index, _ = split_class_data(test_data, forget_classes, num_to_forget=len(test_data))
        data['retain'] = flatten_subset(Subset(train_data, train_retain_index))
        data['valid'] = flatten_subset(Subset(valid_data, valid_retain_index))
        data['test'] = flatten_subset(Subset(test_data, test_retain_index))
        data['forget'] = flatten_subset(Subset(train_data, train_forget_index))
        data_retain = flatten_subset(Subset(train_data, class_retain_index))

        data_unlearn['train'] = flatten_subset(Subset(train_data, train_forget_index))
        data_unlearn['valid'] = flatten_subset(Subset(valid_data, valid_forget_index))
        data_unlearn['test'] = flatten_subset(Subset(test_data, test_forget_index))
        
    return data, data_unlearn, data_retain

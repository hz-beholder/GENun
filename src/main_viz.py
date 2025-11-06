import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from transforms import Transforms
from utils import load_model, seed_everything
from data_tool import DataLoaderTool, DataStore, construct_data


def extract_features(model, data_loader, device):
    features, labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            features.append(model(inputs).detach().cpu().numpy())
            labels.append(labels.numpy())
    return np.concatenate(features), np.concatenate(labels)


if __name__ == "__main__":
    seed = 0  # 3407
    cuda = 0
    num_samples = 10000  # -1 #
    num_workers = 4
    dataset = 'cifar10'
    preproc_train_transform = 'normal'
    preproc_test_transform = 'test'
    
    valid_size = 3000   # 5000  #
    batch_size = 128
    num_to_forget = 1000   # 5000  #
    forget_classes = None
    
    
    in_dir = f'./outs/{dataset}_{num_samples}/model_bases/'
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    
    mod_fn = ''
    model_path = os.path.join(in_dir, mod_fn)
    
    dt_mean, dt_std, dt_size = DataStore.get_normalizer(dataset)
    feature_dims, num_classes = DataStore.get_dataset_info(dataset)
    processor = Transforms(dt_mean, dt_std, dt_size)
    train_transform = processor.get_transform(preproc_train_transform, num_classes)
    test_transform = processor.get_transform(preproc_test_transform, num_classes)
    
    dataset_conf = { "num_samples": num_samples, "select_classes": None}
    train_data, test_data = DataLoaderTool.load_dataset(dataset, train_transform, test_transform, **dataset_conf)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    data, data_unlearn, data_remain = construct_data(train_data, test_data, train_size=num_samples, valid_size=valid_size, 
                                                     forget_size=num_to_forget, forget_classes=forget_classes)
    

    model_ = load_model(model_path).to(device)
    model_.fc = nn.Identity()
    model_.eval()
    
    features, labels = extract_features(model_, test_loader, device)
    
    tsne = TSNE(n_components=2, random_state=42)
    feature_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(feature_2d[:, 0], feature_2d[:, 1], c=labels, cmap='tab10', s=5, alpha=0.5)
    plt.colorbar(scatter, ticks=range(num_classes), label='Classes')
    plt.title('t-SNE visualization of features')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.show()
    
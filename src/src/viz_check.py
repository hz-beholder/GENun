import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_tool import DataLoaderTool, DataStore, construct_data
from model_deep import get_model
from transforms import Transforms
from utils import load_model, seed_everything


os.environ['OPENBLAS_NUM_THREADS'] = '64'

cuda = 5
seed = 0
num_samples = -1
num_workers = 4
batch_size = 128
num_to_forget = 1000
forget_classes = None
valid_size = 5000

arch = "resnet18"
dataset='cifar10'
preproc_train_transform = "normal"
preproc_test_transform = "test"

model_path = f"./outs/cifar10_{num_samples}_difval/model_bases/cifar10_resnet18/ORG/" + \
       f"cifar10_resnet18_seed-{seed}_Nf-{num_to_forget}_ep-25_bs-256_lr-[0_0001-0_001]_wd-5e-05_opt-Adam_trTrans-normal_tsTrans-test_regular_1e-07_sched-None.pth"


seed_everything(seed)
device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')


dt_mean, dt_std, dt_size = DataStore.get_normalizer(dataset)
feature_dims, num_classes = DataStore.get_dataset_info(dataset)
processor = Transforms(dt_mean, dt_std, dt_size)
train_transform = processor.get_transform(preproc_train_transform, num_classes)
test_transform = processor.get_transform(preproc_test_transform, num_classes)
transform = None

dataset_conf = { "num_samples": num_samples, "select_classes": None}
train_data, test_data = DataLoaderTool.load_dataset(dataset, train_transform, test_transform, **dataset_conf)

data_, data_unlearn, data_retain = construct_data(train_data, test_data, train_size=num_samples, valid_size=valid_size, 
                                                     forget_size=num_to_forget, forget_classes=forget_classes)

retain_loader = DataLoader(data_['retain'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
forget_loader = DataLoader(data_['forget'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
 
# init_mod_ = get_model(arch, feature_dims, num_classes, pretrained=True).to(device)
model = load_model(model_path)

# Function to extract features
def extract_features(loader, model, device):
    model.fc = nn.Identity()  # Remove the last fully connected layer
    model = model.to(device)
    
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.detach().cpu())
            labels.append(targets)
    
    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels


def find_top_k_same_class_samples(feature, label, features_set, labels_set, K=5):
    # Get same-class samples, excluding the ones in exclude_indices
    same_class_idx = (labels_set == label)
    same_class_features = features_set[same_class_idx]    
    same_class_features_np = same_class_features.clone().detach().numpy().copy()

    similarities = cosine_similarity(feature.reshape(1, -1), same_class_features_np)[0]
    top_k_indices = similarities.argsort()[-K:][::-1]
    return same_class_features[list(top_k_indices)]

# Extract features for training and test sets
train_features, train_labels = extract_features(retain_loader, model, device)
test_features, test_labels = extract_features(test_loader, model, device)
forget_features, forget_labels = extract_features(forget_loader, model, device)

K = 20

# Initialize lists to store selected features
selected_ft_train = []
selected_ft_test = []
selected_lb_train = []
selected_lb_test = []

# For each sample in S, find the top-K same-class samples from both training and test sets
for i in range(len(forget_features)):
    feature = forget_features[i]
    label = forget_labels[i]
    
    top_k_train = find_top_k_same_class_samples(feature, label, train_features, train_labels, K)
    top_k_test = find_top_k_same_class_samples(feature, label, test_features, test_labels, K)
    
    # Append the features and labels
    selected_ft_train.extend(top_k_train)
    selected_ft_test.extend(top_k_test)
    selected_lb_train.extend([label] * K)
    selected_lb_test.extend([label] * K)

print(f"size of one_sample: {len(top_k_train), len(top_k_test)}")
print(f"size of selected_features: {len(selected_ft_train)}")
print(f"size of selected_labels: {len(selected_lb_train)}")
# Convert to tensors

# selected_features = torch.stack(selected_features)
# selected_labels = torch.tensor(selected_labels)

np.save('./data_org_feature.npy', {'forget': (forget_features, forget_labels), 
                                 'train': (train_features, train_labels), 
                                 'test': (test_features, test_labels)})

np.save('./data_select_feature.npy', {'select_train': (selected_ft_train, selected_lb_train), 
                               'select_test': (selected_ft_test, selected_lb_test)})

# # Apply t-SNE to visualize the feature distribution
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(selected_features)

# # Plot t-SNE results
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=selected_labels, cmap='tab10', alpha=0.7)
# plt.legend(*scatter.legend_elements(), title="Classes")
# plt.title('t-SNE Visualization of CIFAR-10 Features')
# plt.show()

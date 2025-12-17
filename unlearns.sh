#! /bin/bash

cuda_id=$1
dataset=$2
data_aug=$3
num_samples=$4
unlearn_size=$5
arch=$6
forget_class=$7
mixup_prob=$8
model_path=$9
unlearn_method=${10}
seed=${11}
valid_size=${12}

# seed=1  # 3407  #
# valid_size=1000 # 5000  # 5000 #
test_transform="test"
fisher_type="golaker"  #"variate" #
train_mode="ovr"
out_suffix=""
# num_samples=10000
# unlearn_size=5000
# model='resnet18'
# dataset='cifar10'

maxlr=0.002
epochs=25
patience=35
batch_size=256
last_k=0
lamb=0
recursion_depth=0
r_averaging=1
mask_threshold=0.5
dynamic_regular=False
wga_beta=7


unlearner='src/main_unlearn.py'

suffix="" #"_models"   # option: "", "-mx", "ag", "mag"
if [ "$data_aug" = "normal" ]; then
    ## non-robust train
    # augment=False
    train_transform='normal'
    mixup='none'
elif    [ "$data_aug" = "augment" ]; then
    ## Only augmentation train
    # augment=True
    train_transform='random'
    mixup='none'
    suffix="_ag"
elif [ "$data_aug" = "cutmix" ]; then
    ## Only cutmix train
    # augment=False
    train_transform='normal'
    mixup='cutmix'
    suffix="_cmx"
# 在 unlearns.sh 第58-63行，修改为：
elif [ "$data_aug" = "cutout" ]; then
    ## Only cutout train
    # augment=False
    train_transform='cutout'
    mixup='none'
    suffix="_cot"
elif [ "$data_aug" = "mixup" ]; then
    ## Only mix-up train
    # augment=False
    train_transform='normal'
    mixup='mixup'
    suffix="_mx"
elif [ "$data_aug" = "augment_cutmix" ]; then
    ## augmentation and mix-up train
    # augment=True
    train_transform='random'
    mixup='cutmix'
    suffix="_cmag"
elif [ "$data_aug" = "augment_mixup" ]; then
    ## augmentation and mix-up train
    # augment=True
    train_transform='random'
    mixup='mixup'
    suffix="_mag"
elif [ "$data_aug" = "autoaug" ]; then
    ## AutoAugment train
    # augment=False
    train_transform='autoaug'
    mixup='none'
    suffix="_autoaug"
elif [ "$data_aug" = "randerase" ]; then
    ## RandErasing train
    train_transform='randerase'
    mixup='none'
    suffix="_randerase"
elif [ "$data_aug" = "randaug" ]; then
    ## RandAugment train
    train_transform='randaug'
    mixup='none'
    suffix="_randaug"
elif [ "$data_aug" = "augmix" ]; then
    ## AugMix train
    train_transform='augmix'
    mixup='none'
    suffix="_augmix"
elif [ "$data_aug" = "ba" ]; then
    ## Basic Augmentation (Random Crop & Flip)
    train_transform='random'
    mixup='none'
    suffix="_ba"
elif [ "$data_aug" = "ba_cutmix" ]; then
    ## Basic Augmentation + CutMix
    train_transform='random'
    mixup='cutmix'
    suffix="_ba_cutmix"
elif [ "$data_aug" = "ba_mixup" ]; then
    ## Basic Augmentation + MixUp
    train_transform='random'
    mixup='mixup'
    suffix="_ba_mixup"
else
    echo "Invalid Data Augmentation Method"
    exit 1
fi
echo "Data Preprocess: $data_aug :(Augment: $augment, Mixup: $mixup ($mixup_prob))"


if [ -n "$out_suffix" ]; then
    out_suffix="_$out_suffix"
fi

if [ $forget_class != "none" ]; then
    out_suffix="${out_suffix}_FGCLS-${forget_class}"
fi

if [ "$mixup_prob" != "1.0" ]; then
    out_suffix="${out_suffix}_mxp-${mixup_prob}"
fi


output_path="./outs/${dataset}_${num_samples}${suffix}${out_suffix}/"
exp_name="logs/${dataset}_${num_samples}${suffix}${out_suffix}"

# output_path="./outs_sz/${dataset}_${num_samples}_${unlearn_size}${out_suffix}/"
# exp_name="logs_sz/${dataset}_${num_samples}_${unlearn_size}${out_suffix}"

echo "Output Path: $output_path"
echo "Exp Name: $exp_name"


# Model training or unlearning
if [ $unlearn_method == "ORG" ] || [ $unlearn_method == "none" ] || [ $unlearn_method == "original" ]; then  # just train the original model
    ## Original model
    unlearn_apoc="none"
    maxlr=0.0002
    minlr=0.0002
    epochs=15
    patience=10  #35
    echo "Original: (lr, # epochs): " $maxlr, $epochs
elif [ $unlearn_method == "RT" ] || [ $unlearn_method == "scratch" ]; then
    ## Ground Truth ``
    ## Scratch Retrained model
    unlearn_apoc="scratch"
    maxlr=0.0002
    minlr=0.0002
    epochs=15
    patience=10
    echo "Scratch: (lr, # epochs): " $maxlr, $epochs
elif [ $unlearn_method == "FT" ] || [ $unlearn_method == "finetune" ]; then
    ## Finetune
    unlearn_apoc="finetune"
    maxlr=1e-3
    minlr=1e-3
    patience=2
    epochs=10  #5  #
    echo "Finetune: (lr, # epochs): ", $maxlr, $epochs
elif [ $unlearn_method == "L1FT" ] || [ $unlearn_method == "l1_sparse" ]; then
    ## Finetune
    unlearn_apoc="l1_sparse"
    maxlr=1e-3
    minlr=1e-3
    patience=1
    epochs=10  #5  #
    lamb=1e-3
    dynamic_regular=True  # False 
    echo "L1-sparse: (lr, # epochs, lamb, dynamic_regular): ", $maxlr, $epochs, $lamb, $dynamic_regular
elif [ $unlearn_method == "GA" ] || [ $unlearn_method == "neggrad" ]; then
    ## Negative Gradient / Gradient Ascent
    unlearn_apoc="negative_gradient"
    patience=3
    epochs=15
    maxlr=5e-5 #1e-4
    minlr=5e-5 
    echo "Negative Gradient: (lr, # epochs): " $maxlr, $epochs
elif [ $unlearn_method == "WGA" ] || [ $unlearn_method == "wga" ]; then
    ## Weighted Gradient Ascent
    unlearn_apoc="WGA"
    patience=1
    epochs=15
    maxlr=5e-5
    minlr=5e-5
    echo "Weighted Gradient Ascent: (lr, # epochs, beta): " $maxlr, $epochs, $wga_beta
elif [ $unlearn_method == "RL" ] || [ $unlearn_method == "randomlabel" ]; then
    ## Random Label
    unlearn_apoc="random_label"
    patience=3
    maxlr=5e-5
    minlr=5e-5
    epochs=15
    echo "Random Label: (lr, # epochs): " $maxlr, $epochs

elif [ $unlearn_method == "LKL" ] || [ $unlearn_method == "lastk" ]; then
    ## Last K-layer
    unlearn_apoc="lastklayer"
    patience=5
    last_k=1
    epochs=20
    maxlr=1e-3
    minlr=1e-3
    echo "Last K-layer: (lr, # epochs, last-k): " $maxlr, $epochs, $last_k
elif [ $unlearn_method == "FRA" ] || [ $unlearn_method == "fisher" ]; then
    ## Fisher Approximation
    unlearn_apoc="fisherapprox"
    lamb="1e-8"
    maxlr=1e-3
    minlr=1e-3
    echo "Fisher Approximation: (lambda): " $lamb
elif [ $unlearn_method == "UI" ] || [ $unlearn_method == "influence" ]; then
    ## Fisher Approximation
    unlearn_apoc="influence_unlearn"
    last_k=2
    r_averaging=1
    recursion_depth=10
    maxlr=1e-3
    minlr=1e-3
    echo "Inflence Unlearning: (last_k, recursion_depth, r_averaging): " $last_k, $recursion_depth, $r_averaging

elif [ $unlearn_method == "SALUN" ] || [ $unlearn_method == "salun" ]; then
    ## SalUn
    unlearn_apoc="salun"
    patience=3
    maxlr=2e-4
    minlr=2e-4
    epochs=10
    mask_threshold=0.5
    echo "Random Label: (lr, # epochs, mask_threshold): " $maxlr, $epochs, $mask_threshold
fi

python $unlearner --out_dir=$output_path --exp_name=$exp_name --seed=$seed --cuda=$cuda_id \
                  --dataset=$dataset --num_samples=$num_samples --valid_size=$valid_size \
                  --num_to_forget=$unlearn_size --forget_classes=$forget_class \
                  --arch=$arch --unlearn_method=$unlearn_apoc --resume=$model_path \
                  --epochs=$epochs --patience=$patience --batch_size=$batch_size --maxlr=$maxlr --minlr=$minlr --lamb=$lamb --mask_threshold=$mask_threshold \
                  --train_transform=$train_transform --test_transform=$test_transform --dynamic_regular=$dynamic_regular \
                  --mixup_type=$mixup --mixup_prob=$mixup_prob \
                  --last_k=$last_k --fisher_type=$fisher_type --train_mode=$train_mode \
                  --recursion_depth=$recursion_depth --r_averaging=$r_averaging \
                  --wga_beta=$wga_beta
echo "Done!"
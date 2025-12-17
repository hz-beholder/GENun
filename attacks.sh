#! /bin/bash

cuda_id=$1
data_aug=$2
arch=$3
test_ratio=$4  # 0.2  # 0.5  #
mixup_prob=$5
dataset=$6
num_samples=$7
unlearn_size=$8
forget_class=$9
model_path=${10}
unlearn_method=${11}
seed=${12}
valid_size=${13}
out_suffix=${14}

# arch='resnet18'
# dataset='cifar10'

# out_suffix=$3
# num_samples=10000
# unlearn_size=1000
# seed=3407
# valid_size=5000
epochs=20
batch_size=256
stacked=False

attacker='src/main_attack.py'

suffix=""   # option: "", "-mx", "ag", "mag"
if [ $data_aug == "normal" ]; then
    ## non-robust train
    # augment=False
    train_transform='normal'
    mixup='none'
elif [ $data_aug == "augment" ]; then
    ## Only augmentation train
    # augment=True
    train_transform='random'
    mixup='none'
    suffix="_ag"
elif [ $data_aug == "cutmix" ]; then
    ## Only cutmix train
    # augment=False
    train_transform='normal'
    mixup='cutmix'
    suffix="_cmx"
elif [ $data_aug == "mixup" ]; then
    ## Only mix-up train
    # augment=False
    train_transform='normal'
    mixup='mixup'
    suffix="_mx"
elif [ $data_aug == "augment_cutmix" ]; then
    ## augmentation and mix-up train
    # augment=True
    train_transform='random'
    mixup='cutmix'
    suffix="_cmag"
elif [ $data_aug == "augment_mixup" ]; then
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
echo "Data Preprocess: $data_aug :(Augment: $augment, Mixup: $mixup (prob. = $mixup_prob))"


if [ -n "$out_suffix" ]; then
    out_suffix="_$out_suffix"
fi

if [ $forget_class != "none" ]; then
    out_suffix="${out_suffix}_FGCLS-${forget_class}"
fi

if [ "$mixup_prob" != "1.0" ]; then
    out_suffix="${out_suffix}_mxp-${mixup_prob}"
fi

output_path="./outs/${dataset}_$num_samples${suffix}${out_suffix}/"
exp_name="logs/${dataset}_$num_samples${suffix}${out_suffix}" # "RobExp_$dataset"
echo "Output Path: $output_path"
echo "Exp Name: $exp_name"
echo "Model path: $model_path"


linear_layers="1"
nonlinear_layers="1,3,6" #"1,2"  #"1,3,6"  #"1,2" #"1,3,4,6,8,9"  # "1,3,6,7,12,15,17,19,22,33,49"  #
convs_mlplayers=(10 200)
simple_mlplayers=(50 100 200)
loss_mlplayers=(10 20 50)
# if [ "$num_samples" = 50000 ]; then
#     loss_mlplayers=(10 20 50 100)
# fi

attack_models=("lr" "svm" "rf" "mlp")  #
attack_features=("loss" "posterior" "linear" "nonlinear") # "entropy"  "entropy" # "nonlinear" # 
declare -A feature_layers=( ["entropy"]="1" ["loss"]="1" ["posterior"]="1" ["linear"]=$linear_layers ["nonlinear"]=$nonlinear_layers )


for attack_feature in "${attack_features[@]}"; do
    if [ "${attack_feature}" = "posterior" -o "${attack_feature}" = "loss" -o "${attack_feature}" = "entropy" ]; then
        posterior_opts=("true")
    else
        posterior_opts=("false") #("false" "true")
    fi
    # echo "$unlearn_method" "$attack_feature" "$posterior"
    IFS=',' read -r -a last_k_layers <<< "${feature_layers[${attack_feature}]}"
    # last_k_layers="${feature_layers[${attack_feature}]}"
    for K in "${last_k_layers[@]}"; do
        for posterior in "${posterior_opts[@]}"; do
            for attack_model in "${attack_models[@]}"; do
                if [ "$attack_model" = "mlp" ]; then
                    if [ "$attack_feature" = "nonlinear" ]; then
                        layers=("${convs_mlplayers[@]}")
                    elif [ "$attack_feature" = "loss" -o "$attack_feature" = "entropy" ]; then
                        layers=("${loss_mlplayers[@]}")
                    else
                        layers=("${simple_mlplayers[@]}")
                    fi
                    for layer in "${layers[@]}"; do
                        # echo "hidden layer: $layer"
                        # echo "$unlearn_method" "$attack_model" "$attack_feature" "$K" "$posterior" "$layer"
                        python $attacker  --out_dir=$output_path --exp_name=$exp_name --seed=$seed --cuda=$cuda_id \
                                        --dataset=$dataset --num_samples=$num_samples --valid_size=$valid_size \
                                        --num_to_forget=$unlearn_size --forget_classes=$forget_class \
                                        --arch=$arch --model_path=$model_path --attack_model=$attack_model \
                                        --attack_feature=$attack_feature --last_k=$K --hidden_layer_sizes=$layer \
                                        --test_ratio=$test_ratio --stacked=$stacked --include_posterior=$posterior \
                                        --batch_size=$batch_size --unlearn_method=$unlearn_method
                    done
                else
                    layer=-1
                    # echo "$unlearn_method" "$attack_model" "$attack_feature" "$K" "$posterior" "$model"
                    python $attacker  --out_dir=$output_path --exp_name=$exp_name --seed=$seed --cuda=$cuda_id \
                                        --dataset=$dataset --num_samples=$num_samples --valid_size=$valid_size \
                                        --num_to_forget=$unlearn_size --forget_classes=$forget_class \
                                        --arch=$arch --model_path=$model_path --attack_model=$attack_model \
                                        --attack_feature=$attack_feature --last_k=$K --hidden_layer_sizes=$layer \
                                        --test_ratio=$test_ratio --stacked=$stacked --include_posterior=$posterior \
                                        --batch_size=$batch_size --unlearn_method=$unlearn_method
                fi
            done
        done
    done
    echo ""
done
echo ""
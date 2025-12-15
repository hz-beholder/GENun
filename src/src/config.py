# Directory configuration

from const import SingletonString

# DATASET = "cifar10"    # "cifar10_mixup" #  "cifar10_augs" # "cifar10_augmix" # "mnist"
# ROOT_PATH = "outs_10k" #"robust_outs"   #"outs" #  #cifar10_scratch   #"debug_T"  #"outs"
# DATASET += single_str.str_
# ROOT_PATH = "/" + DATASET

single_str = SingletonString()
if not single_str.content.startswith("./"):
    ROOT_PATH = "./" + single_str.content
else:
    ROOT_PATH = single_str.content

DATA_PATH = "data/"
ORIGINAL_DATASET_PATH = DATA_PATH + "dataset/"
PROCESSED_DATASET_PATH = DATA_PATH + "processed_dataset/"

MODEL_PATH = ROOT_PATH + "/model_bases/"
CHECKPOINT_PATH = ROOT_PATH + "/checkpoints/"
ATTACK_MODEL_PATH = ROOT_PATH + "/attack_models/"
ATTACK_DATA_PATH = ROOT_PATH + "/attack_data/"
ATTACK_RESULT_PATH = ROOT_PATH + "/attack_results/"
EVAL_RESULT_PATH = ROOT_PATH + "/eval_results/"
# SPLIT_DATA_PATH = ROOT_PATH + "/split_index/"


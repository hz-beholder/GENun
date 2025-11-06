from enum import Enum

## Data tranform format
class PreprocTransformConfig:
    _instance = None  # Keep instance reference
    _augment = False  # Boolean attribute

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PreprocTransformConfig, cls).__new__(cls)
        return cls._instance

    @property
    def augment(self):
        return PreprocTransformConfig._augment

    @augment.setter
    def augment(self, value):
        if isinstance(value, bool):
            PreprocTransformConfig._augment = value
        else:
            raise ValueError("Augment must be a boolean value")

## Data tranform format
class SingletonString:
    _instance = None  # Keep instance reference
    _content = ""  # str attribute

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonString, cls).__new__(cls)
        return cls._instance

    @property
    def content(self):
        return SingletonString._content

    @content.setter
    def content(self, value):
        if isinstance(value, str):
            SingletonString._content = value
        else:
            raise ValueError("Augment must be a boolean value")


NUM_CLASSES_DICT = {
    "none": -1,
    "cifar10": 10,
    "cifar100": 100,
    "living17": 17,
    "entity30": 30,
    "stl10": 10,
    "domainnet-sketch": 40,
    "domainnet-art": 40,
    "domainnet-real": 40,
    "domainnet-painting": 40,
    "pairedCIFAR": 10,
    "blendedCIFAR": 10,
    "blendedcifar10": 10,
    "blendedstl10": 10,
}

## pre-trained model on ImageNet dataset
norm_dict = {
    "cifar10_mean": [0.485, 0.456, 0.406],
    "cifar10_std": [0.228, 0.224, 0.225],
    "cifar100_mean": [0.485, 0.456, 0.406],
    "cifar100_std": [0.228, 0.224, 0.225],
    "mnist_mean": [0.485, 0.456, 0.406],
    "mnist_std": [0.228, 0.224, 0.225],
    "living17_mean": [0.485, 0.456, 0.406],
    "living17_std": [0.228, 0.224, 0.225],
    "entity30_mean": [0.485, 0.456, 0.406],
    "entity30_std": [0.228, 0.224, 0.225],
    "pairedcifar10_mean": [0.485, 0.456, 0.406],
    "pairedcifar10_std": [0.228, 0.224, 0.225],
    "blendedcifar10_mean": [0.485, 0.456, 0.406],
    "blendedcifar10_std": [0.228, 0.224, 0.225],
    "stl10_mean": [0.485, 0.456, 0.406],
    "stl10_std": [0.228, 0.224, 0.225],
    "clip_mean": [0.48145466, 0.4578275, 0.40821073],
    "clip_std": [0.26862954, 0.26130258, 0.27577711],
    "domainnet_mean": [0.485, 0.456, 0.406],  # from domainnet py
    "domainnet_std": [0.228, 0.224, 0.225],
    "vits8-dino_mean": [0.485, 0.456, 0.406],
    "vits8-dino_std": [0.229, 0.224, 0.225]
}


CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

CBAR_CORRUPTIONS = [
    "blue_noise_sample",
    "brownish_noise",
    "checkerboard_cutout",
    "inverse_sparkles",
    "pinch_and_twirl",
    "ripple",
    "circular_motion_blur",
    "lines",
    "sparkles",
    "transverse_chromatic_abberation",
]

CBAR_CORRUPTIONS_SEV = {
    "caustic_refraction": [2.35, 3.2, 4.9, 6.6, 9.15],
    "inverse_sparkles": [1.0, 2.0, 4.0, 9.0, 10.0],
    "sparkles": [1.0, 2.0, 3.0, 5.0, 6.0],
    "perlin_noise": [4.6, 5.2, 5.8, 7.6, 8.8],
    "blue_noise_sample": [0.8, 1.6, 2.4, 4.0, 5.6],
    "plasma_noise": [4.75, 7.0, 8.5, 9.25, 10.0],
    "checkerboard_cutout": [2.0, 3.0, 4.0, 5.0, 6.0],
    "cocentric_sine_waves": [3.0, 5.0, 8.0, 9.0, 10.0],
    "single_frequency_greyscale": [1.0, 1.5, 2.0, 4.5, 5.0],
    "brownish_noise": [1.0, 2.0, 3.0, 4.0, 5.0],
}

# class ExtendedEnum(Enum):
#     @classmethod
#     def list(cls):
#         return list(map(lambda c: c.value, cls))

# class MODETYPE(Enum):
#     SHADOW_MODEL = "shadow"
#     TARGET_MODEL = "target"


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class DATATYPE(Enum):
    CATEGORY = "cat"
    IMAGE = "image"

class UNLEARN(Enum):
    NONE = "none"
    ORIGINAL = "original"
    SCRATCH = "scratch"
    FINETUNE = "finetune"
    RANDOM_LABEL = "random_label"
    NEG_GRAD = "negative_gradient"
    LAST_K_LAYER = "lastllayer"
    FISHER_APPROX = "fisherapprox"
    CERTIFIED_REMOVAL = "certified_removal"
    # SISA = "sisa"


class CONSTRUCTIONSELF(Enum):
    RAW = "raw"

class CONSTRUCTIONMETHOD(Enum):
    RAW = "raw"
    DIRECT_DIFFERENCE = "direct_difference"
    SORTED_DIFFERENCE = "sorted_difference"
    DIRECT_CONCAT = "direct_concat"
    SORTED_CONCAT = "sorted_concat"
    L2_DISTANCE = "l2_distance"

import torch
import random
import torchvision
from torchvision import transforms
from PIL import ImageFilter, Image

from src.datasets.potsdam import Potsdam
from src.datasets.lovaDa import LoveDA
from src.datasets.vaihingen import Vaihingen


from src.datasets.data_statistics import get_data_mean_and_stdev

DATASET = {
    'potsdam': Potsdam,
    'loveDa': LoveDA,
    'vaihingen': Vaihingen,
}


def zscore_image(img_tensor):
    img_tensor -= img_tensor.mean([-1, -2], keepdim=True)
    img_tensor /= img_tensor.std([-1, -2], keepdim=True)
    return img_tensor

def get_image_datasets(
        dataset_name,
        default_augmentations='none',
    ):
    load_transforms = TRANSFORMS[default_augmentations]
    train_transforms, test_transforms = load_transforms(
        dataset=dataset_name, 
    )
    if dataset_name == "potsdam":
        train_dataset = DATASET[dataset_name](
            train=True,
            image_transforms=train_transforms
        )
        val_dataset = None
    elif dataset_name == 'loveDa':
        train_dataset = DATASET[dataset_name](
            train=True,
            image_transforms=train_transforms
        )
        val_dataset = None
    elif dataset_name == 'vaihingen':
        train_dataset = DATASET[dataset_name](
            train=True,
            image_transforms=train_transforms
        )
        val_dataset = None
        print("数据集名称为vaihingen!")
    else:
        train_dataset = DATASET[dataset_name](
            train=True,
            image_transforms=train_transforms
        )
        val_dataset = DATASET[dataset_name](
            train=False,
            image_transforms=test_transforms,
        )
    return train_dataset, val_dataset


def load_image_transforms(dataset):
    if 'cifar' in dataset:
        train_transforms = transforms.ToTensor()
        test_transforms = transforms.ToTensor()
    elif 'potsdam' or 'loveDa' or 'vaihingen' in dataset:
        # train_transforms = transforms.ToTensor()
        train_transforms = transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=224),
            transforms.ToTensor(),
        ])
        test_transforms = None
    elif dataset in ['mscoco'] or 'meta_' in dataset:
        train_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
    else:
        return None, None

    return train_transforms, test_transforms


def load_default_transforms(dataset):
    if 'cifar' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                std=[0.247, 0.243, 0.261]),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                std=[0.247, 0.243, 0.261]),
        ])
    elif 'potsdam' in dataset:
        color_jitter = torchvision.transforms.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([color_jitter], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.ToTensor(),
            normalize])
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # train_transforms = torchvision.transforms.Compose([
        #     transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        #     transforms.RandomApply(
        #         [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        #     ),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomApply([loader.GaussianBlur([0.1, 2.0])], p=0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     normalize,
        # ])
        test_transforms = None
    elif dataset in ['mscoco'] or 'meta_' in dataset:
        mean, std = get_data_mean_and_stdev(dataset)
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return None, None
    
    return train_transforms, test_transforms


def load_default_unnorm_transforms(dataset, **kwargs):
    if 'cifar' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.ToTensor()
    elif dataset in ['mscoco'] or 'meta_' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
    else:
        return None, None

    return train_transforms, test_transforms


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


TRANSFORMS = {
    True: load_default_transforms,
    False: load_image_transforms,
    'all': load_default_transforms,
    'all_unnorm': load_default_unnorm_transforms,
    'none': load_image_transforms, 
}

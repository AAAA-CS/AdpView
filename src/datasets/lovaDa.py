import numpy as np
import glob
import os
import torch.utils.data as data
from torchvision import transforms, datasets
from PIL import Image
from src.datasets.root_paths import DATA_ROOTS


class LoveDA(data.Dataset):
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self,
            root=DATA_ROOTS['loveDa'],
            train=True,
            image_transforms=None,
    ):
        super().__init__()
        # if not os.path.isdir(root):
        #     os.makedirs(root)
        # self.dataset = datasets.cifar.CIFAR10(
        #     root,
        #     train=train,
        #     download=True,
        #     transform=image_transforms,
        # )
        # self.dataset = datasets.ImageFolder(root=root, transform=image_transforms)
        self.files = glob.glob(os.path.join(root, '*.tif'))
        self.transformer = image_transforms
        print("loveDa数据集transformer:....", self.transformer)

    def __getitem__(self, index):
        # pick random number
        neg_index = np.random.choice(np.arange(self.__len__()))
        # img_data, label = self.dataset.__getitem__(index)
        # img2_data, _ = self.dataset.__getitem__(index)
        # neg_data, _ = self.dataset.__getitem__(neg_index)

        img = Image.open(self.files[index])
        # print("img............................,,,,,,,,,.")
        # print(np.array(img))        #正常值[0, 255]
        # img = self.transformer(img)
        # img_data = img
        # img2_data = img
        img_data = self.transformer(img)
        img2_data = self.transformer(img)

        neg_data = Image.open(self.files[neg_index])
        neg_data = self.transformer(neg_data)

        # build this wrapper such that we can return index
        data = [index, img_data.float(), img2_data.float(), neg_data.float(), 0]
        return tuple(data)

    def __len__(self):
        # return len(self.dataset)
        return len(self.files)

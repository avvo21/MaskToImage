import torch
import numpy as np
import random
from torchvision.datasets import Cityscapes
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from utils import encode_mask

class CityscapesPair(Cityscapes):
    def __init__(self, root, split, mode, target_type, augment=False):
        super().__init__(root, split, mode, target_type)
        self.augment = augment

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)
        mask = encode_mask(mask)

        image = TF.resize(image, [256, 512], interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [256, 512], interpolation=transforms.InterpolationMode.NEAREST)

        if self.augment:
            # Horizontal Flip on image and mask
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        image = TF.to_tensor(image) # Convert PIL image to PyTorch tensor in CxHxW format with values in [0, 1]
        image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Resulting [-1,1]

        mask = torch.from_numpy(np.array(mask)).long()

        return mask, image


class CityscapesDataManager:
    def __init__(self, args):
        self.args = args

    def get_loaders(self):

        trainset = CityscapesPair(
            root=self.args.dataset_path,
            split='train',
            mode='fine',
            target_type='semantic',
            augment=self.args.augment  # Activate only if passed from args
        )

        valset = CityscapesPair(
            root=self.args.dataset_path,
            split='val',
            mode='fine',
            target_type='semantic',
            augment=False  # Validation is always false
        )

        trainloader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            drop_last=True
        )

        valloader = DataLoader(
            valset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers
        )

        return trainloader, valloader
# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as tdata
from load_dataset.autoaugment import ImageNetPolicy
from load_dataset.random_erasing import RandomErasing

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           subset_size=1,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False,
                           autoaugment=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg1 = "[!] valid_size should be in the range [0, 1]."
    error_msg2 = "[!] subset_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg1
    assert ((subset_size >= 0) and (valid_size <= 1)), error_msg2

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224,224)),
            transforms.RandomHorizontalFlip(),
        ])
        if autoaugment:
            train_transform.transforms.append(ImageNetPolicy())
        train_transform.transforms.append(transforms.ToTensor())
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    train_transform.transforms.append(normalize)

    # load the dataset
    train_dataset = datasets.ImageNet(
        root=data_dir, split='train',
        transform=train_transform,
    )

    valid_dataset = datasets.ImageNet(
        root=data_dir, split='train',
        transform=valid_transform,
    )

    num_train = len(train_dataset)
    # indices_all = list(range(num_train))
    split_subset = int(np.floor(subset_size * num_train))
    indices_subset = list(range(split_subset))
    split_valid = int(np.floor(valid_size * split_subset))

    if shuffle:
        #np.random.seed(random_seed)
        np.random.shuffle(indices_subset)

    train_idx, valid_idx = indices_subset[split_valid:], indices_subset[:split_valid]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = tdata.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = tdata.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)

def get_train_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False,
                           autoaugment=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    # define transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
        ])
        if autoaugment:
            train_transform.transforms.append(ImageNetPolicy())
        train_transform.transforms.append(color_transform)
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        train_transform.transforms.append(RandomErasing(0.2, mode='pixel', max_count=1, num_splits=False, device='cpu'))
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

    # load the dataset
    train_dataset = datasets.ImageNet(
        root=data_dir, split='train',
        transform=train_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))

    if shuffle:
        #np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx = indices
    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = tdata.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader

def get_test_loader(data_dir,
                    batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageNet(
        root=data_dir, split='val',
        transform=transform,
    )

    data_loader = tdata.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


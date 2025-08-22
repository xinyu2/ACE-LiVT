# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image

class CIFAR10_LT(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self,
                 root,
                 imb_type='exp',
                 imb_factor=0.01,
                 rand_number=0,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(CIFAR10_LT, self).__init__(root, train, transform,
                                               target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        if train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type,imb_factor)
            self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([
                the_class,
            ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class CIFAR100_LT(CIFAR10_LT):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

class DatasetLT(datasets.ImageFolder):

    def get_cls_num(self):
        cls_num = [0] * len(self.classes)
        for img in self.imgs:
            cls_num[img[1]] += 1
        return cls_num

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if args.dataset == 'cifar10-LT':
        dataset = CIFAR10_LT(root=args.data_path,
                            imb_type='exp',
                            imb_factor=1/args.imbf,
                            rand_number=0,
                            train=is_train,
                            download=True,
                            transform=transform)
    elif args.dataset == 'cifar100-LT':
        dataset = CIFAR100_LT(root=args.data_path,
                            imb_type='exp',
                            imb_factor=1/args.imbf,
                            rand_number=0,
                            train=is_train,
                            download=True,
                            transform=transform)
    else:
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = DatasetLT(root, transform=transform)
    print(dataset)
    return dataset

def build_dataset2(is_train, mask_root, args):
    transform = build_transform(is_train, args)
    if args.dataset == 'cifar10-LT':
        dataset = CIFAR10_LT(root=args.data_path,
                            imb_type='exp',
                            imb_factor=1/args.imbf,
                            rand_number=0,
                            train=is_train,
                            download=True,
                            transform=transform)
    elif args.dataset == 'cifar100-LT':
        dataset = CIFAR100_LT(root=args.data_path,
                            imb_type='exp',
                            imb_factor=1/args.imbf,
                            rand_number=0,
                            train=is_train,
                            download=True,
                            transform=transform)
    else:
        root = os.path.join(mask_root, 'train')
        dataset = DatasetLT(root, transform=transform)
    print(dataset)
    return dataset

def get_mean_std(args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if args.dataset == 'ImageNet-LT':
        mean = [0.479672, 0.457713, 0.407721]
        std = [0.278976, 0.271203, 0.286062]
    elif args.dataset == 'iNat18':
        mean = [0.466, 0.471, 0.380]
        std = [0.195, 0.194, 0.192]
    elif args.dataset == 'ImageNet-BAL':
        mean = [0.480767, 0.457071, 0.407718]
        std = [0.279940, 0.272481, 0.286038]
    elif args.dataset == 'ImageNet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.dataset == 'Place':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else: # Cifar 10 or 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    return mean, std


def build_transform(is_train, args):

    mean, std = get_mean_std(args)
    
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)
    # to maintain same ratio w.r.t. 224 images
    test_transform = transforms.Compose([
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return test_transform


class combine_Dataset(Dataset):

    def __init__(self, ds1, ds2):
        self.img_path = []
        self.labels = []
        self.transform = ds1.transform
        for img in ds1.imgs:
            p = img[0]
            if '._' in p:
                p = ''.join(p.split('._'))
            self.img_path.append(p)

        for img in ds2.imgs:
            p = img[0]
            if '._' in p:
                p = ''.join(p.split('._'))
            self.img_path.append(p)    

        self.labels = ds1.targets + ds2.targets
        cls_num = [0] * len(ds1.classes)
        for img in ds1.imgs:
            cls_num[img[1]] += 1
        self.cls_num = cls_num
    
    def get_cls_num(self):
        return self.cls_num

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

class combine_Datasets(Dataset):

    def __init__(self, ds1, ds2):
        self.img_path = []
        self.labels = []
        self.transform = [None, ds1.transform, ds2.transform]
        for img in ds1.imgs:
            p = img[0]
            if '._' in p:
                p = ''.join(p.split('._'))
            self.img_path.append((1, p))

        self.labels = ds1.targets

        cls_num = [0] * len(ds1.classes)
        for img in ds1.imgs:
            cls_num[img[1]] += 1
        self.cls_num = cls_num

        for idx, img in enumerate(ds2.imgs):
            p, lbl = img[0], img[1]
            if '._' in p:
                p = ''.join(p.split('._'))
            self.img_path.append((2,p))
            self.labels.append(lbl)
        
    
    def get_cls_num(self):
        return self.cls_num

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        (ds_idx, path) = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform[ds_idx](sample)

        return sample, label

class combine_Dataset_mf(Dataset):

    def __init__(self, ds1, ds2):
        self.img_path = []
        self.labels = []
        self.transform = ds1.transform
        for img in ds1.imgs:
            p = img[0]
            if '._' in p:
                p = ''.join(p.split('._'))
            self.img_path.append(p)

        self.labels = ds1.targets

        cls_num = [0] * len(ds1.classes)
        for img in ds1.imgs:
            cls_num[img[1]] += 1
        self.cls_num = cls_num

        tmp_cls_num = np.array(cls_num)
        many_shot = tmp_cls_num > 100
        medium_shot = (tmp_cls_num <= 100) & (tmp_cls_num > 20)
        few_shot = tmp_cls_num <= 20

        for idx, img in enumerate(ds2.imgs):
            p, lbl = img[0], img[1]
            if '._' in p:
                p = ''.join(p.split('._'))
            if medium_shot[lbl] | few_shot[lbl]:
                self.img_path.append(p)
                self.labels.append(lbl)
        
    
    def get_cls_num(self):
        return self.cls_num

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

if __name__ == '__main__':
    dataset_train = DatasetLT(os.path.join('/diskC/xzz/ImageNet-LT', 'train'), transform=None)
    cls_num = dataset_train.get_cls_num()
    print(cls_num)

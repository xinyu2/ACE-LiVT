import numpy as np
import os
from tqdm import tqdm
import json
import shutil
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet-LT Training')
parser.add_argument('--data', type=str, default='ImageNet') # iNaturalist
parser.add_argument('--dist', type=str, default='LT') #BAL
parser.add_argument('--task', type=str, default='train') #val
parser.add_argument('--work', type=str, default='count') #link
args = parser.parse_args()

PREFIX = '/data/lab/yan/xinyu/data'
PREFIX2 = '/data/lab/yan/xinyu/DATA'
def create_iNat18_old(task='train'):
    src_root = '/path/to/iNat18'
    dst_root = '/diskC/xzz/iNat18'

    if not os.path.exists(os.path.join(dst_root, task, '0')):
        for i in range(8142):
            pth = os.path.join(dst_root, task, str(i))
            os.makedirs(pth, exist_ok=True)

    with open(os.path.join(src_root, f'iNat18_{task}.json')) as f:
        meta = json.load(f)
        for anno in tqdm(meta['annotations']):
            img_path = anno['fpath'].replace('./downloaed/iNat18/','')
            img_name = anno['fpath'].split('/')[-1]
            cls_name = str(anno['category_id'])
            src_pth = os.path.join(src_root, img_path)
            dst_pth = os.path.join(dst_root, task, cls_name, img_name)
            shutil.copy(src_pth, dst_pth)

def create_iNat18(task='train'):
    src_root = f'{PREFIX}/iNaturalist18'
    dst_root = F'{PREFIX}/iNat18'

    if not os.path.exists(os.path.join(dst_root, task, '0')):
        for i in range(8142):
            pth = os.path.join(dst_root, task, str(i))
            os.makedirs(pth, exist_ok=True)

    with open(os.path.join(src_root, f'iNaturalist18_{task}.txt'), 'r') as f:
        num_lines = sum(1 for line in f)

    with open(os.path.join(src_root, f'iNaturalist18_{task}.txt')) as f:
        for line in tqdm(f, total=num_lines):
            img_path = line.split()[0]
            cls_name = line.split()[1]
            img_name = img_path.split('/')[-1]
            src_pth = os.path.join(src_root, img_path)
            dst_pth = os.path.join(dst_root, task, cls_name, img_name)
            # shutil.copy(src_pth, dst_pth)
            os.symlink(src_pth, dst_pth)

def create_ImageNet_BAL_old(task='train'):
    src_root = f'{PREFIX}/ImageNet_LT/{task}'
    dst_root = f'{PREFIX}/ImageNet_BAL/{task}'
    img_cnt = 160
    
    folds = os.listdir(src_root)
    for fold in tqdm(folds):
        fold_src = os.path.join(src_root, fold)
        fold_dst = os.path.join(dst_root, fold)
        os.makedirs(fold_dst, exist_ok=True)
        imgs = os.listdir(fold_src)
        for img in imgs[:img_cnt]:
            src_pth = os.path.join(fold_src, img)
            dst_pth = os.path.join(fold_dst, img)
            shutil.copy(src_pth, dst_pth)

def create_ImageNet_BAL(task='train'):
    src_root = f'{PREFIX}/ImageNet_LT/{task}'
    dst_root = f'{PREFIX}/ImageNet_BAL/{task}'
    img_cnt = 160
    
    folds = os.listdir(src_root)
    for fold in tqdm(folds):
        fold_src = os.path.join(src_root, fold)
        fold_dst = os.path.join(dst_root, fold)
        os.makedirs(fold_dst, exist_ok=True)
        imgs = os.listdir(fold_src)
        for img in imgs[:img_cnt]:
            src_pth = os.path.join(fold_src, img)
            dst_pth = os.path.join(fold_dst, img)
            # shutil.copy(src_pth, dst_pth)
            os.symlink(src_pth, dst_pth)

def create_ImageNet_LT(task='train'):
    src_text_root = f'{PREFIX}/ImageNet_LT'
    src_root = '/data/lab/tao/imagenet/'
    dst_root = f'{PREFIX2}/ImageNet-LT'
    img_cnt = 160
    
    folds = os.listdir(os.path.join(src_root, 'train'))
    for fold in tqdm(folds):
        fold_dst = os.path.join(dst_root, task, fold)
        os.makedirs(fold_dst, exist_ok=True)

    with open(os.path.join(src_text_root, f'ImageNet_LT_{task}.txt'), 'r') as f:
        num_lines = sum(1 for line in f)
 
    with open(os.path.join(src_text_root, f'ImageNet_LT_{task}.txt')) as f:
        for line in tqdm(f, total=num_lines):
            img_path = line.split()[0]
            img_split = img_path.split('/')[0]
            img_fold = img_path.split('/')[1]
            img_name = img_path.split('/')[-1]
            src_pth = os.path.join(src_root, img_split, img_fold, img_name)
            dst_pth = os.path.join(dst_root, task, img_fold, img_name)
            os.symlink(src_pth, dst_pth)

def count_ImageNet_BAL(task='train'):
    src_root = f'{PREFIX}/ImageNet_LT/{task}'
    
    total = 0
    folds = os.listdir(src_root)
    for fold in tqdm(folds):
        curdir = os.path.join(src_root, fold)
        samples = os.listdir(curdir)
        total += len(samples)
    print(f"total samples = {total}")    

if __name__ == '__main__':
    if args.data == 'ImageNet':
        if args.work == 'link':
            if args.dist == 'BAL':
                create_ImageNet_BAL(task=args.task)
            else:
                create_ImageNet_LT(task=args.task)
        else:
            count_ImageNet_BAL(task=args.task)
    if args.data == 'iNaturalist':
        if args.work == 'link':
            create_iNat18(task=args.task)
        else:
            print(f"count iNat")
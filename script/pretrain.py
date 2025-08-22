import os, sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer

def pretrain():
    T = Trainer()
    T.task = 'pretrain'
    T.note = f'mae_vit_ltlt'
    T.batch = 512
    T.epochs = 800
    T.warmup_epochs = 40
    T.device = '0,1'
    T.input_size = 224
    T.accum_iter = 4
    T.device = '0,1'
    T.dataset = 'ImageNet-LT'
    T.model = f'mae_vit_base_patch16'
    T.mask_ratio = 0.75
    T.blr = 1.5e-4
    T.weight_decay = 0.05
    T.num_workers = 16
    T.resume = ''
    # T.resume = '/data/lab/yan/xinyu/ACE-LiVT/ckpt/pretrain/ImageNet-LT/mae_vit_base/checkpoint.pth'
    T.pretrain()

pretrain()
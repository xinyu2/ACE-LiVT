import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH

PREFIX = '/data/lab/yan/xinyu/'
finetune_model = 'vit_base_patch16'
note = 'lt0818_vit_b224'

def eval_model():
    T = Trainer()
    T.task = 'evaluate'
    T.dataset = 'ImageNet-LT'
    T.note = note
    # T.resume = f'{PREFIX}/ACE-LiVT/ckpt/git-result/checkpoint.pth'
    # T.resume = f'{PREFIX}/ACE-LiVT/ckpt/finetune/ImageNet-LT/vit_base_patch16/vit_b224_0721_full/checkpoint.pth'
    T.resume = f'{PREFIX}/ACE-LiVT/ckpt/finetune/ImageNet-LT/vit_base_patch16/vit_b224_0818_ltlt/checkpoint.pth'
    T.batch = 128
    T.device = '0'
    T.model = finetune_model
    T.input_size = 224
    T.global_pool = True
    T.num_workers = 16
    T.master_port = 29600
    T.evaluate()


eval_model()
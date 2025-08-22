import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH

PREFIX = '/data/lab/yan/xinyu/'
# pretrain_model = 'mae_vit_base'
pretrain_model = 'mae_vit_ltlt'

def run_imagenetlt():
    T = Trainer()
    T.task = 'finetune'
    T.note = 'vit_b224_0818_ltlt'
    T.ckpt = f'{PREFIX}/ACE-LiVT/ckpt/pretrain/ImageNet-LT/{pretrain_model}/checkpoint.pth'
    

    T.dataset = 'ImageNet-LT'
    T.nb_classes = 1000

    T.epochs = 100
    T.device = '0,1'

    T.batch = 128
    T.accum_iter = 4

    T.model = 'vit_base_patch16' #'vit_base_patch16'
    T.input_size = 224
    T.drop_path = 0.1

    # T.resume = f'{PREFIX}/ACE-LiVT/ckpt/finetune/ImageNet-LT/{T.model}/{T.note}/checkpoint.pth'

    T.clip_grad = None
    T.weight_decay = 0.05
    T.adamW2 = 0.99

    T.lr = None
    T.blr = 1e-3
    T.layer_decay = 0.75
    T.min_lr = 1e-6
    T.warmup_epochs = 10

    T.color_jitter = None
    T.aa = 'rand-m9-mstd0.5-inc1'

    T.reprob = 0.25
    T.remode = 'pixel'
    T.recount = 1
    T.resplit = False

    T.mixup = 0.8
    T.cutmix = 1.0
    T.cutmix_minmax = None
    T.mixup_prob = 1.0
    T.mixup_switch_prob = 0.5
    T.mixup_mode = 'batch'

    T.loss = 'Bal_BCE'
    T.bal_tau = 1.0
    T.smoothing = 0.1

    T.global_pool = True

    T.seed = 0
    T.prit = 20

    T.num_workers = 16
    T.master_port = 29500

    T.finetune()

run_imagenetlt()


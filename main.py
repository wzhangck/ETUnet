import argparse
from utils.data_utils import get_loader
from medical.trainer import Trainer, Validator
from monai.inferers import SlidingWindowInferer
import torch
import torch.nn.parallel
import torch.utils.data.distributed
import numpy as np
from monai.metrics import DiceMetric,HausdorffDistanceMetric
from monai.utils.enums import MetricReduction
from medical.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses.dice import DiceLoss
from medical.model.ETUNetModel import ETUNet
import random

parser = argparse.ArgumentParser(description='ETUNet segmentation pipeline for BRATS Challenge')
parser.add_argument('--model_name', default="ETUNet", help='the model will be trained')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--logdir', default=None, type=str, help='directory to save the tensorboard logs1_32_64')
parser.add_argument('--fold', default=None, type=int, help='data fold')
parser.add_argument('--pretrain_model_path', default='./model.pt', type=str, help='pretrained model name')
parser.add_argument('--load_pretrain', action="store_true", help='pretrained model name')
parser.add_argument('--data_dir', default=None, type=str, help='dataset directory')
parser.add_argument('--json_list', default=None, type=str, help='dataset json file')
parser.add_argument('--max_epochs', default=500, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
parser.add_argument('--sw_batch_size', default=1, type=int, help='number of sliding window batch size')
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--val_every', default=10, type=int, help='validation frequency')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization name')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--feature_size', default=24, type=int, help='feature size')
parser.add_argument('--in_channels', default=4, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=3, type=int, help='number of output channels')
parser.add_argument('--cache_dataset', action='store_true', help='use monai Dataset class')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.0, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.0, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
parser.add_argument('--dropout_path_rate', default=0.1, type=float, help='drop path rate')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--infer_overlap', default=0.25, type=float, help='sliding window inference overlap')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=30, type=int, help='number of warmup epochs')
parser.add_argument('--resume_ckpt', action='store_true', help='resume training from pretrained checkpoint')


def post_pred_func(pred):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    return pred


def setup_seed(seed): # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    args = parser.parse_args()
    args.logdir = './logs/' + args.logdir
    main_worker(args=args)

def main_worker(args):
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    setup_seed(2023)
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    args.device=device
    args.test_mode = False
    train_loader, val_loader = get_loader(args)
    if args.rank == 0:
        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)

    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    model = ETUNet(model_num=args.in_channels,
                   out_channels=args.out_channels,
                   image_size=inf_size,
                   fea=(8, 8, 32, 64, 128, 8),
                   window_size=(2, 2, 2),
                   self_num_layer=4,
                   token_mixer_size=128,
                   )

    window_infer = SlidingWindowInferer(roi_size=inf_size,
                                        sw_batch_size=args.sw_batch_size,
                                        overlap=args.infer_overlap,
                                       )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params/1e6)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.','')] = v
        model.load_state_dict(new_state_dict, strict=False)

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.to(device)

    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight
                                     )

    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.optim_lr,
                                      weight_decay=args.reg_weight
                                      )
    elif args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.optim_lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.reg_weight
                                    )
    else:
        raise ValueError('Unsupported Optimization Procedure: ' + str(args.optim_name))

    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.max_epochs)
    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.max_epochs)
    else:
        scheduler = None

    dice_metric = DiceMetric(include_background=True,
                             reduction=MetricReduction.MEAN_BATCH,
                             get_not_nans=True)

    hd95_metric = HausdorffDistanceMetric(include_background=True,
                                          reduction=MetricReduction.MEAN_BATCH,
                                          get_not_nans=True,
                                          percentile=95)
    # validator = Validator(args,
    #                       model,
    #                       val_loader,
    #                       class_list=("TC", "WT", "ET"),
    #                       metric_functions=[["dice", dice_metric],['hd95',hd95_metric]],
    #                       sliding_window_infer=window_infer,
    #                       post_label=None,
    #                       post_pred=post_pred_func
    #                       )

    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)

    trainer = Trainer(args,
                      train_loader,
                      validator=None,
                      loss_func=dice_loss,
                      )

    best_acc = trainer.train(model,
                             optimizer=optimizer,
                             scheduler=scheduler,
                             )

    return best_acc

if __name__ == '__main__':
    main()

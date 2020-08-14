import argparse
import os
import sys
import time
import uuid
from collections import deque
from typing import Optional
from TF_logger import Logger
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from dataset.AVA_video_v2 import AVA_video
from backbone.base import Base as BackboneBase
from config.train_config import TrainConfig as Config
from dataset.base import Base as DatasetBase
from extention.lr_scheduler import WarmUpMultiStepLR
from logger import Logger as Log
from model import Model

################################################   TSM ###########################
import argparse
from mmcv import Config as Config1
from mmaction import __version__
from mmaction.datasets import get_trimmed_dataset
from mmaction.apis import (train_network, init_dist, get_root_logger,
                           set_random_seed)
from mmaction.models import build_recognizer

from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

###############################################  TSM ################################
os.environ['DISPLAY'] = 'localhost:12.0'
log_file = str("TSM_trainloss:") + 'loss-remove.txt'

def tourch_script():
    weights ='weights/slowfast_weight.pth'  #   '/home/ganhaiyang/output/ava/temp_4/model_save/2019-12-26-11-06-33/model-80.pth'
    backbone_name = Config.BACKBONE_NAME
    dataset=AVA_video(Config.TRAIN_DATA)
    backbone = BackboneBase.from_name(backbone_name)()
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    chkpt = torch.load(weights, map_location=device)
    model = Model(
        backbone, dataset.num_classes(), pooler_mode=Config.POOLER_MODE,
        anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
        rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N,
        anchor_smooth_l1_loss_beta=Config.ANCHOR_SMOOTH_L1_LOSS_BETA,
        proposal_smooth_l1_loss_beta=Config.PROPOSAL_SMOOTH_L1_LOSS_BETA
    ).to(device)

    try:
        model_dict=model.module.state_dict()
    except AttributeError:
        model_dict = model.state_dict()  # 读取参数，
    # 将pretrained_dict里不属于model_dict的键剔除掉
        chkpt = {k: v for k, v in chkpt.items() if k in model_dict}
    print("load pretrain model")
    model_dict.update(chkpt)
    model.load_state_dict(model_dict)

    # z转换为评估模型
    model.eval()
    # 向模型中输入数据以得到模型参数
    e1 = torch.rand(1, 3, 64, 300, 400).cuda()
    e2 = torch.rand(1, 3, 4).cuda()

    traced_script_module = torch.jit.trace(model,(e1,e2))
    traced_script_module.save("slowfast_50_eval_three.pt")
    print("out put save")
    exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Train for TSM')
    parser.add_argument('--config',default='config_files/sthv2/tsm_baseline.py', help = 'Train config file path')
    args = parser.parse_args()
    return args


def _train( backbone_name,path_to_checkpoints_dir, path_to_resuming_checkpoint):              # backbone_name,
    args = parse_args()
    cfg = Config1.fromfile(args.config)
    # logger = Logger('./logs')
    dataset=AVA_video(Config.TRAIN_DATA)
    dataloader = DataLoader(dataset, batch_size=4,num_workers=8, collate_fn=DatasetBase.padding_collate_fn,pin_memory=True,shuffle=False)   # batch_size=4,num_workers=8,shuffle=True，
    Log.i('Found {:d} samples'.format(len(dataset)))
    backbone = BackboneBase.from_name(backbone_name)()

    # backbone1 = build_recognizer(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    os.environ['CUDA_VISIBLE_DEVICES'] = Config.GPU_OPTION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")              # :0
    model = Model(
        backbone,
        dataset.num_classes(),
        pooler_mode=Config.POOLER_MODE,
        anchor_ratios=Config.ANCHOR_RATIOS,
        anchor_sizes=Config.ANCHOR_SIZES,
        rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N,
        rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N,
        anchor_smooth_l1_loss_beta=Config.ANCHOR_SMOOTH_L1_LOSS_BETA,
        proposal_smooth_l1_loss_beta=Config.PROPOSAL_SMOOTH_L1_LOSS_BETA
    ).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])  # multi-Gpu
    model.to(device)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    ################################################## add #####################
    # optimizer_MultiStepLR = torch.optim.SGD(net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[500, 10000, 90000, 180000], gamma=0.1)
    # scheduler = WarmUpMultiStepLR(optimizer, milestones=Config.STEP_LR_SIZES, gamma=Config.STEP_LR_GAMMA,
    #                               factor=Config.WARM_UP_FACTOR, num_iters=Config.WARM_UP_NUM_ITERS)
    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)       #类似list，限制长度的deque增加超过限制数的项时，另一边的项会自动删除。
    mean_losses = deque(maxlen=100)

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(path_to_checkpoints_dir, 'summaries','logdir',cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    summary_writer = SummaryWriter(logdir)    #    summary_writer = SummaryWriter(os.path.join(path_to_checkpoints_dir, 'summaries'))
    should_stop = False

    num_steps_to_display = Config.NUM_STEPS_TO_DISPLAY
    num_steps_to_snapshot = Config.NUM_STEPS_TO_SNAPSHOT
    num_steps_to_finish = Config.NUM_STEPS_TO_FINISH

    if path_to_resuming_checkpoint is not None:
        step = model.module.load(path_to_resuming_checkpoint, optimizer, scheduler)
        print("load from:",path_to_resuming_checkpoint)
    device_count = torch.cuda.device_count()
    assert Config.BATCH_SIZE % device_count == 0, 'The batch size is not divisible by the device count'
    Log.i('Start training with {:d} GPUs ({:d} batches per GPU)'.format(torch.cuda.device_count(), Config.BATCH_SIZE // torch.cuda.device_count()))

    print("loading data ... ")
    while not should_stop:
        for n_iter, (_, image_batch, _, bboxes_batch, labels_batch,detector_bboxes_batch) in enumerate(dataloader):
            batch_size = image_batch.shape[0]
            image_batch = image_batch.cuda()
            bboxes_batch = bboxes_batch.cuda()
            labels_batch = labels_batch.cuda()
            detector_bboxes_batch=detector_bboxes_batch.cuda()

            proposal_class_losses = \
                model.train().forward(image_batch, bboxes_batch, labels_batch,detector_bboxes_batch)     #eval().
            proposal_class_loss = proposal_class_losses.mean()
            loss = proposal_class_loss
            mean_loss=proposal_class_losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            mean_losses.append(mean_loss.item())
            summary_writer.add_scalar('train/proposal_class_loss', proposal_class_loss.item(), step)
            summary_writer.add_scalar('train/loss', loss.item(), step)

            if n_iter % 10000 == 0:
                for name, param in model.named_parameters():
                    name = name.replace('.', '/')
                    if name.find("conv") >= 0:
                        summary_writer.add_histogram(name, param.data.cpu().numpy(), global_step=n_iter)
                        summary_writer.add_histogram(name + 'grad', param.grad.data.cpu().numpy(),global_step=n_iter)
            step += 1
            if step == num_steps_to_finish:                  #222670
                should_stop = True
            if step % num_steps_to_display == 0:          #20
                elapsed_time = time.time() - time_checkpoint
                print("time_checkpoint :",time_checkpoint,"elapsed_time：",elapsed_time)
                time_checkpoint = time.time()
                steps_per_sec = num_steps_to_display / elapsed_time
                samples_per_sec = batch_size * steps_per_sec
                eta = (num_steps_to_finish - step) / steps_per_sec / 3600
                avg_loss = sum(losses) / len(losses)
                avg_mean_loss=sum(mean_losses) / len(mean_losses)
                lr = scheduler.get_lr()[0]
                print_string='[Step {0}] Avg. Loss = {avg_loss:.6f}, Learning Rate = {lr:.8f} ({samples_per_sec:.2f} samples/sec; ETA {eta:.1f} hrs)'\
                .format(step,avg_loss=avg_loss,lr=lr,samples_per_sec=samples_per_sec,eta=eta)
                print(print_string)
                with open(log_file, 'a') as f:
                    f.writelines(print_string + '\n')

            model_save_dir = os.path.join(path_to_checkpoints_dir, 'model_save',cur_time)
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            if step % num_steps_to_snapshot == 0 or should_stop:      #20000
                path_to_checkpoint = model.module.save(model_save_dir, step, optimizer, scheduler)    #model.save
                Log.i('Model has been saved to {}'.format(path_to_checkpoint))

            if should_stop:
                break
    Log.i('Done')


if __name__ == '__main__':
    def main():
        # transform torchscript model:
        # tourch_script()
        # exit(0)

        backbone_name = Config.BACKBONE_NAME
        path_to_outputs_dir = Config.PATH_TO_OUTPUTS_DIR
        path_to_resuming_checkpoint =Config.PATH_TO_RESUMEING_CHECKPOINT
        if not os.path.exists(path_to_outputs_dir):
            os.mkdir(path_to_outputs_dir)
        Log.initialize(os.path.join(path_to_outputs_dir, 'train.log'))
        Log.i('Arguments:')
        _train(backbone_name,path_to_outputs_dir, path_to_resuming_checkpoint)




    main()

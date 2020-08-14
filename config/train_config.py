import ast
from typing import List, Tuple

from config.config import Config


class TrainConfig(Config):

    RPN_PRE_NMS_TOP_N= 12000
    RPN_POST_NMS_TOP_N = 2000
    ANCHOR_SMOOTH_L1_LOSS_BETA = 1.0
    PROPOSAL_SMOOTH_L1_LOSS_BETA = 1.0

    BATCH_SIZE=2   # defaul
    LEARNING_RATE = 0.0001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    STEP_LR_SIZES = [90000,180000]
    STEP_LR_GAMMA = 0.1
    WARM_UP_FACTOR = 0.3333
    WARM_UP_NUM_ITERS = 500
    NUM_STEPS_TO_DISPLAY = 20
    NUM_STEPS_TO_SNAPSHOT = 2000   #20000
    NUM_STEPS_TO_FINISH = 222670
    GPU_OPTION = '2'
    TRAIN_DATA='/home/gan/data/video_caption_database/video_database/ava/preproc_train/backup/ava_train_removebadlist_v2.2.csv'

    PATH_TO_RESUMEING_CHECKPOINT = None              # '/home/ganhaiyang/dataset/ava/ava_weights/slowfast_weight.pth'
    PATH_TO_OUTPUTS_DIR = '/home/gan/home/ganhaiyang/Alg_Proj/Recog_Proj/TSM-Detection/output'




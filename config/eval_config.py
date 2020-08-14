from typing import List, Tuple
from config.config import Config


class EvalConfig(Config):

    RPN_PRE_NMS_TOP_N = 6000
    RPN_POST_NMS_TOP_N = 300
    VAL_DATA='/home/ganhaiyang/output/ava/ava_val_removebadlist_v2.2.csv'   #ava_train_v2.2_sub_5.txt
    PATH_TO_CHECKPOINT='/home/ganhaiyang/dataset/ava/ava_weights/slowfast_weight.pth'     #'/home/ganhaiyang/output/ava/temp_4/model_save/2019-12-10-19-37-43/model-20000.pth'
    PATH_TO_RESULTS='result-slowfast-rm.txt'
    PATH_TO_EXCLUSIONS='ava_val_excluded_timestamps_v2.2.csv'
    # PATH_TO_ACTION_LIST='ava_action_list_v2.2.pbtxt'
    PATH_TO_ACTION_LIST='ava_action_list_v2.2_for_activitynet_2019.pbtxt'
    # PATH_TO_LABLE='/home/ganhaiyang/dataset/ava/preproc_fallDown/ava_v1.0_extend_annot.csv'   #'ava_train_v2.2_sub_5.txt'
    PATH_TO_LABLE='ava_val_removebadlist_v2.2.csv'   #'ava_train_v2.2_sub_5.txt'

    KEEP=0.05


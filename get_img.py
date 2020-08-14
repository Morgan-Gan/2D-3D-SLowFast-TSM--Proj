# Author Morven Gan
# The road to success is not crowded, because there are not many persistent people

from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image

import argparse
from deep_sort import DeepSort
from collections import deque
from backbone.base import Base as BackboneBase
from config.train_config import TrainConfig
from config.eval_config import EvalConfig
from config.config import Config
from model import Model
import os
import numpy

videofile = "/home/ganhaiyang/dataset/ava/v_WalkingWithDog_g08_c03.avi"
cap = cv2.VideoCapture(videofile)
assert cap.isOpened(), 'Cannot capture source'

frames = 0
##########################################################
last = np.array([])
##########################################################
#######for sp detec##########
buffer = deque(maxlen=64)
resize_width = 400
resize_height = 300

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('/home/ganhaiyang/dataset/ava/test/%d.jpg' % count, frame)
        count+=1
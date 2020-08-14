from typing import Tuple

import torchvision
from torch import nn

# import backbone.base
# from backbone.slowfastnet import tsm as tsm1
# from backbone.hidden_for_roi_maxpool import hidden50
# class tsm(backbone.base.Base):
#
#     def __init__(self):
#         super().__init__(False)
#
#     def features(self):
#         print("slowfast_res50")
#         tsmResnet101 = tsm1()
#         hidden = hidden50()
#         num_features_out = 2304
#         num_hidden_out = 2304*3*3
#
#         return tsmResnet101, hidden, num_features_out, num_hidden_out
#
# if __name__ == '__main__':
#     s=tsm()
#     s.features()

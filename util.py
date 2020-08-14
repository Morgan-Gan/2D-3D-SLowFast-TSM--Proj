
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from bbox import bbox_iou


COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]


def draw_bbox(img, box, cls_name, identity=None, offset=(0,0)):
    '''
        draw box of an id
    '''
    x1,y1,x2,y2 = [int(i+offset[idx%2]) for idx,i in enumerate(box)]
    # set color and label text
    color = COLORS_10[identity%len(COLORS_10)] if identity is not None else COLORS_10[0]
    label = '{} {}'.format(cls_name, identity)
    # box text and bar
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
    cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
    cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
    return img


# def draw_bboxes(img, bbox, identities=None, offset=(0,0)):
#     for i,box in enumerate(bbox):
#         x1,y1,x2,y2 = [int(i) for i in box]
#         x1 += offset[0]
#         x2 += offset[0]
#         y1 += offset[1]
#         y2 += offset[1]
#         # box text and bar
#         id = int(identities[i]) if identities is not None else 0
#         color = COLORS_10[id%len(COLORS_10)]
#         #label = '{}{}'.format("id", id)
#         label = '{}'.format(id)
#         t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
#         cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
#         cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
#         cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 1)
#     return img

#######################################################################################################
def draw_bboxes(img, bbox, identities=None, distance=None, speed = None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        dist = distance[i]
        sp = speed[i]

        color = COLORS_10[id%len(COLORS_10)]
        #label = '{}{}'.format("id", id)
        label = '{}'.format(id)
        label_dist = 'dist: {:.2f}'.format(dist)
        label_sp = 'speed: {:.2f}/s'.format(sp)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        dist_size = cv2.getTextSize(label_dist, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        sp_size = cv2.getTextSize(label_sp, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 1)
        cv2.putText(img, label_dist, (x1, y1-4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        cv2.putText(img, label_sp, (x1, y1-4-dist_size[1]-4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img
#########################################################################################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix

#predict_transform 函数把检测特征图转换成二维张量，张量的每一行对应边界框的属性
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    #锚点的维度与 net 块的 height 和 width 属性一致。这些属性描述了输入图像的维度，比检测图的规模大（二者之商即是步幅）。因此，我们必须使用检测特征图的步幅分割锚点。
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    #Sigmoid the  centre_X, centre_Y. and object confidencce  对 (x,y) 坐标和 objectness 分数执行 Sigmoid 函数操作。
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    #Add the center offsets   将网格偏移添加到中心坐标预测中：
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset
    #log space transform height and the width  将锚点应用到边界框维度中：
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    #Softmax the class scores   将 sigmoid 激活函数应用到类别分数中：
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
    prediction[:,:,:4] *= stride
    return prediction

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def get_im_dim(im):
    im = cv2.imread(im)
    w,h = im.shape[1], im.shape[0]
    return w,h

#去除类中重复元素
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

"""#该函数的输入为预测结果、置信度（objectness 分数阈值）、num_classes（我们这里是 80）和 nms_conf（NMS IoU 阈值）
write_results 函数输出一个形状为 Dx8 的张量；其中 D 是所有图像中的「真实」检测结果，每个都用一行表示。每一个检测结果都有
 8 个属性，即：该检测结果所属的 batch 中图像的索引、4 个角的坐标、objectness 分数、有最大置信度的类别的分数、该类别的索引。
"""
def write_results(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    # try:
    #     ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    # except:
    #     return 0
    #使用每个框的两个对角坐标能更轻松地计算两个框的 IoU。
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]

    #一次只能完成一张图像的置信度阈值设置和 NMS
    batch_size = prediction.size(0)
    # output = prediction.new(1, prediction.size(2) + 1)
    write = False
    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        #Get rid of the zero entries 目的是处理无检测结果的情况。在这种情况下，我们使用 continue 来跳过对本图像的循环。
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        if non_zero_ind.shape[0] > 0:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        else:
            continue
        if image_pred_.shape[0] == 0:
            continue

        # # only person
        # # print(image_pred_)
        # person_mask = image_pred_ * (image_pred_[:, -1] == 0).float().unsqueeze(1)
        # person_mask_ind = torch.nonzero(person_mask[:, -2])
        # # print(image_pred_)
        #
        # # print(person_mask_ind.shape)
        # # print(person_mask_ind)
        #
        #
        # if person_mask_ind.shape[0] > 0:
        #     image_pred_ = image_pred_[person_mask_ind.squeeze(),:].view(-1,7)
        #     # print(image_pred_)
        #     # print(123)
        # else:
        #     continue
        # if image_pred_.shape[0] == 0:
        #     continue
        # # end of only person

        #Get the various classes detected in the image
        try:     #因为同一类别可能会有多个「真实」检测结果，所以我们使用一个名叫 unique 的函数来获取任意给定图像中存在的类别。
            img_classes = unique(image_pred_[:,-1])
        except:
             continue

        #WE will do NMS classwise  提取特定类别（用变量 cls 表示）的检测结果。
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
             #sort the detections such that the entry with the maximum objectness
             #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
                    except IndexError:
                        break
                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask
                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    try:
        return output
    except:
        return 0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 00:12:16 2018
@author: ayooshmac
每个anchor的属性(x,y,w,h,s,s_cls1,s_cls2...)，s是方框含有目标的置信度得分，s_cls1,s_cls_2等是方框所含目标对应每类的概率
输入的feature map(prediction变量) 维度为(batch_size, num_anchors*bbox_attrs, grid_size, grid_size)，
"""

def predict_transform_half(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)    ##416//13=32
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    grid_size = inp_dim // stride    # feature map每条边格子的数量，416//32=13

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    if CUDA:
        x_offset = x_offset.cuda().half()
        y_offset = y_offset.cuda().half()
    # 这里的x_y_offset对应的是最终的feature map中每个格子的左上角坐标，比如有13个格子，刚x_y_offset的坐标就对应为(0,0),(0,1)…(12,12) .view(-1, 2)将tensor变成两列，unsqueeze(0)在0维上添加了一维。
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
    #log space transform height and the width
    anchors = torch.HalfTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()       #长度为6的list(三个anchors每个2个坐标)，
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Softmax the class scores
    prediction[:,:,5: 5 + num_classes] = nn.Softmax(-1)(Variable(prediction[:,:, 5 : 5 + num_classes])).data
    prediction[:,:,:4] *= stride
    return prediction

'''
    必须使我们的输出满足 objectness 分数阈值和非极大值抑制（NMS），以得到后文所提到的「真实」检测结果。要做到这一点就要用 write_results函数。
    函数的输入为预测结果、置信度（objectness 分数阈值）、num_classes（我们这里是 80）和 nms_conf（NMS IoU 阈值）。
    write_results()首先将网络输出方框属性(x,y,w,h)转换为在网络输入图片(416x416)坐标系中，方框左上角与右下角坐标(x1,y1,x2,y2)，以方便NMS操作。
    然后将方框含有目标得分低于阈值的方框去掉，提取得分最高的那个类的得分max_conf，同时返回这个类对应的序号max_conf_score,
    然后进行NMS操作。最终每个方框的属性为(ind,x1,y1,x2,y2,s,s_cls,index_cls)，ind 是这个方框所属图片在这个batch中的序号，
    x1,y1是在网络输入图片(416x416)坐标系中，方框左上角的坐标；x2,y2是在网络输入图片(416x416)坐标系中，方框右下角的坐标。
    s是这个方框含有目标的得分,s_cls是这个方框中所含目标最有可能的类别的概率得分，index_cls是s_cls对应的这个类别所对应的序号.
    '''
def write_results_half(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    # confidence: 输入的预测shape=(1,10647, 85)。conf_mask: shape=(1,10647) => 增加一维度之后 (1, 10647, 1)
    conf_mask = (prediction[:,:,4] > confidence).half().unsqueeze(2)
    prediction = prediction*conf_mask
    
    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0

    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    
    batch_size = prediction.size(0)
    output = prediction.new(1, prediction.size(2) + 1)
    write = False
    
    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]           #二维tensor 维度为10647x85
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)    #nms_conf（NMS IoU 阈值），类对应的序号max_conf_score
        max_conf = max_conf.half().unsqueeze(1)
        max_conf_score = max_conf_score.half().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)    # shape=(10647, 5+1+1=7)，(x1,y1,x2,y2,s,s_cls,index_cls)。
        # 与得分最高的这个类的分数s_cls(max_conf)和对应类的序号index_cls(max_conf_score)
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:]
        except:
            continue
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1].long()).half()

        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).half().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind]
             #sort the detections such that the entry with the maximum objectness
             #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).half().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask
                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind]
            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    return output

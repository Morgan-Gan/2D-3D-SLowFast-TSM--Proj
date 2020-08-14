from __future__ import division
import time

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

os.environ['DISPLAY'] = 'localhost:12.0'
def index2class():
    file_path = '/home/ganhaiyang/dataset/ava/ava_labels/ava_action_list_v2.0.csv'
    with open(file_path) as f:
        i2c_dic = {line.split(',')[0]: line.split(',')[1] for line in f}
    return i2c_dic

def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    if CUDA:
        img_ = img_.cuda()
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns a Variable 
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()                                    #实现从BGR到RGB的转换
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def normalize(frame):
    # Normalize the buffer
    frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
    return frame

def to_tensor(buffer):
    # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
    # D = Depth (in this case, time), H = Height, W = Width, C = Channels
    return buffer.transpose([3, 0, 1, 2])

def imshow(bboxes, labels, probs,ids,count):
    for bbox, lables, prob,i in zip(bboxes, labels, probs,ids):
        count_2 = 0
        for lable, p in zip(lables, prob):
            count_2 = count_2 + 1
            bbox = np.array(bbox)
            lable = int(lable)
            p = float(p)
            real_x_min = int(bbox[0])
            real_y_min = int(bbox[1])
            real_x_max = int(bbox[2])
            real_y_max = int(bbox[3])
            # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
            cv2.rectangle(frame, (real_x_min, real_y_min), (real_x_max, real_y_max), (0, 0, 255), 4)  # 红色
            cv2.putText(frame, index2class()[str(lable)].split("(")[0] + ':' + str(round(p, 2)),
                        (real_x_min + 15, real_y_max - 15 * count_2),cv2.FONT_HERSHEY_COMPLEX,  0.5, (0, 0, 255), 1, False)
            cv2.putText(frame, "id:"+str(i), (real_x_min + 10, real_y_min + 20),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, False)
        cv2.imwrite('/home/ganhaiyang/dataset/ava/save_frames/%d.jpg' % count, frame)

def arg_parse():
    """
    Parse arguements to the detect module
    """
    # 创建一个ArgumentParser对象，格式: 参数名, 目标参数(dest是字典的key),帮助信息,默认值,类型
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
    parser.add_argument("--video", dest = 'video', help =  "Video to run detection upon", default = "/home/fs/data/video_for_test_reid/192.168.123.64_01_20190529141711976.mp4", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5) #confidence  目标检测结果置信度阈值"
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4) #nms_thresh  NMS非极大值抑制阈值
    parser.add_argument("--cfg", dest = 'cfgfile', help ="Config file", default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =   "weightsfile",default = "weights/yolov3.weights", type = str)   #/home/ganhaiyang/dataset/ava/ava_weights/yolov3.weights
    parser.add_argument("--reso", dest = 'reso', help = "Input resolution of the network.",default = "416", type = str) #reso "网络输入分辨率. 分辨率越高,则准确率越高,速度越慢; 反之亦然.
    #parser.add_argument("--scales", dest="scales", help="缩放尺度用于检测", default="1,2,3", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()                                                                     # args是一个namespace类型的变量，即argparse.Namespace, 可以像easydict一样使用,就像一个字典，key来索引变量的值
    confidence = float("0.5")
    nms_thesh = float("0.4")
    start = 0
    CUDA = torch.cuda.is_available()
    num_classes = 80                                                                      # coco 数据集有80类
    CUDA = torch.cuda.is_available()
    bbox_attrs = 5 + num_classes
    
    print("Loading network.....")
    model = Darknet("cfg/yolov3.cfg")                                                     ## Darknet类中初始化时得到了网络结构和网络的参数信息，保存在net_info，module_list中
    # Load weights
    if args.weightsfile.endswith('.weights'):                                             # darknet format将权重文件载入，并复制给对应的网络结构model中
        model.load_weights(args.weightsfile)                                              # model.load_darknet_weights(opt.weights_path)
        print("load .weights file ")
    elif args.weightsfile.endswith('.pt'):                                                # pytorch format
        checkpoint = torch.load(args.weightsfile, map_location='cpu')
        model.load_state_dict(checkpoint)  # ['model']
        print("load .pt file ")
    print("Darknet Network successfully loaded")

    print("load deep sort network....")
    deepsort = DeepSort("/home/ganhaiyang/output/deepsort/checkpoint/ckpt.t7")
    print("Deep Sort Network successfully loaded")

    # 网络输入数据大小
    model.net_info["height"] = args.reso                                                 # model类中net_info是一个字典。’height’是图片的宽高，因为图片缩放到416x416，所以宽高一样大
    inp_dim = int(model.net_info["height"])                                              #inp_dim是网络输入图片尺寸（如416*416）
    assert inp_dim % 32 == 0                                                             #如果设定的输入图片的尺寸不是32的倍数或者不大于32，抛出异常
    assert inp_dim > 32

    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model.to(device)

    model.eval()
    #######for sp detec##########
    #初始化模型
    path_to_checkpoint =  "/home/ganhaiyang/dataset/ava/ava_weights/slowfast_weight.pth"
    backbone_name = Config.BACKBONE_NAME
    backbone = BackboneBase.from_name(backbone_name)()
    model_sf = Model(backbone, 81, pooler_mode=Config.POOLER_MODE, anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=TrainConfig.RPN_PRE_NMS_TOP_N,rpn_post_nms_top_n=TrainConfig.RPN_POST_NMS_TOP_N).cuda()
    model_sf.load(path_to_checkpoint)

    # videofile = "/data/video_caption_database/ava/ava/preproc_train/clips/gjdgj04FzR0/1611.mp4"     #2DUITARAsWQ
    # videofile = "/home/ganhaiyang/dataset/ava/FFOutputvideo2110.avi"
    videofile = "/home/ganhaiyang/dataset/ava/persondog.mp4"

    cap = cv2.VideoCapture(videofile)
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    ##########################################################
    last = np.array([])
    last_time = time.time()
    ##########################################################
    start = time.time()
    #######for sp detec##########
    buffer = deque(maxlen=64)
    resize_width=400
    resize_height=300

    count=0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            #######for sp detec##########
            f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                    #frame shape tuple :1920,2560,3
            # will resize frames if not already final size
            f = cv2.resize(frame, (resize_width, resize_height))
            f=normalize(f)
            buffer.append(f)

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                            #500
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                          #333
            scale = [resize_width / frame_width, resize_height / frame_height]

            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            with torch.no_grad():                                                         # 取消梯度计算
                output = model(Variable(img), CUDA)                                       #torch.Size([1, 10647, 85])
            # 8 个属性，即：该检测结果所属的 batch 中图像的索引、4 个角的坐标、objectness 分数、有最大置信度的类别的分数、该类别的索引。
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)  #tuple 7,8

            if type(output) == int:
                frames += 1
                print("FPS1 of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            
            #应该将方框的坐标转换为相对于填充后的图片中包含原始图片区域的计算方式。
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            # 将相对于输入网络图片(416x416)的方框属性变换成原图按照纵横比不变进行缩放后的区域的坐标。
            # scaling_factor*img_w和scaling_factor*img_h是图片按照纵横比不变进行缩放后的图片，即原图是768x576按照纵横比长边不变缩放到了416*372。
            # 经坐标换算,得到的坐标还是在输入网络的图片(416x416)坐标系下的绝对坐标，但是此时已经是相对于416*372这个区域的坐标了，而不再相对于(0,0)原点。
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2                #x1=x1−(416−scaling_factor*img_w)/2,x2=x2-(416−scaling_factor*img_w)/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2                #y1=y1-(416−scaling_factor*img_h)/2,y2=y2-(416−scaling_factor*img_h)/2
            output[:,1:5] /= scaling_factor

            # 如果映射回原始图片中的坐标超过了原始图片的区域，则x1,x2限定在[0,img_w]内，img_w为原始图片的宽度。如果x1,x2小于0.0，令x1,x2为0.0，如果x1,x2大于原始图片宽度，令x1,x2大小为图片的宽度。
            # 同理，y1,y2限定在0,img_h]内，img_h为原始图片的高度。clamp()函数就是将第一个输入对数的值限定在后面两个数字的区间
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

            output = output.cpu().data.numpy()
            bbox_xywh = output[:, 1:5]
            bbox_xywh[:,2] = bbox_xywh[:,2] - bbox_xywh[:,0]
            bbox_xywh[:,3] = bbox_xywh[:,3] - bbox_xywh[:,1]
            bbox_xywh[:, 0] = bbox_xywh[:, 0] + (bbox_xywh[:,2])/2
            bbox_xywh[:, 1] = bbox_xywh[:, 1] + (bbox_xywh[:, 3])/2

            cls_conf = output[:, 5]
            cls_ids = output[:, 7]

            if bbox_xywh is not None:
                mask = cls_ids == 0.0
                bbox_xywh = bbox_xywh[mask]
                cls_conf = cls_conf[mask]
                #if bbox_xywh[0]==0 and bbox_xywh[1]==0 and bbox_xywh[2]==0 and bbox_xywh[3]==0:continue
                #print("***********{}".format(bbox_xywh))
                #cv2.imshow("debug",orig_im)
                #cv2.waitKey(0)
                outputs = deepsort.update(bbox_xywh, cls_conf, orig_im)                    #Bbox+ID，naarry 3,5
#######################################################################################
                # print('outputs = {}'.format(outputs))
                # outputs = np.array(outputs)
                # print(outputs)
                #
                # now_time = time.time()
                # diff_time = now_time-last_time
                # last_time = now_time
                # print('diff_time = {}'.format(diff_time))
                #
                # distance = []
                # speed = []
                # # a = time.time()
                # for i in range(outputs.shape[0]):
                #     if last.shape[0] == 0:
                #         last = np.array([np.insert(outputs[i], 5, [0])],dtype = 'float')
                #         distance.append(0)
                #         speed.append(0)
                #
                #     else:
                #         if outputs[i][4] not in last[:, 4]:
                #             last = np.vstack([last, np.array([np.insert(outputs[i], 5, [0])])])
                #             distance.append(0)
                #             speed.append(0)
                #
                #         else:
                #             index = np.where(last[:, 4] == outputs[i][4])[0][0]
                #             center1 = np.array(
                #                 [(outputs[i][2] + outputs[i][0]) / 2, (outputs[i][1] + outputs[i][3]) / 2])
                #             center2 = np.array(
                #                 [(last[index][2] + last[index][0]) / 2, (last[index][1] + last[index][3]) / 2])
                #             # print(center1 - center2)
                #             move = np.sqrt(np.sum((center1 - center2) * (center1 - center2)))
                #             # print(move)
                #             last[index][:4] = outputs[i][:4]
                #             last[index][-1] += move
                #             distance.append(last[index][-1])
                #             speed.append(move/diff_time)
                # # print('diff = {}'.format(time.time()-a))
                # print('speed = {}'.format(speed))
                # print('last = {}'.format(last))
                # print('distance = {}'.format(distance))

#########################################################################################
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]                                             #获得Bbox
                    identities = outputs[:, -1]                                            #获得Bbox的ID号
                    # print("out_info:",bbox_xyxy,identities)
                    # # ori_im = draw_bboxes(orig_im, bbox_xyxy, identities, offset=(0, 0))
                    # # ################################################################################################
                    # # ori_im = draw_bboxes(orig_im, bbox_xyxy, identities, distance, speed, offset=(0, 0))
                    # # #################################################################################################
            print("len(buffer):",len(buffer))
            if len(buffer)==64:
                if count%3==0:
                    #把buffer转为tensor
                    b=buffer                                                                #deque 64
                    a = time.time()
                    b=np.array(b,dtype=np.float32)
                    print("time:", time.time() - a)
                    b = to_tensor(b)                                                        #shape = 3,64,300,400
                    image_batch=torch.tensor(b, dtype=torch.float).unsqueeze(0).cuda()      #shape =1,3,64,300,400

                    # 把bbox转为tensor
                    bbox_xyxy=np.array(bbox_xyxy,dtype=np.float)                            #转化为数组
                    bbox_xyxy[:, [0, 2]] *= scale[0]
                    bbox_xyxy[:, [1, 3]] *= scale[1]
                    detector_bboxes=torch.tensor(bbox_xyxy, dtype=torch.float).unsqueeze(0).cuda()

                    #模型forward:image_batch(tensor):1,3,64,300,400;detector_bboxes(tensor):1,3,4
                    with torch.no_grad():
                        detection_bboxes, detection_classes, detection_probs = \
                            model_sf.eval().forward(image_batch, detector_bboxes_batch=detector_bboxes)
                    detection_bboxes=np.array(detection_bboxes.cpu())
                    detection_classes=np.array(detection_classes)
                    detection_probs=np.array(detection_probs)

                    #得到对应的分类标签
                    detection_bboxes[:, [0, 2]] /= scale[0]
                    detection_bboxes[:, [1, 3]] /= scale[1]
                imshow(detection_bboxes,detection_classes,detection_probs,identities,count)
                count += 1

            ##**************显示图片打开**********************
            # cv2.imshow("frame", orig_im)
            # key = cv2.waitKey(0)
            # if key & 0xFF == ord('q'):
            #     break
            # frames += 1
            # print("FPS2 of the video is {:5.2f}".format( frames / (time.time() - start)))
        else:
            break
    

    
    


from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools

class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)
        
def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--images", dest = 'images', help = "Image / Directory containing images to perform detection upon",default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help =  "Image / Directory to store detections to", default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help =  "Config file",default = "cfg/yolov3-two.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile",default = "/home/ganhaiyang/output/yolo/weight/yolov3_two_ckpt_73.pt", type = str)   #/home/ganhaiyang/Yolo_Proj/yolov3/converted.weights
    parser.add_argument("--reso", dest = 'reso', help = "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection", default = "1,2,3", type = str)
    return parser.parse_args()

if __name__ ==  '__main__':
    args = arg_parse()
    scales = args.scales
#        scales = [int(x) for x in scales.split(',')]
#        args.reso = int(args.reso)
#        num_boxes = [args.reso//32, args.reso//16, args.reso//8]    
#        scale_indices = [3*(x**2) for x in num_boxes]
#        scale_indices = list(itertools.accumulate(scale_indices, lambda x,y : x+y))
#    
#        
#        li = []
#        i = 0
#        for scale in scale_indices:        
#            li.extend(list(range(i, scale))) 
#            i = scale
#        
#        scale_indices = li
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    # CUDA = torch.cuda.is_available()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CUDA = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = 2
    classes = load_classes('cfg/coco-two.names')  #data/coco.names

    #初始化网络并载入权重Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)

    # Load weights
    if args.weightsfile.endswith('.weights'):  # darknet format
        model.load_weights(args.weightsfile)      #model.load_darknet_weights(opt.weights_path)
        print("load .weights file ")
    elif args.weightsfile.endswith('.pt'):  # pytorch format
        checkpoint = torch.load(args.weightsfile, map_location='cpu')
        model.load_state_dict(checkpoint)    #['model']
        print("load .pth file ")
        # del checkpoint

    # model.load_weights(args.weightsfile)
    print("Network successfully loaded")
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()
    #Set the model in evaluation mode
    model.eval()
    #读取输入图像,从磁盘读取图像或从目录读取多张图像。图像的路径存储在一个名为 imlist 的列表中。
    read_dir = time.time()  #是一个用于测量时间的检查点。（我们会遇到多个检查点）
    #Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
    if not os.path.exists(args.det):
        os.makedirs(args.det)

    load_batch = time.time()
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    
    if CUDA:
        im_dim_list = im_dim_list.cuda()
#创建 batch
    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1
    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]
   # 检测循环
#在 batch 上迭代，生成预测结果，检测的所有图像的预测张量（形状为 Dx8，write_results 函数的输出）
    # 连接起来。如果 write_results 函数在 batch 上的输出是一个 int 值（0），也就是说没有检测结果，那么我们就继续跳过循环的其余部分。
    i = 0
    write = False
    model(get_test_input(inp_dim, CUDA), CUDA)
    start_det_loop = time.time()
    objs = {}

    for batch in im_batches:
        #load the image 
        start = time.time()
        if CUDA:
            batch = batch.cuda()
        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        #flatten the prediction vector 
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)
#        prediction = prediction[:,scale_indices]
        #get the boxes with object confidence > threshold
        #Convert the cordinates to absolute coordinates
        #perform NMS on these boxes, and save the results 
        #I could have done NMS and saving seperately to have a better abstraction
        #But both these operations require looping, hence 
        #clubbing these ops in one loop instead of two. 
        #loops are slower than vectorised operations.
        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        if type(prediction) == int:
            i += 1
            continue
        end = time.time()
#        print(end - start)
        prediction[:,0] += i*batch_size
        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))
        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            # print(output)
            print("----------------------------------------------------------")
        i += 1

        if CUDA:
            torch.cuda.synchronize()    #确保 CUDA 核与 CPU 同步
  #在图像上绘制边界框 :  try-catch 模块来检查是否存在单个检测结果。如果不存在，就退出程序。
    try:
        output
    except NameError:
        print("No detections were made")
        exit()

#在我们绘制边界框之前，我们的输出张量中包含的预测结果对应的是该网络的输入大小，而不是图像的原始大小。
#因此，在我们绘制边界框之前，让我们将每个边界框的角属性转换到图像的原始尺寸上。
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    # print(output)
    output_recast = time.time()
    class_load = time.time()
    colors = pkl.load(open("data/pallete", "rb"))    # pickle 文件，其中包含很多可以随机选择的颜色

   #绘制边界框
    draw = time.time()
    def write(x, batches, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        return img

    list(map(lambda x: write(x, im_batches, orig_ims), output))   #画边界框
    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))  #创建了一个地址列表
    list(map(cv2.imwrite, det_names, orig_ims))   #将带有检测结果的图像写入到 det_names 中的地址
    end = time.time()
    
    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()
    
    
        
        
    
    

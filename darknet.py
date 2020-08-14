from __future__ import division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from util import count_parameters as count
from util import convert2cpu as cpu
from util import predict_transform

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
        
def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))       #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))    #img是【h,w,channel】，这里的img[:,:,::-1]是将第三个维度channel从opencv的BGR转化为pytorch的RGB，然后transpose((2,0,1))的意思是将[height,width,channel]->[channel,height,width]
    img_ = img_[np.newaxis,:,:,:]/255.0        #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                    # Convert to Variable
    return img_

#定义parse_cfg，该函数使用配置文件的路径作为输入。
def parse_cfg(cfgfile):
    """
    Takes a configuration file
    将每个块存储为词典。这些块的属性和值都以键值对的形式存储在词典中。解析过程中，
    将这些词典（由代码中的变量 block 表示）添加到列表 blocks 中。我们的函数将返回该 block。
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    """
    #首先将配置文件内容保存在字符串列表中
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')     #store the lines in a list等价于readlines
    lines = [x for x in lines if len(x) > 0] #get read of the empty lines # 去掉空行
    lines = [x for x in lines if x[0] != '#']    # 去掉以#开头的注释行
    lines = [x.rstrip().lstrip() for x in lines]  # 去掉左右两边的空格(rstricp是去掉右边的空格，lstrip是去掉左边的空格)
    # cfg文件中的每个块用[]括起来最后组成一个列表，一个block存储一个块的内容，即每个层用一个字典block存储。

    #然后，遍历预处理后的列表，得到块。
    block = {}
    blocks = []
    for line in lines:
        if line[0] == "[":               #This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)      # add it the blocks list
                block = {}                # re-init the block  # 覆盖掉已存储的block,新建一个空白块存储描述下一个块的信息(block是字典)
            block["type"] = line[1:-1].rstrip()     # 把cfg的[]中的块名作为键type的值
        else:
            key,value = line.split("=")     #按等号分割
            block[key.rstrip()] = value.lstrip()   #左边是key(去掉右空格)，右边是value(去掉左空格)，形成一个block字典的键值对
    blocks.append(block)     # 退出循环，将最后一个未加入的block加进去
    return blocks
#    print('\n\n'.join([repr(x) for x in blocks]))
# 配置文件定义了6种不同type
# 'net': 相当于超参数,网络全局配置的相关参数
# {'convolutional', 'net', 'route', 'shortcut', 'upsample', 'yolo'}

import pickle as pkl
class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
    def forward(self, x):
        padded_x = F.pad(x, (0,self.pad,0,self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x

class EmptyLayer(nn.Module):
    """
       为shortcut layer / route layer 准备, 具体功能不在此实现，在Darknet类的forward函数中有体现
       """
    def __init__(self):
        super(EmptyLayer, self).__init__()
        
#定义一个新的层 DetectionLayer 保存用于检测边界框的锚点。'''yolo 检测层的具体实现, 在特征图上使用锚点预测目标区域和类别, 功能函数在predict_transform中'''
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        global CUDA
        prediction = x
        prediction = predict_transform(prediction, inp_dim, self.anchors, num_classes, confidence, CUDA)
        return prediction

class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H*stride, W*stride)
        return x
        
class ReOrgLayer(nn.Module):
    def __init__(self, stride = 2):
        super(ReOrgLayer, self).__init__()
        self.stride= stride
    def forward(self,x):
        assert(x.data.dim() == 4)
        B,C,H,W = x.data.shape
        hs = self.stride
        ws = self.stride
        assert(H % hs == 0),  "The stride " + str(self.stride) + " is not a proper divisor of height " + str(H)
        assert(W % ws == 0),  "The stride " + str(self.stride) + " is not a proper divisor of height " + str(W)
        x = x.view(B,C, H // hs, hs, W // ws, ws).transpose(-2,-3).contiguous()
        x = x.view(B,C, H // hs * W // ws, hs, ws)
        x = x.view(B,C, H // hs * W // ws, hs*ws).transpose(-1,-2).contiguous()
        x = x.view(B, C, ws*hs, H // ws, W // ws).transpose(1,2).contiguous()
        x = x.view(B, C*ws*hs, H // ws, W // ws)
        return x

#创建构建块
"""
现在我们将使用上面 parse_cfg 返回的列表来构建 PyTorch 模块，作为配置文件中的构建块。
列表中有 5 种类型的层。PyTorch 为 convolutional 和 upsample 提供预置层。我们将通过扩展 nn.Module 类为其余层写自己的模块。
create_modules 函数使用 parse_cfg 函数返回的 blocks 列表：
"""
def create_modules(blocks):
    net_info = blocks[0]      # blocks[0]存储了cfg中[net]的信息，它是一个字典，#Captures the information about the input and pre-processing
    module_list = nn.ModuleList()  # module_list用于存储每个block,每个block对应cfg文件中一个块，类似[convolutional]里面就对应一个卷积块
    index = 0    #indexing blocks helps with implementing route  layers (skip connections)
    prev_filters = 3  #初始值对应于输入数据3通道，用来存储我们需要持续追踪被应用卷积层的卷积核数量（上一层的卷积核数量（或特征图深度））
    output_filters = []   #我们不仅需要追踪前一层的卷积核数量，还需要追踪之前每个层。随着不断地迭代，我们将每个模块的输出卷积核数量添加到 output_filters 列表上。
    
    for x in blocks:    #这里，我们迭代block[1:] 而不是blocks，因为blocks的第一个元素是一个net块，它不属于前向传播。
        module = nn.Sequential()  # 这里每个块用nn.sequential()创建为了一个module,一个module有多个层

        # check the type of block
        # create a new module for the block
        # append to module_list

        if (x["type"] == "net"):
            continue
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            ''' 1. 卷积层 '''
            # 获取激活函数/批归一化/卷积层参数（通过字典的键获取值） #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False #卷积层后接BN就不需要bias
            except:
                batch_normalize = 0
                bias = True   #卷积层后无BN层就需要bias
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
                
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
            
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)    # 给定参数负轴系数0.1
                module.add_module("leaky_{0}".format(index), activn)

        #If it's an upsampling layer
        #We use Bilinear2dUpsampling  没有使用 Bilinear2dUpsampling实际使用的为最近邻插值
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])   #这个stride在cfg中就是2，所以下面的scale_factor写2或者stride是等价的
#            upsample = Upsample(stride)
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
        
        #If it is a route layer,route layer -> Empty layer
        # route层的作用：当layer取值为正时，输出这个正数对应的层的特征，如果layer取值为负数，输出route层向后退layer层对应层的特征
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation: 正值
            if start > 0: 
                start = start - index
            if end > 0:   # 若end>0，由于end= end - index，再执行index + end输出的还是第end层的特征
                end = end - index
            route = EmptyLayer()   #创建路由层提取关于层属性的值，将其表示为一个整数，并保存在一个列表中。
           #然后我们得到一个新的称为 EmptyLayer 的层，顾名思义，就是空的层。
            module.add_module("route_{0}".format(index), route)
#在路由层之后的卷积层会把它的卷积核应用到之前层的特征图（可能是拼接的）上。以下的代码更新了 filters 变量以保存路由层输出的卷积核数量。
            if end < 0:  #若end<0，则end还是end，输出index+end(而end<0)故index向后退end层的特征。
                filters = output_filters[index + start] + output_filters[index + end]
            else:   #如果没有第二个参数，end=0，则对应下面的公式，此时若start>0，由于start = start - index，再执行index + start输出的还是第start层的特征;若start<0，则start还是start，输出index+start(而start<0)故index向后退start层的特征。
                filters= output_filters[index + start]

        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            from_ = int(x["from"])
            shortcut = EmptyLayer()   #使用空的层，因为它还要执行一个非常简单的操作（加）。没必要更新 filters 变量,因为它只是将前一层的特征图添加到后面的层上而已。
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)
            module.add_module("maxpool_{}".format(index), maxpool)
        
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        else:
            print("Something I dunno")
            assert False

#在这个回路结束时，我们做了一些统计（bookkeeping.）
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1
   #这总结了此回路的主体。在 create_modules 函数后，我们获得了包含 net_info 和 module_list 的元组。
    return (net_info, module_list)

#对 nn.Module 类别进行子分类，并将我们的类别命名为 Darknet。我们用 members、blocks、net_info 和 module_list 对网络进行初始化
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
    def get_blocks(self):
        return self.blocks
    def get_module_list(self):
        return self.module_list

      #forward 主要有两个目的。一，计算输出；二，尽早处理的方式转换输出检测特征图（例如转换之后，这些不同尺度的检测图就能够串联，不然会因为不同维度不可能实现串联）。
    def forward(self, x, CUDA):
        detections = []
        modules = self.blocks[1:]  #迭代 self.block[1:] 而不是 self.blocks，因为 self.blocks 的第一个元素是一个 net 块，它不属于前向传播。
        outputs = {}   #We cache the outputs for the route layer
        write = 0
        for i in range(len(modules)):
            module_type = (modules[i]["type"])
            if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":
                x = self.module_list[i](x)
                outputs[i] = x
            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                outputs[i] = x
            elif  module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = outputs[i-1] + outputs[i+from_]
                outputs[i] = x
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
                #Get the number of classes
                num_classes = int (modules[i]["classes"])
                #Output the result
                x = x.data      ## 这里得到的是预测的yolo层feature map
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if type(x) == int:
                    continue
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
                outputs[i] = outputs[i-1]
        try:
            return detections
        except:
            return 0

    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
        #The first 4 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4. IMages seen 
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        #The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype = np.float32)
        #迭代地加载权重文件到网络的模块上。
        ptr = 0

        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]

                #保持一个称为 ptr 的变量来追踪我们在权重数组中的位置。现在，如果 batch_normalize 检查结果是 True，则我们按以下方式加载权重：
                if (batch_normalize):
                    bn = model[1]
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                
    def save_weights(self, savedfile, cutoff = 0):
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1
        fp = open(savedfile, 'wb')
        # Attach the header at the top of the file
        self.header[3] = self.seen
        header = self.header
        header = header.numpy()
        header.tofile(fp)
        # Now, let us save the weights 
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]
            if (module_type) == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]
                if (batch_normalize):
                    bn = model[1]
                    #If the parameters are on GPU, convert them back to CPU
                    #We don't convert the parameter to GPU
                    #Instead. we copy the parameter and then convert it to CPU
                    #This is done as weight are need to be saved during training
                    cpu(bn.bias.data).numpy().tofile(fp)
                    cpu(bn.weight.data).numpy().tofile(fp)
                    cpu(bn.running_mean).numpy().tofile(fp)
                    cpu(bn.running_var).numpy().tofile(fp)
                else:
                    cpu(conv.bias.data).numpy().tofile(fp)
                #Let us save the weights for the Convolutional layers
                cpu(conv.weight.data).numpy().tofile(fp)
               




#
#dn = Darknet('cfg/yolov3.cfg')
#dn.load_weights("yolov3.weights")
#inp = get_test_input()
#a, interms = dn(inp)
#dn.eval()
#a_i, interms_i = dn(inp)

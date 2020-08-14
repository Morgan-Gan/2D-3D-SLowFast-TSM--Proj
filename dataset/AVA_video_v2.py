import os
import numpy as np
import torch.utils.data
from PIL import Image, ImageOps
from bbox1 import BBox
from typing import Tuple, List, Type, Iterator
import matplotlib.pyplot as plt
import PIL
import torch.utils.data.dataset
import torch.utils.data.sampler
from PIL import Image
from torch import Tensor
import operator
from torchvision.transforms import transforms
import cv2
from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from get_ava_performance import ava_val
from config.config import Config
from config.eval_config import EvalConfig
from config.train_config import TrainConfig
import pandas as pa
from torch.utils.data import DataLoader, Dataset
Height=224             ## default (slowfast) 250
Width=224
class AVA_video(Dataset):
    #用一个字典保存多标签信息，存放到data_dic中
    class info():
        def __init__(self, img_class, bbox,h,w,img_position):
            self.img_class = [int(img_class)]
            self.bbox = bbox
            self.height=h
            self.weight=w
            self.img_position=img_position
        def __repr__(self):
            return 'info[img_class={0}, bbox={1}]'.format(self.img_class, self.bbox)

    def __init__(self,data_dir,discard=True):
        self.bboxes=[]
        self.labels=[]
        self.image_ratios = []
        self.image_position=[]
        self.widths=[]
        self.heights=[]
        #根据name获取detector_bbox
        self.detector_bboxes_list=[]
        #for debug
        self.name_list=[]
        self.i2c_dic=self.index2class()
        self.data_dic = {}
        self.data_dic_real={}
        self.data_size={}
        self.data_format={}
        self.detector_bbox_dic={}
        self.path_to_data_dir= '/home/gan/data/video_caption_database/video_database/ava/'
        path_to_AVA_dir = os.path.join(self.path_to_data_dir,'preproc_train')
        self.path_to_videos = os.path.join(path_to_AVA_dir, 'clips')
        self.path_to_keyframe = os.path.join(path_to_AVA_dir, 'keyframes')
        self.discard=discard
        self.imshow_lable_dir=data_dir
        path_to_video_ids_txt = os.path.join(path_to_AVA_dir, data_dir)
        print("Using Training Data:", path_to_video_ids_txt)
        path_to_detector_result_txt=os.path.join(path_to_AVA_dir,Config.DETECTOR_RESULT_PATH)
        #得到每个视频的大小，通过读取第一张keyframe，存放到data_size中去
        self.get_video_size()
        # 得到每个视频的格式，存放到data_format中去
        self.get_video_format()
        #读取文件，key是文件名(aa/0930)
        self.read_file_to_dic(path_to_video_ids_txt,self.data_dic)
        #合并之前的得到的信息，得到一个合并后的dic
        self.make_multi_lable(self.data_dic)
        # 获取detector的predict_bbox
        self.read_file_to_dic(path_to_detector_result_txt, self.detector_bbox_dic)
        #对字典中的数据进行整理，变成list的形式
        self.trans_dic_to_list()

    def get_video_size(self):
        for frame in sorted(os.listdir(self.path_to_keyframe)):
            img=os.listdir(os.path.join(self.path_to_keyframe, frame))[0]
            img=cv2.imread(os.path.join(self.path_to_keyframe, frame,img))
            img_shape=img.shape
            self.data_size[frame]=(img_shape[0],img_shape[1])

    def get_video_format(self):
        for video in sorted(os.listdir(self.path_to_videos)):
            video_0 = os.listdir(os.path.join(self.path_to_videos, video))[0]
            self.data_format[video]='.'+video_0.split('.')[1]
        # print('data_format',self.data_format)    #'-5KQ66BBWC4': '.mkv', '053oq2xB3oU': '.mp4',
    #dic的key对应一个list,存放着该片段对应的n条信息8tiz63
    def read_file_to_dic(self,filename,dic):
        with open(filename, 'r') as f:
            data = f.readlines()
            for line in data:
                content = line.split(',')                   # ['2FIHxnZKg6A', '1151', '0.045', '0.252', '0.838', '0.852', '80', '109\n']
                # print("content:",content)
                key=content[0]+"/"+str(int(content[1]))     #9Rcxr3IEX4E/1236
                img_h=int(self.data_size[content[0]][0])      #360或480
                img_w = int(self.data_size[content[0]][1])
                if key not in dic:
                    dic[key] = [AVA_video.info(content[6],BBox(  # convert to 0-based pixel index
                        left=float(content[2])*img_w ,
                        top=float(content[3])*img_h ,
                        right=float(content[4])*img_w,
                        bottom=float(content[5])*img_h),img_h,img_w,key)]
                else:
                    dic[key].append(AVA_video.info(content[6], BBox(  # convert to 0-based pixel index
                        left=float(content[2]) * img_w,
                        top=float(content[3]) * img_h,
                        right=float(content[4]) * img_w,
                        bottom=float(content[5]) * img_h), img_h, img_w, key))
            # print('data_dic:',self.data_dic)      # {'-5KQ66BBWC4/902': [info[img_class=[9], bbox=BBox[l=37.4, t=54.4, r=137.5, b=292.0]], info[img_class=[12],
    def make_multi_lable(self,dic):
        for key in dic:
            pre=None
            #print("before:",dic[key])
            temp=[]
            for info in dic[key]:
                if pre==None:
                    pre=info
                    temp.append(info)
                elif operator.eq(info.bbox.tolist(),pre.bbox.tolist()):
                        temp[-1].img_class.append(info.img_class[0])
                        #这是个陷坑
                        #dic[key].remove(info)
                else:
                    pre=info
                    temp.append(info)
            dic[key]=temp

    #把dic的信息转换成一一对应的list信息（bboxes，labels，detector_bboxes_list，width，height,ratio,image_position)
    def trans_dic_to_list(self):
        for key in self.data_dic:
            #如果这个框被检测到
            if(key in self.detector_bbox_dic):
                a=self.data_dic[key]
                #一个box对应一个list的标签
                #把bbox转为【【】，【】】的形式
                self.bboxes.append([item.bbox.tolist() for item in self.data_dic[key]])
                self.labels.append([item.img_class for item in self.data_dic[key]])
                assert len(self.bboxes)==len(self.labels)
                self.detector_bboxes_list.append([item.bbox.tolist() for item in self.detector_bbox_dic[key]])
                width = int(self.data_dic[key][0].weight)
                self.widths.append(width)
                height = int(self.data_dic[key][0].height)
                self.heights.append(height)
                ratio = float(width / height)
                self.image_ratios.append(ratio)
                self.image_position.append(self.data_dic[key][0].img_position)
                # print("self.labels:",self.labels,"self.image_position :",self.image_position)     #self.labels: [[[9], [12, 17, 80], [80, 9], [9], [9], [80, 9]]] self.image_position : ['-5KQ66BBWC4/902']
            else:
                continue

    def generate_one_hot(self,lable):
        one_hot_lable=np.zeros((len(lable),81))
        for i,box_lable in enumerate(lable):
            for one in box_lable:
                for j in range(81):
                    if j==int(one):
                        one_hot_lable[i][j]=1
        return one_hot_lable

    def __len__(self) -> int:
        return len(self.image_position)
    def num_classes(self):
        return 81

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        buffer,scale,index = self.loadvideo(self.image_position, index, 1)
        bboxes = torch.tensor(self.bboxes[index], dtype=torch.float)
        one_hot_lable=self.generate_one_hot(self.labels[index])
        labels = torch.tensor(one_hot_lable, dtype=torch.float)
        detector_bboxes=torch.tensor(self.detector_bboxes_list[index])
        #image = Image.open(self.path_to_keyframe+'/'+self.image_position[index]+".jpg")
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer,self.image_position,index)
        buffer=torch.tensor(buffer, dtype=torch.float)
        scale = torch.tensor(scale, dtype=torch.float)
        img=self.image_position[index]

        bboxes[:,[0,2]]*= scale[0]
        bboxes[:,[1,3]]*= scale[1]
        detector_bboxes[:,[0,2]]*= scale[0]
        detector_bboxes[:,[1,3]]*= scale[1]

        return self.image_position[index], buffer, scale, bboxes, labels,detector_bboxes,(self.heights[index],self.widths[index])

    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        norm = []
        for i, frame in enumerate(buffer):
            if np.shape(frame)[2]!=3:
                print(np.shape(frame))
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
            buffer[i] = frame
            norm.append(frame)
        return np.array(norm,dtype="float32")

    def to_tensor(self, buffer,image_position,index):
        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        if len(np.shape(buffer))!=4:
            print('WRONG:',image_position[index], np.shape(buffer))
        try:
            buffer.transpose([3, 0, 1, 2])
        except:
            print(image_position[index],np.shape(buffer))
        return buffer.transpose([3, 0, 1, 2])

    def loadvideo(self,image_position,index,frame_sample_rate):
        formate_key = image_position[index].split('/')[0]
        fname=self.path_to_videos + '/' + image_position[index] + self.data_format[formate_key]
        remainder = np.random.randint(frame_sample_rate)
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        #训练时丢弃帧数过少的数据
        if True:
            while frame_count<80 or frame_height<=0 or frame_width<=0:
                capture.release()
                print('discard_video,frame_num:',frame_count,'dir:',fname,frame_height,frame_width)
                index = np.random.randint(self.__len__())
                formate_key = image_position[index].split('/')[0]
                fname = self.path_to_videos + '/' + image_position[index] + self.data_format[formate_key]
                capture = cv2.VideoCapture(fname)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #将图片缩放，遵循frcnn方式
        if frame_count<80:
            print("fname no:",fname,frame_width,frame_height)
        if frame_height==0 or frame_width==0:
            print("WARNING:SHIT DATA")

        resize_height=Height
        resize_width=Width
        scale=[resize_width/frame_width,resize_height/frame_height]
        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = 0
        frame_keep_count=72
        if frame_count==120:
            start_idx=43
            end_idx=115
        if frame_count==100:
            start_idx =26
            end_idx =98
        if frame_count==93:
            start_idx =18
            end_idx =90
        if frame_count!=120 and frame_count!=100 and frame_count!=93:
            end_idx=frame_count-1
            start_idx=end_idx-72
        buffer=[]
        #将数据填入空的buffer
        count = 0
        retaining = True
        sample_count = 0
        # read in each frame, one at a time into the numpy buffer array
        #end_idx=120
        while (count < end_idx and retaining):
            retaining, frame = capture.read()
            count += 1
            if count <= start_idx:
                continue
            if retaining is False:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # will resize frames if not already final size
            if (frame_height != resize_height) or (frame_width != resize_width):
                frame = cv2.resize(frame, (resize_width, resize_height))
            #buffer[sample_count] = frame
            buffer.append(frame)
        capture.release()
        if len(buffer) !=72:
            print("len(buffer)",len(buffer),frame_count,end_idx)
        if len(buffer)<72:
            try:
                for i in range(72-len(buffer)):
                    temp=buffer[-1]
                    buffer.append(temp)
                assert len(buffer)==72
            except:
                buffer.append(np.zeros((resize_height,resize_width,3)))
                print('fail padding',fname)
        if len(buffer)!=72:
            buffer=[]
            for i in range(72):
                buffer.append(np.zeros((resize_height, resize_width, 3)))
                print('fail padding???', fname)
                with open("discaed","a") as f:
                    f.write(fname+"\n")
        # if len(buffer)!=64:
        #     print('fail',fname)
        return buffer,scale,index

    def evaluate(self, path_to_results_dir: str,all_image_ids, bboxes: List[List[float]], classes: List[int], probs: List[float],img_size) -> Tuple[float, str]:
        #self._write_results(path_to_results_dir,all_image_ids, bboxes, classes, probs,img_size)
        ava_val()

    def _write_results(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]],classes, probs: List[float],img_size):
        f = open(path_to_results_dir,mode='a+')
        print(len(image_ids),len(bboxes),len(classes),len(probs))
        assert len(image_ids)==len(bboxes)==len(classes)==len(probs)
        for image_id, _bbox, _cls, _prob in zip(image_ids, bboxes, classes, probs):
            print("image_id:", image_id)
            print("bbox:", _bbox)
            print("cls:", _cls)
            print("prob:", _prob)
            print("info:", len(_bbox), len(_cls), len(_prob))
            assert len(_bbox) == len(_cls) == len(_prob)
            for bbox, cls, prob in zip(_bbox, _cls, _prob):
            #print(str(image_id.split('/')[0]),str(image_id.split('/')[1]), bbox[0]/int(img_size[1]), bbox[1], bbox[2], bbox[3],(int(cls)+1),prob,img_size[1],int(img_size[0]))
                x1=0 if bbox[0]/int(img_size[1])<0 else bbox[0]/int(img_size[1])
                y1=0 if bbox[1]/int(img_size[0])<0 else bbox[1]/int(img_size[0])
                x2=1 if bbox[2]/int(img_size[1])>1 else bbox[2]/int(img_size[1])
                y2=1 if bbox[3]/int(img_size[0])>1 else bbox[3]/int(img_size[0])
                print(str(image_id.split('/')[0]),str(image_id.split('/')[1]),x1,y1,x2,y2)
                for c,p in zip(cls,prob):
                    f.write('{:s},{:s},{:f},{:f},{:f},{:f},{:s},{:s}\n'.format(str(image_id.split('/')[0]),str(image_id.split('/')[1]), x1, y1, x2, y2,str(c),str(p)))
        f.close()

    def index2class(self):
        file_path = '/home/gan/data/video_caption_database/video_database/ava/preproc_train/ava_action_list_v2.0.csv'
        with open(file_path) as f:
            i2c_dic = {line.split(',')[0]: line.split(',')[1] for line in f}
        return i2c_dic

    def draw_bboxes_and_show(self,frame,frame_num,bboxes,labels,key_frame_start,key_frame_end,scale=1,probs=[]):
            if frame_num > key_frame_start and frame_num < key_frame_end:
                count = 0
                count_2=0
                # Capture frame-by-frame
                if len(probs)==0:#标签
                    for bbox, lable in zip(bboxes, labels):
                        count = count + 1
                        bbox = np.array(bbox)
                        lable = int(lable)
                        real_x_min = int(bbox[0] / scale)
                        real_y_min = int(bbox[1] / scale)
                        real_x_max = int(bbox[2] / scale)
                        real_y_max = int(bbox[3] / scale)
                        # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
                        cv2.rectangle(frame, (real_x_min, real_y_min), (real_x_max, real_y_max),(17,238,105), 4)#绿色
                        cv2.putText(frame, self.i2c_dic[str(lable)], (real_x_min + 15, real_y_min + 15 * count),
                                    cv2.FONT_HERSHEY_COMPLEX,0.5, (17,238,105), 1, False)
                else:#预测
                    for bbox,lable,prob in zip(bboxes,labels,probs):
                        count_2 = count_2 + 1
                        bbox=np.array(bbox)
                        lable = int(lable)
                        prob=float(prob)
                        print("probs",probs)
                        real_x_min = int(bbox[0]/scale)
                        real_y_min = int(bbox[1]/scale)
                        real_x_max = int(bbox[2]/scale)
                        real_y_max = int(bbox[3]/scale)
                        # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
                        cv2.rectangle(frame, (real_x_min, real_y_min), (real_x_max, real_y_max), (255, 0, 0), 4)  # 红色
                        cv2.putText(frame, self.i2c_dic[str(lable)]+':'+str(round(prob,2)), (real_x_max - 50, real_y_min + 15 * count),
                                    cv2.FONT_HERSHEY_COMPLEX,0.5, (255, 0, 0), 1, False)

    def test(self,item_num,frame_start=0.35,frame_end=0.95):
        for i in range(item_num):
            print(i)
            result=self.__getitem__(i)
            bboxes=result[3]
            labels=result[4]
            _scale=float(result[2])
            print('scale:',_scale)
            print ('bboxes:',bboxes)
            print ('labels:',labels)
            print('dir:',self.path_to_keyframe + '/' + result[0])
            formate_key = self.image_position[i].split('/')[0]
            cap = cv2.VideoCapture(self.path_to_videos + '/' + self.image_position[i] + self.data_format[formate_key])
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            key_frame_start = int(frame_count * frame_start)
            key_frame_end = int(frame_count * frame_end)
            frame_num = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                frame_num = frame_num + 1
                self.draw_bboxes_and_show(frame,frame_num,bboxes,labels,key_frame_start,key_frame_end,scale=_scale)
                if ret == True:
                    # 显示视频
                    cv2.imshow('Frame', frame)
                    # 刷新视频
                    cv2.waitKey(10)
                    # 按q退出
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break
    def imshow(self,item_num,frame_start=0.5,frame_end=0.9):
        for i in range(item_num):
            result=self.__getitem__(i)
            name=result[0]
            real_bboxes=[item.bbox.tolist() for item in self.data_dic_real[name]]
            real_lables=[item.img_class for item in self.data_dic_real[name]]
            probs=result[5]
            print(type(probs[0]))
            kept_indices = list(np.where(np.array(probs) > 0.2))
            bboxes=np.array(result[3])[kept_indices]
            labels=np.array(result[4])[kept_indices]
            probs=np.array(probs)[kept_indices]
            scale=result[2]
            print('scale:',scale)
            print ('bboxes:',real_bboxes)
            print ('labels:',real_lables)
            print('dir:',self.path_to_keyframe + '/' + result[0])
            formate_key = self.image_position[i].split('/')[0]
            cap = cv2.VideoCapture(self.path_to_videos+'/'+self.image_position[i]+self.data_format[formate_key])
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            key_frame_start = int(frame_count * frame_start)
            key_frame_end = int(frame_count * frame_end)
            frame_num = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                frame_num = frame_num + 1
                self.draw_bboxes_and_show(frame,frame_num, bboxes, labels, key_frame_start, key_frame_end, scale=scale,probs=probs)
                self.draw_bboxes_and_show(frame,frame_num, real_bboxes, real_lables, key_frame_start, key_frame_end)
                if ret == True:
                    # 显示视频
                    cv2.imshow('Frame', frame)
                    # 刷新视频
                    cv2.waitKey(0)
                    # 按q退出
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break
    def show_net_input(self,buffer,detect_bbox,label_bbox,labels,scale):
        label_bbox=np.array(label_bbox)
        detect_bbox=np.array(detect_bbox)
        label_bbox[:,[0, 2]] *= scale[0]
        label_bbox[:,[1, 3]] *= scale[1]
        detect_bbox[:,[0, 2]] *= scale[0]
        detect_bbox[:,[1, 3]] *= scale[1]
        print("detect_bbox:", np.round(detect_bbox,1))
        print("label_bbox:", np.round(label_bbox,1))
        print("labels:", labels)
        for f in buffer:
            for i,r in enumerate(label_bbox):
                cv2.rectangle(f, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 170, 17), 1)
                for n, l in enumerate(labels[i]):
                    cv2.putText(f, self.i2c_dic[str(l)], (int(r[0]) + 10, int(r[1]) + 10* n),
                                cv2.FONT_HERSHEY_COMPLEX,0.4,(255, 255, 0), 1, False)
            for d in detect_bbox:
                cv2.rectangle(f, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (255, 255, 255), 1)
            cv2.imshow('Frame', f)
            # 刷新视频
            cv2.waitKey(0)

if __name__ == '__main__':
    a=AVA_video('/home/ganhaiyang/output/ava/result.txt')
    a.imshow(10)

# if __name__ == '__main__':
#     train_dataloader = \
#         DataLoader(AVA_video(TrainConfig.TRAIN_DATA), batch_size=2, collate_fn=DatasetBase.padding_collate_fn,shuffle=True,num_workers=0)
#     for n_iter,( _, image_batch, _, bboxes_batch, labels_batch,detector_bboxes_batch) in enumerate(train_dataloader):
#         print("n_iter: ", n_iter)

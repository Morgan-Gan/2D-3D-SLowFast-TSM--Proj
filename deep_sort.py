import torch
import numpy as np
from deep.feature_extractor import Extractor
from sort.nn_matching import NearestNeighborDistanceMetric
from sort.preprocessing import non_max_suppression
from sort.detection import Detection
from sort.tracker import Tracker
from torch.jit import ScriptModule, script_method, trace


class DeepSort(object):                  #   DeepSort(torch.jit.ScriptModule):          #按视频帧顺序处理，每一帧的处理
    def __init__(self, model_path):
        super(DeepSort, self).__init__()
        self.min_confidence = 0.3      #根据置信度对检测框进行过滤，即对置信度不足够高的检测框及特征予以删除；
        self.nms_max_overlap = 1.0     #对检测框进行非最大值抑制，消除一个目标身上多个框的情况；
        self.extractor = Extractor(model_path, use_cuda=True)    #读取当前帧目标检测框的位置及各检测框图像块的深度特征
        max_cosine_distance = 0.2
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    # @script_method
    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections，features:为特征向量
        features = self._get_features(bbox_xywh, ori_img)
        # dectections包含 self.tlwh(左上角xy),self.confidence,self.feature
        # dectections为ndarray格式
        # 置信度筛选和nms可以考虑删除
        detections = [Detection(bbox_xywh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression( boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs

    # @script_method
    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    # @script_method
    def _get_features(self, bbox_xywh, ori_img):
        features = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            feature = self.extractor(im)[0]
            features.append(feature)
        if len(features):
            features = np.stack(features, axis=0)
        else:
            features = np.array([])
        return features

if __name__ == '__main__':
    # deepsort = DeepSort("deep/checkpoint/ckpt.t7")
    # # np1 = np.array([[321.28506, 246.26743, 122.440674, 137.36693], [236.89459, 255.63495, 125.13829, 153.11089],[86.5215, 192.09995, 79.40466, 243.59027]], dtype=np.float32)
    # # np2 = np.array([0.9823241, 0.963252, 0.9410824], dtype=np.float32)
    # # im = Image.open('origin_image.jpg')
    # # print(deepsort.update.code)
    # module = torch.jit.script(deepsort)
    # # m = torch.jit.script(deepsort.update, 'scriptmodule.pt')
    # torch.jit.save(m, 'scriptmodule.pt')
    pass

import os
from typing import Union, Tuple, List, Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from config.eval_config import EvalConfig
from backbone.base import Base as BackboneBase
from bbox1 import BBox
from roi.pooler import Pooler



class Model(nn.Module):

    def __init__(self,
                 backbone: BackboneBase,
                 num_classes: int,
                 pooler_mode: Pooler.Mode,
                 anchor_ratios: List[Tuple[int, int]],
                 anchor_sizes: List[int],
                 rpn_pre_nms_top_n: int,
                 rpn_post_nms_top_n: int,
                 anchor_smooth_l1_loss_beta: Optional[float] = None,
                 proposal_smooth_l1_loss_beta: Optional[float] = None
                 ):
        super().__init__()
        self.features, hidden, num_features_out, num_hidden_out = backbone.features()
        self.detection = Model.Detection(pooler_mode, hidden, num_hidden_out, num_classes, proposal_smooth_l1_loss_beta)

    def forward(self, image_batch: Tensor,
                gt_bboxes_batch: Tensor = None, gt_classes_batch: Tensor = None, detector_bboxes_batch: Tensor = None):
        fast_feature = self.features(image_batch)                          # 调用slowfast网络forward的返回，TSM只有 一个返回                       ##   fast_feature,slow_feature
        fast_feature = fast_feature.unsqueeze(0)
        fast_feature = fast_feature.permute(0, 2, 1, 3, 4)                 ## transpose(a,b)  a,b维度置换
        batch_size, _, _,image_height, image_width = image_batch.shape
        _,_, _,features_height, features_width = fast_feature.shape           ## slow_feature.shape
        if self.training:
            proposal_classes, proposal_class_losses = self.detection.forward(fast_feature, detector_bboxes_batch, gt_classes_batch, gt_bboxes_batch)    ## ,slow_feature
            return  proposal_class_losses
        else:
            #on work
            detector_bboxes_batch = detector_bboxes_batch.squeeze(dim=0)   #torch.Size([17, 4])
            proposal_classes = self.detection.forward(fast_feature,detector_bboxes_batch)   #torch.Size([17, 81])     ## ,slow_feature
            detection_bboxes, detection_classes, detection_probs = self.detection.generate_detections(detector_bboxes_batch, proposal_classes, image_width, image_height)
            return  detection_bboxes, detection_classes, detection_probs

    def save(self, path_to_checkpoints_dir: str, step: int, optimizer: Optimizer, scheduler: _LRScheduler) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir, 'model-{}.pth'.format(step))
        checkpoint = {
            'state_dict': self.state_dict(),
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str, optimizer: Optimizer = None, scheduler: _LRScheduler = None) -> 'Model':
        checkpoint = torch.load(path_to_checkpoint)
        self.load_state_dict(checkpoint['state_dict'])
        step=0
        return step

    class Detection(nn.Module):
        def __init__(self, pooler_mode: Pooler.Mode, hidden: nn.Module, num_hidden_out: int, num_classes: int, proposal_smooth_l1_loss_beta: float):
            super().__init__()
            self._pooler_mode = pooler_mode
            self.hidden = hidden
            self.num_classes = num_classes
            self._proposal_class = nn.Linear(num_hidden_out, num_classes)
        #working
        def forward(self,fast_feature, proposal_bboxes: Tensor,
                    gt_classes_batch: Optional[Tensor] = None, gt_bboxes_batch: Optional[Tensor] = None)  \
                -> Union[Tuple[Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:                ## slow_feature,   , Tensor
            batch_size = fast_feature.shape[0]                                                       ## fast_feature:{1,256,16,16]
            feature=nn.AvgPool3d(kernel_size=(fast_feature.shape[2], 1, 1))(fast_feature).squeeze(2)    ## feature: [1,2304,16,16]
            # slow_feature = nn.AvgPool3d(kernel_size=(slow_feature.shape[2], 1, 1))(slow_feature).squeeze(2)
            # feature=torch.cat([fast_feature, slow_feature],dim=1)
            if not self.training:
                assert batch_size==1
                proposal_batch_indices = torch.arange(end=batch_size, dtype=torch.long, device=proposal_bboxes.device).view(-1, 1).repeat(1, proposal_bboxes.shape[0])[0]
                pool = Pooler.apply(feature, proposal_bboxes, proposal_batch_indices, mode=Pooler.Mode.POOLING)
                hidden = self.hidden(pool)
                proposal_classes = self._proposal_class(hidden)
                return proposal_classes
            else:
                #过滤掉补充的0
                # find labels for each `proposal_bboxes`
                ious = BBox.iou(proposal_bboxes, gt_bboxes_batch)
                proposal_max_ious, proposal_assignments = ious.max(dim=2)
                fg_masks = proposal_max_ious >= 0.85
                if len(fg_masks.nonzero()) > 0:
                    #fg_masks.nonzero()[:, 0]是在获取batch
                    proposal_bboxes=proposal_bboxes[fg_masks.nonzero()[:, 0], fg_masks.nonzero()[:, 1]]
                    batch_indices=fg_masks.nonzero()[:, 0]
                    labels=gt_classes_batch[fg_masks.nonzero()[:, 0], proposal_assignments[fg_masks]]
                else:
                    print('bbox warning')
                    fg_masks = proposal_max_ious >= 0.5
                    proposal_bboxes = proposal_bboxes[fg_masks.nonzero()[:, 0], fg_masks.nonzero()[:, 1]]
                    batch_indices = fg_masks.nonzero()[:, 0]
                    labels = gt_classes_batch[fg_masks.nonzero()[:, 0], proposal_assignments[fg_masks]]

                #  # #空间池化，拼接
                pool = Pooler.apply(feature, proposal_bboxes, batch_indices, mode=Pooler.Mode.POOLING)
                # print("******** pool shape *******", pool.shape)   [6, 2048, 3, 3]

                hidden = self.hidden(pool)
                proposal_classes = self._proposal_class(hidden)
                proposal_class_losses = self.loss(proposal_classes, labels,batch_size,batch_indices)
                return proposal_classes, proposal_class_losses

        def loss(self, proposal_classes: Tensor,gt_proposal_classes: Tensor, batch_size,batch_indices) -> Tuple[Tensor, Tensor]:
            cross_entropies = torch.zeros(batch_size, dtype=torch.float, device=proposal_classes.device).cuda()
            for batch_index in range(batch_size):
                selected_indices = (batch_indices == batch_index).nonzero().view(-1)
                input=proposal_classes[selected_indices]
                target=gt_proposal_classes[selected_indices]
                if torch.numel(input)==0 or torch.numel(target)==0:
                    continue
                assert torch.numel(input)==torch.numel(target)
                cross_entropy =F.multilabel_soft_margin_loss(input=proposal_classes[selected_indices],target=gt_proposal_classes[selected_indices],reduction="mean")
                torch.nn.MultiLabelSoftMarginLoss
                cross_entropies[batch_index] = cross_entropy
            return cross_entropies

        def generate_detections(self, proposal_bboxes: Tensor, proposal_classes: Tensor, image_width: int, image_height: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            batch_size = proposal_bboxes.shape[0]
            detection_bboxes = BBox.clip(proposal_bboxes, left=0, top=0, right=image_width, bottom=image_height)
            detection_probs = F.sigmoid(proposal_classes)
            detection_zheng=detection_probs>=EvalConfig.KEEP
            all_detection_classes=[]
            all_detection_probs=[]
            for label,prob in zip(detection_zheng,detection_probs):
                detection_classes = []
                detection_p=[]
                for index,i in enumerate(label):
                    if i==1:
                        detection_classes.append(index)
                        detection_p.append(prob[index].item())
                all_detection_classes.append(detection_classes)
                all_detection_probs.append(detection_p)
            return detection_bboxes, all_detection_classes, all_detection_probs
            #****************准换模型转Tensor时打开******************
            # return detection_bboxes, torch.Tensor(all_detection_classes), torch.Tensor(all_detection_probs)



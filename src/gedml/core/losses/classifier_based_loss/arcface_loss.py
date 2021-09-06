import torch
import numpy as np 

from .large_margin_softmax_loss import LargeMarginSoftmaxLoss

class ArcFaceLoss(LargeMarginSoftmaxLoss):
    """
    paper: `ArcFace: Additive Angular Margin Loss for Deep Face Recognition <https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html>`_
    """
    def __init__(self, margin=28.6, scale=64, **kwargs):
        super(ArcFaceLoss, self).__init__(margin=margin, scale=scale, **kwargs)

    def initiate_margin(self):
        self.margin = np.radians(self.margin)
    
    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        angles = self.get_angles(cosine_of_target_classes)
        return torch.cos(angles + self.margin)
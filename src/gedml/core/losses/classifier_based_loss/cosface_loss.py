import torch

from .large_margin_softmax_loss import LargeMarginSoftmaxLoss

class CosFaceLoss(LargeMarginSoftmaxLoss):
    """
    paper: `CosFace: Large Margin Cosine Loss for Deep Face Recognition <http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_CosFace_Large_Margin_CVPR_2018_paper.html>`_
    """
    def __init__(self, margin=0.35, scale=64, **kwargs):
        super(CosFaceLoss, self).__init__(margin=margin, scale=scale, **kwargs)
    
    def initiate_margin(self):
        pass

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        return cosine_of_target_classes - self.margin
    
    
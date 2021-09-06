import torch

from .contrastive_loss import ContrastiveLoss

class SignalToNoiseRatioLoss(ContrastiveLoss):
    """
    paper: `Signal-To-Noise Ratio: A Robust Distance Metric for Deep Metric Learning <https://openaccess.thecvf.com/content_CVPR_2019/html/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.html>`_
    """
    def __init__(
        self,
        **kwargs
    ):
        super(SignalToNoiseRatioLoss, self).__init__(**kwargs)
    
    def required_metric(self):
        return ["snr"]
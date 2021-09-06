import torchvision.transforms.functional as F
from PIL import Image

class TwoCropsTransformWrapper(object):
    """
    Take two random crops of one image as the query and key.
    modified from: https://github.com/facebookresearch/moco
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, img):
        query = self.base_transform(img)
        key = self.base_transform(img)
        return {
            "data": query,
            "addition_data": key
        }

# class CombineCropsTransformWrapper(object):
#     def __init__(self, base_transform):
#         self.base_transform = base_transform
    
#     def __call__(self, img):
#         stream1 = self.base_transform(img)
#         stream2 = self.base_transform(img)
#         return {
#             "data": # TODO: 
#         }

class DefaultTransformWrapper(object):
    """
    Default wrapper.
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, img):
        img = self.base_transform(img)
        return {
            "data": img,
            "addition_data": img
        }
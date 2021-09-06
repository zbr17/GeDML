import torchvision.transforms.functional as F
from PIL import Image


class ConvertToBGR(object):
    """
    Converts a PIL image from RGB to BGR.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        if len(img.getbands()) == 3:
            r, g, b = img.split()
        else:
            r, g, b = img, img, img
        img = Image.merge("RGB", (b, g, r))
        return img

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class Multiplier(object):
    """
    Multiply the pixel value by a constant.
    """
    def __init__(self, multiple):
        self.multiple = multiple

    def __call__(self, img):
        return img*self.multiple

    def __repr__(self):
        return "{}(multiple={})".format(self.__class__.__name__, self.multiple)
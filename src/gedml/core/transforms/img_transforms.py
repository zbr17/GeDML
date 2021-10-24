from PIL import Image
import random
from PIL import ImageFilter
import torchvision.transforms as transforms

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


class GaussianBlur(object):
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    """
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
    
    def __call__(self, img: Image):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __repr__(self):
        return "{}(sigma={})".format(self.__class__.__name__, self.sigma)

def RandomGaussianBlur(sigma=[0.1, 2.0], p=0.8):
    """
    Random Gaussian Blur.
    """
    return transforms.RandomApply(
        transforms=[GaussianBlur(sigma=sigma)],
        p=p
    )

def RandomColorJitter(brightness, contrast, saturation, hue, p=0.5):
    """
    Random Color Jitter.
    """
    return transforms.RandomApply(
        transforms=[transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )],
        p=p
    )
    
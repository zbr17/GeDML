"""
There are two kinds of ``transforms``:

1. **augumentation method**. Define how an image is augumented.
2. **wrapping method**. Define how to combine multi streams.
"""

from .img_transforms import (
    ConvertToBGR,
    Multiplier,
    RandomGaussianBlur,
    RandomColorJitter
)
from .wrapper_transforms import (
    TwoCropsTransformWrapper,
    DefaultTransformWrapper
)
from torchvision.transforms import (
    Resize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomGrayscale,
    ColorJitter,
    CenterCrop,
    ToTensor,
    Normalize,
    Compose,
)
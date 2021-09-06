"""
List
+++++++++++++++++++++++++++++++++++

1. `CUB200 <http://www.vision.caltech.edu/visipedia/CUB-200.html>`_
2. `Cars196 <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_
3. `Stanford Online Products <https://cvgl.stanford.edu/projects/lifted_struct/>`_
4. `ImageNet <http://www.image-net.org/>`_

Directory format
+++++++++++++++++++++++++++++++++++
.. code:: bash

    (dataset_root)
        - train
            - class0
                pic1.jpg
                ...
            ...
        - test
            ...
"""

from .cub200 import CUB200
from .cars196 import Cars196
from .online_products import OnlineProducts
from .imagenet import ImageNet
from .mini_imagenet import MiniImageNet
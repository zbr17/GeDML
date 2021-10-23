import os, sys
from os.path import join as opj
sys.path.append(
    os.path.abspath(
        opj(__file__, "../../src")
    )
)
import gedml

# debug code
from gedml.core.samplers import MPerClassSampler
from gedml.core.datasets import CUB200

import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

dataset = CUB200(
    root="/home/zbr/Workspace/datasets",
    phase="train",
    transform=transform
)

labels = dataset.get_labels()
sampler = MPerClassSampler(dataset.get_labels(), m=2, batch_size=32)

for idx, data_id in enumerate(iter(sampler)):
    print(idx, data_id, labels[data_id])

print(len(sampler))

import torch
g = torch.Generator()
g.manual_seed(0)
indices = torch.randperm(10).tolist()

print(indices)

from gedml.core.models import DeiT_S
model = DeiT_S()
data = torch.randn(1, 3, 224, 224)
output = model(data)

pass

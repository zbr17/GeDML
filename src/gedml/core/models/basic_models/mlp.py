import torch
import torch.nn as nn 

from ...modules import WithRecorder

"""
Normal multi-layer perceptron
"""

class MLP(WithRecorder):
    '''
    Normal multi-layer perceptron.

    Args:
        layer_size_list (list):
            The numbers of neurals in each layer.
        first_relu (bool):
            Whether to set ReLU at the beginning of the MLP.
        last_relu (bool):
            Whether to set ReLU at the end of the MLP.
    
    Example:
        >>> model = MLP(layer_size_list=[512, 100], first_relu=False)
    '''
    def __init__(self, layer_size_list, first_relu=True, last_relu=False, input_dim=None, output_dim=None):
        super(MLP, self).__init__()
        if input_dim is not None:
            layer_size_list[0] = int(input_dim)
        if output_dim is not None:
            layer_size_list[-1] = int(output_dim)
        self.layer_size_list = [int(item) for item in layer_size_list]
        self.first_relu = first_relu
        self.last_relu = last_relu
        # construct MLP
        layer_list = [nn.ReLU(inplace=True)] if first_relu else []
        for idx in range(len(layer_size_list)-1):
            layer_list.append(nn.Linear(layer_size_list[idx], layer_size_list[idx+1]))
            if idx != len(layer_size_list) - 2:
                layer_list.append(nn.ReLU(inplace=True))
        if last_relu:
            layer_list.append(nn.ReLU(inplace=True))
        # compose 
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]
    
    def forward(self, data):
        return self.net(data)

"""
multi-layer perceptron with batch normalization
"""

class BatchNormLayer(WithRecorder):
    def __init__(self, in_dim, out_dim, is_bn=True, is_relu=True, **kwargs):
        super(BatchNormLayer, self).__init__(**kwargs)
        layers_list = []
        # add linear 
        layers_list.append(nn.Linear(in_dim, out_dim))
        # add batch norm
        if is_bn:
            layers_list.append(nn.BatchNorm1d(out_dim))
        # add ReLU
        if is_relu:
            layers_list.append(nn.ReLU(inplace=True))
        # compose
        self.net = nn.Sequential(*layers_list)
        
    def forward(self, x):
        return self.net(x)

class BatchNormMLP(WithRecorder):
    """
    Multi-layer perceptron with batch normalization.

    Args:
        layer_size_list (list):     
            The numbers of neurals in each layer. (N + 1)
        relu_list (list):
            Whether relu is added.
        bn_list (list):
            Whether bn is added.
        first_bn (bool):
            Whether to set BN at the beginning of the MLP.
    
    Example:
        >>> model = BatchNormMLP(
            layer_size_list=[512, 512, 1024],
            relu_list=[True, False],
            bn_list=[True, False],
            first_bn=False
        )
    """
    def __init__(self, layer_size_list, relu_list, bn_list, first_bn=False, input_dim=None, output_dim=None, **kwargs):
        super(BatchNormMLP, self).__init__(**kwargs)
        if input_dim is not None:
            layer_size_list[0] = int(input_dim)
        if output_dim is not None:
            layer_size_list[-1] = int(output_dim)
        layer_size_list = [int(item) for item in layer_size_list]
        self.layer_size_list = layer_size_list
        self.relu_list = relu_list
        self.bn_list = bn_list
        self.layers_num = len(self.relu_list)
        self.first_bn = first_bn
        
        layers_list = []
        if self.first_bn:
            layers_list.append(
                nn.BatchNorm1d(self.layer_size_list[0])
            )

        for i in range(self.layers_num):
            layers_list.append(
                BatchNormLayer(
                    in_dim=self.layer_size_list[i],
                    out_dim=self.layer_size_list[i+1],
                    is_bn=self.bn_list[i],
                    is_relu=self.relu_list[i]
                )
            )
        self.net = nn.Sequential(*layers_list)
    
    def forward(self, data):
        return self.net(data)

# if __name__ == "__main__":
#     model = BatchNormMLP(
#         layer_size_list=[512, 512, 1024],
#         relu_list=[True, False],
#         bn_list=[True, False],
#         first_bn=False
#     )
#     print(model)

if __name__ == '__main__':
    model = MLP(layer_size_list=[512, 100], first_relu=False)
    print(model)
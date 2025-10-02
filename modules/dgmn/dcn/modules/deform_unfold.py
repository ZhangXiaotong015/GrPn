import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair, _triple

from ..functions.deform_unfold import deform_unfold, deform_unfold_3D


class DeformUnfold(nn.Module):

    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 deformable_groups=1,
                 bias=False):
        assert not bias
        super(DeformUnfold, self).__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups


    def forward(self, input, offset):
        return deform_unfold(input, offset, self.kernel_size, self.stride,
                           self.padding, self.dilation,
                           self.deformable_groups)

class DeformUnfold3D(nn.Module):

    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 deformable_groups=1,
                 bias=False):
        assert not bias
        super(DeformUnfold3D, self).__init__()

        # Support for 3D
        self.kernel_size = _triple(kernel_size)  # Use _triple for 3D
        self.stride = _triple(stride)  # Use _triple for 3D
        self.padding = _triple(padding)  # Use _triple for 3D
        self.dilation = _triple(dilation)  # Use _triple for 3D
        self.deformable_groups = deformable_groups

    def forward(self, input, offset):
        # Call the 3D deform_unfold function
        return deform_unfold_3D(input, offset, self.kernel_size, self.stride,
                                self.padding, self.dilation,
                                self.deformable_groups)


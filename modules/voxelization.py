import torch
import torch.nn as nn

import modules.functional as F

__all__ = ['Voxelization', 'DeVoxelization']


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(inc,outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride,padding=1),
            nn.BatchNorm3d(outc),
            nn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        # print("out:",out.shape)
        return out

class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride,padding=1),
            nn.BatchNorm3d(outc),
            nn.ReLU(True),
            nn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1,padding=1),
            nn.BatchNorm3d(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                nn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                nn.BatchNorm3d(outc),
            )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

class Voxelization(nn.Module):
    def __init__(self, resolution, layer_ind, normalize=True, eps=0, in_channels=None, kernel_size=3):
        super().__init__()
        self.r = int(resolution)
        self.layer_ind = layer_ind
        self.normalize = normalize
        self.eps = eps
        out_channels = in_channels
        voxel_layers = [
            BasicConvolutionBlock(in_channels,in_channels,kernel_size,stride=1),
            ResidualBlock(in_channels,out_channels,kernel_size,stride=1),
            ResidualBlock(out_channels,out_channels,kernel_size,stride=1)
         ]
        self.voxel_layers = nn.Sequential(*voxel_layers)

    def forward(self, features, batch, mode=None):
        if mode is None:
            # coords = batch.points[self.layer_ind].detach() # (N,3)
            coords = batch[f'points_{self.layer_ind}'].detach() # FLOPs calculation
        else:
            coords = batch.points_target[self.layer_ind].detach() # (N,3)
        coords = coords.permute(1,0).unsqueeze(0) # (1,3,N)
        features = features.permute(1,0).unsqueeze(0) # (1,C,N)
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1) # (1,3,N)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        voxelized_features = F.avg_voxelize(features, vox_coords, self.r) # (1,C,D,H,W)==(1,64,32,32,32)...

        voxelized_features = self.voxel_layers(voxelized_features)

        return voxelized_features, norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')


class DeVoxelization(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = int(resolution)

    def forward(self, voxel_features, voxel_coords):
        voxel_features_p = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        return voxel_features_p
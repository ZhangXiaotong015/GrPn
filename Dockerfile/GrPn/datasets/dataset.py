#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling datasets
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import os
import SimpleITK as sitk
import numpy as np
import sys
import re
import random
import math
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils.config import Config
from utils.mayavi_visu import *
from skimage.measure import label as la
# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """

    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([t1 + t2 * t3,
                  t7 - t9,
                  t11 + t12,
                  t7 + t9,
                  t1 + t2 * t15,
                  t19 - t20,
                  t11 - t12,
                  t19 + t20,
                  t1 + t2 * t24], axis=1)

    return np.reshape(R, (-1, 3, 3))
    
    
def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (features is None):
        return cpp_subsampling.subsample(points,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    else:
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)


def batch_grid_subsampling(points, batches_len, features=None, labels=None,
                           sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    R = None
    B = len(batches_len)
    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    #######################
    # Sunsample and realign
    #######################

    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_labels

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels


def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)

def nomalize(xyz_origin):
    # normalizing based on liver size
    xyz_scope = np.array([max(xyz_origin[:, 0]) - min(xyz_origin[:, 0]), max(xyz_origin[:, 1]) - min(xyz_origin[:, 1]),
                        max(xyz_origin[:, 2]) - min(xyz_origin[:, 2])])
    xyz_min = np.array([min(xyz_origin[:, 0]), min(xyz_origin[:, 1]), min(xyz_origin[:, 2])])
    # xyz_min = np.array([x_axis * Spacing_arr[0], y_axis * Spacing_arr[1], z_axis * Spacing_arr[2]])
    xyz_origin = (xyz_origin - xyz_min) / xyz_scope
    xyz_origin = xyz_origin.astype(np.float32)
    return xyz_origin, xyz_min, xyz_scope


def normalize_to_neg1_pos1(xyz_origin):
    xyz_scope = np.array([max(xyz_origin[:, 0]) - min(xyz_origin[:, 0]), 
                          max(xyz_origin[:, 1]) - min(xyz_origin[:, 1]), 
                          max(xyz_origin[:, 2]) - min(xyz_origin[:, 2])])
    xyz_min = np.array([min(xyz_origin[:, 0]), 
                        min(xyz_origin[:, 1]), 
                        min(xyz_origin[:, 2])])
    
    # Normalize to [0, 1]
    xyz_origin = (xyz_origin - xyz_min) / xyz_scope
    
    # Scale to [-1, 1]
    xyz_origin = (xyz_origin * 2) - 1
    xyz_origin = xyz_origin.astype(np.float32)

    return xyz_origin


def inverse_normalize(xyz_normalized, xyz_min, xyz_scope):
    # Reverse the normalization
    xyz_original = xyz_normalized * xyz_scope + xyz_min
    return xyz_original

def inverse_normalize_from_neg1_pos1(xyz_normalized, xyz_origin):
    # Compute the scope and minimum values from the original data
    xyz_scope = np.array([max(xyz_origin[:, 0]) - min(xyz_origin[:, 0]), 
                          max(xyz_origin[:, 1]) - min(xyz_origin[:, 1]), 
                          max(xyz_origin[:, 2]) - min(xyz_origin[:, 2])])
    xyz_min = np.array([min(xyz_origin[:, 0]), 
                        min(xyz_origin[:, 1]), 
                        min(xyz_origin[:, 2])])
    
    # Scale back from [-1, 1] to [0, 1]
    xyz_normalized = (xyz_normalized + 1) / 2
    
    # Denormalize to the original range
    xyz_origin_reconstructed = (xyz_normalized * xyz_scope) + xyz_min
    xyz_origin_reconstructed = xyz_origin_reconstructed.astype(np.float32)

    return xyz_origin_reconstructed

def window_level_processing(image):
    win_min = -100
    win_max = 300
    # print("win_max, win_min:", win_max, win_min)
    image = (image - win_min) / (win_max - win_min)
    image[image > 1] = 1
    image[image < 0] = 0

    return image

# ----------------------------------------------------------------------------------------------------------------------
#
#           Class definition
#       \**********************/


class Dataset_test(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, config, name=None):
        """
        Initialize parameters of the dataset here.
        """
        self.name = name
        self.config = config
        self.neighborhood_limits = []
        self.test_labels = []
        # Dict from labels to names
        self.label_to_names = {1: 'Couinaud_1',
                               2: 'Couinaud_2',
                               3: 'Couinaud_3',
                               4: 'Couinaud_4',
                               5: 'Couinaud_5',
                               6: 'Couinaud_6',
                               7: 'Couinaud_7',
                               8: 'Couinaud_8'}
        # Initialize a bunch of variables concerning class labels
        self.init_labels()
        # Update number of class and data task in configuration
        config.num_classes = self.num_classes

        test_patch_path = {'ct':[],'points':[],'points_Voxelidx':[]}

        test_patch_path['ct'] = sorted(
            [os.path.join(config.data_CT, f) for f in os.listdir(config.data_CT)]
        )

        test_patch_path['points'] = sorted(
            [os.path.join(config.data_root, f) for f in os.listdir(config.data_root)]
        )

        test_patch_path['points_Voxelidx'] = sorted(
            [os.path.join(config.data_root_voxelidx, f) 
            for f in os.listdir(config.data_root_voxelidx)]
        )
        
        self.test_patch_path = test_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.test_patch_path['points'])

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        points_data = np.loadtxt(self.test_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        points_data_Voxelidx = np.loadtxt(self.test_patch_path['points_Voxelidx'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.test_patch_path['points'][idx].split('/')[-1].split('.')[0]

        N = points_data.shape[0]
        num_samples = 50000
        num_batch = math.ceil(N/num_samples)
        np.random.seed(2025)
        np.random.shuffle(points_data)
        np.random.seed(2025)
        np.random.shuffle(points_data_Voxelidx)
        seg_inputs_list = []
        for b_i in range(num_batch):
            start_idx = b_i * num_samples
            end_idx = (b_i+1) * num_samples
            if b_i==num_batch-1:
                sampled_points = points_data[start_idx:,] 
                sampled_points_Voxelidx = points_data_Voxelidx[start_idx:,] 
            else:
                sampled_points = points_data[start_idx:end_idx,] # (N',4)
                sampled_points_Voxelidx = points_data_Voxelidx[start_idx:end_idx,]
            
            stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)
            'Normalization'
            stacked_points = nomalize(stacked_points)[0].astype(np.float32)

            stacked_points_Voxelidx = sampled_points_Voxelidx[:,:3].astype(np.int32)
            stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)
            stacked_features = window_level_processing(stacked_features)
            stack_lengths = np.array([stacked_points.shape[0]]).astype(np.int32)

            seg_inputs = self.segmentation_inputs(stacked_points, 
                                    stacked_features,
                                    stack_lengths)

            seg_inputs += [case_name]
            seg_inputs += [stacked_points_Voxelidx]
            seg_inputs_list.append(seg_inputs)
        return seg_inputs_list

    def init_labels(self):

        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.ignored_labels = []
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """

        # crop neighbors matrix
        if len(self.neighborhood_limits) > 0:
            return neighbors[:, :self.neighborhood_limits[layer]]
        else:
            return neighbors

    def segmentation_inputs(self,
                            stacked_points,
                            stacked_features,
                            stack_lengths):

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors(stacked_points, pool_p, stack_lengths, pool_b, 2 * r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 3), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)
                up_i = np.zeros((0, 1), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
            if up_i.shape[0] > 0:
                up_i = self.big_neighborhood_filter(up_i, len(input_points)+1)

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_upsamples += [up_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        ###############
        # Return inputs
        ###############

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths
        li += [stacked_features]

        return li







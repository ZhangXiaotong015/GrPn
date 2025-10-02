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



class Dataset_MSD_train(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, config, name=None):
        """
        Initialize parameters of the dataset here.
        """
        self.name = name
        self.config = config
        self.neighborhood_limits = []
        self.training_labels = []
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

        point_data_root = os.path.join(config.data_root,'Points_in_Liver') 
        # point_data_root = os.path.join(config.data_root,'Points_in_Liver_5mm') 

        train_list = np.loadtxt(os.path.join(config.data_root,'train_MSD.txt'), dtype=str)
        
        train_set = []
        for item in train_list:
            train_set.append(int(re.findall(r"\d+",item)[0]))

        # train_patch_path = {'ct':[],'points':[],'couinaud':[]}
        train_patch_path = {'points':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) in train_set:
                    # train_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver','images'), 'hepaticvessel_'+re.findall(r"\d+",name)[0]+'.nii'))
                    train_patch_path['points'].append(os.path.join(root,name))
                    # train_patch_path['couinaud'].append(os.path.join(root.replace('Points_in_Liver','annotation/couinaud'), 'hepaticvessel_'+re.findall(r"\d+",name)[0]+'.nii'))

        # train_patch_path['ct'], train_patch_path['points'], train_patch_path['couinaud'] = sorted(train_patch_path['ct']), sorted(train_patch_path['points']), sorted(train_patch_path['couinaud'])
        train_patch_path['points'] = sorted(train_patch_path['points'])

        self.train_patch_path = train_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.train_patch_path['points'])

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        print(f"Loading data for index: {idx}")
        points_data = np.loadtxt(self.train_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.train_patch_path['points'][idx].split('/')[-1].split('.')[0]
        # image = sitk.ReadImage(self.train_patch_path['ct'][idx])
        # origin = image.GetOrigin()
        # spacing = image.GetSpacing()
        # direction = image.GetDirection()
        # image = sitk.GetArrayFromImage(image) # (slices,512,512)
        # image = window_level_processing(image)
        # D,H,W = image.shape

        # couinaud_mask = sitk.ReadImage(self.train_patch_path['couinaud'][idx])
        # couinaud_mask = sitk.GetArrayFromImage(couinaud_mask) # (slices,512,512)
        # couinaud_mask_onehot = np.eye(9)[couinaud_mask]  # (slices, 512, 512, 9)
        # couinaud_mask_onehot = couinaud_mask_onehot[:,:,:,1:] # (slices, 512, 512, 8)
        # couinaud_mask_onehot = couinaud_mask_onehot.reshape(-1,8) # (D*H*W,8)
        # liver_mask = np.zeros(couinaud_mask.shape)
        # liver_mask[liver_mask>0] = 1 # (slices,512,512)
        # liver_mask = liver_mask.reshape(-1,1) # (D*H*W,1)

        # liver = sitk.ReadImage(self.train_patch_path['ct'][idx].replace('images','annotation/liver'))
        # liver = sitk.GetArrayFromImage(liver)
        # image = image * liver
        # image[image<0] = 0

        N = points_data.shape[0]
        num_samples = int(0.1 * N)
        if num_samples>50000:
            num_samples = 50000
        random_indices = np.random.choice(N, num_samples, replace=False) 
        sampled_points = points_data[random_indices] # (N',5)

        labels = sampled_points[:,4:] # (N',1)
        stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)

        # labels_full = points_data[:,4:] # (N,1)
        # stacked_points_full = points_data[:,:3].astype(np.float32) # (N,3)

        'Visualize the sampled points'
        # self.viz(stacked_points, image, labels, origin, direction, spacing)

        'point data augmentation'
        # # Set the probability for augmentation
        # augmentation_probability = 0.5
        # # Perform augmentation with a 50% chance
        # if random.random() < augmentation_probability:
        #     augmented_stacked_points = self.augmentation_transform(
        #         normalize_to_neg1_pos1(sampled_points[:, :3])
        #     )[0].astype(np.float32)
        #     augmented_stacked_points = inverse_normalize_from_neg1_pos1(
        #         augmented_stacked_points, sampled_points[:, :3]
        #     )
        # else:
        #     # No augmentation, use the original points
        #     augmented_stacked_points = stacked_points

        'Visualize the augmented sampled points'
        # self.viz(augmented_stacked_points, image, labels, origin, direction, spacing)

        'Normalization'
        stacked_points = nomalize(stacked_points)[0].astype(np.float32)
        # stacked_points = nomalize(augmented_stacked_points)[0].astype(np.float32)

        stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)
        'CT value norm'
        stacked_features = window_level_processing(stacked_features)
        
        stack_lengths = np.array([num_samples]).astype(np.int32)

        seg_inputs = self.segmentation_inputs(stacked_points, 
                                 stacked_features,
                                 labels,
                                 stack_lengths)

        # seg_inputs += [labels_full]
        # seg_inputs += [stacked_points_full]
        # seg_inputs += [[D,H,W]]
        seg_inputs += [case_name]
        return seg_inputs

    def viz(self, stacked_points, image, labels, origin, direction, spacing):
        # Step 1: Subtract Origin
        adjusted_coords = stacked_points - origin  # (N, 3)
        # Step 2: Account for Direction
        direction = np.array(direction).reshape((3, 3))
        inv_direction = np.linalg.inv(direction)  # Invert the direction matrix
        direction_corrected_coords = np.dot(adjusted_coords, inv_direction)  # (N, 3)
        # Step 3: Divide by Spacing
        voxel_coords = direction_corrected_coords / spacing  # (N, 3)
        # Step 4: Round and Clip
        voxel_coords = np.round(voxel_coords).astype(int)
        voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, image.shape[2] - 1)  # x-axis
        voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, image.shape[1] - 1)  # y-axis
        voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, image.shape[0] - 1)  # z-axis
        overlay_volume = np.zeros_like(image)
        for i, (x, y, z) in enumerate(voxel_coords):
            overlay_volume[z, y, x] = labels[i]
        # Extract the coordinates of non-zero points in overlay_volume
        z, y, x = np.nonzero(overlay_volume)
        values = overlay_volume[z, y, x]  # The corresponding label values
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot
        scatter = ax.scatter(x, y, z, c=values, cmap='jet', marker='o', s=5)
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Label Values')
        # Label axes
        ax.set_xlabel("X-axis (voxels)")
        ax.set_ylabel("Y-axis (voxels)")
        ax.set_zlabel("Z-axis (voxels)")
        plt.title("3D Overlay Volume Visualization")
        plt.show()

    def init_labels(self):
        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.ignored_labels = []
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random angle in [-5, 5] degrees

                # Create random rotations
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians

                # Generate the first vector in polar coordinates
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random theta in [-5, 5] degrees
                phi = (np.random.rand() * 2 * angle_range) - angle_range  # Random phi in [-5, 5] degrees

                # Create the first vector in Cartesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle within the range
                alpha = (np.random.rand() * 2 * angle_range) - angle_range  # Random alpha in [-5, 5] degrees

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

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
                            labels,
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
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths # [[]*5,[]*5,[]*5,[]*5,[]*5]==[[]*25]
        li += [stacked_features, labels] # [[]*27]

        return li

class Dataset_MSD_train_Disturbance(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, config, name=None):
        """
        Initialize parameters of the dataset here.
        """
        self.name = name
        self.config = config
        self.neighborhood_limits = []
        self.training_labels = []
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

        point_data_root = os.path.join(config.data_root,'Points_in_Liver_Voxelidx') 

        train_list = np.loadtxt(os.path.join(config.data_root,'train_MSD.txt'), dtype=str)
        
        train_set = []
        for item in train_list:
            train_set.append(int(re.findall(r"\d+",item)[0]))

        # train_patch_path = {'ct':[],'points':[],'couinaud':[]}
        train_patch_path = {'ct':[],'points':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) in train_set:
                    train_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver_Voxelidx','images'), 'hepaticvessel_'+re.findall(r"\d+",name)[0]+'.nii'))
                    train_patch_path['points'].append(os.path.join(root,name))
                    # train_patch_path['couinaud'].append(os.path.join(root.replace('Points_in_Liver_Voxelidx','annotation/couinaud'), 'hepaticvessel_'+re.findall(r"\d+",name)[0]+'.nii'))

        # train_patch_path['ct'], train_patch_path['points'], train_patch_path['couinaud'] = sorted(train_patch_path['ct']), sorted(train_patch_path['points']), sorted(train_patch_path['couinaud'])
        train_patch_path['ct'], train_patch_path['points'] = sorted(train_patch_path['ct']), sorted(train_patch_path['points'])

        self.train_patch_path = train_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.train_patch_path['points'])

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        print(f"Loading data for index: {idx}")
        points_data = np.loadtxt(self.train_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.train_patch_path['points'][idx].split('/')[-1].split('.')[0]
        image = sitk.ReadImage(self.train_patch_path['ct'][idx])
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        direction = np.array(direction).reshape((3, 3))
        image = sitk.GetArrayFromImage(image) # (slices,512,512)
        'CT value norm'
        image = window_level_processing(image) 
        D,H,W = image.shape

        # liver = sitk.ReadImage(self.train_patch_path['ct'][idx].replace('images','annotation/liver'))
        # liver = sitk.GetArrayFromImage(liver)
        # image = image * liver
        # image[image<0] = 0

        N = points_data.shape[0]
        num_samples = int(0.1 * N)
        if num_samples>50000:
            num_samples = 50000
        random_indices = np.random.choice(N, num_samples, replace=False) 
        sampled_points = points_data[random_indices] # (N',5) # voxel indices

        'sampled points disturbance'
        sampled_points_disturb = self.Disturbance(sampled_points, image) # voxel indices, normlizaed CT values

        'world coords transform'
        sampled_points_disturb[:,0] = sampled_points_disturb[:,0]*spacing[0]*direction[0,0]+origin[0]
        sampled_points_disturb[:,1] = sampled_points_disturb[:,1]*spacing[1]*direction[1,1]+origin[1]
        sampled_points_disturb[:,2] = sampled_points_disturb[:,2]*spacing[2]*direction[2,2]+origin[2]

        labels = sampled_points_disturb[:,4:] # (N',1)
        stacked_points = sampled_points_disturb[:,:3].astype(np.float32) # (N',3)  # unnormalized world coords

        'Normalization'
        stacked_points = nomalize(stacked_points)[0].astype(np.float32)  # normalized world coords

        stacked_features = sampled_points_disturb[:,3:4].astype(np.float32) # (N',1)  # normalized CT values
        
        stack_lengths = np.array([num_samples]).astype(np.int32)

        seg_inputs = self.segmentation_inputs(stacked_points, 
                                 stacked_features,
                                 labels,
                                 stack_lengths)

        seg_inputs += [case_name]
        return seg_inputs

    def Disturbance(self, points, Image_arr):
        for i in range(len(points)):
            z_axis = int(points[i,2])
            if(z_axis >= Image_arr.shape[0]):
                continue
            x_d = np.random.uniform(-1,1)
            y_d = np.random.uniform(-1,1)
            z_d = np.random.uniform(-1,1)

            points[i, 0] += x_d
            points[i, 1] += y_d
            points[i, 2] += z_d

            y_min = int(math.floor(points[i, 1]))
            y_max = int(math.ceil(points[i, 1]))
            x_min = int(math.floor(points[i, 0]))
            x_max = int(math.ceil(points[i, 0]))
            if(y_min<0 or x_min<0 or y_max>=Image_arr.shape[1] or x_max>=Image_arr.shape[2]):
                continue

            points[i,3] = Image_arr[z_axis,y_min,x_min]
        return points

    def viz(self, stacked_points, image, labels, origin, direction, spacing):
        # Step 1: Subtract Origin
        adjusted_coords = stacked_points - origin  # (N, 3)
        # Step 2: Account for Direction
        direction = np.array(direction).reshape((3, 3))
        inv_direction = np.linalg.inv(direction)  # Invert the direction matrix
        direction_corrected_coords = np.dot(adjusted_coords, inv_direction)  # (N, 3)
        # Step 3: Divide by Spacing
        voxel_coords = direction_corrected_coords / spacing  # (N, 3)
        # Step 4: Round and Clip
        voxel_coords = np.round(voxel_coords).astype(int)
        voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, image.shape[2] - 1)  # x-axis
        voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, image.shape[1] - 1)  # y-axis
        voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, image.shape[0] - 1)  # z-axis
        overlay_volume = np.zeros_like(image)
        for i, (x, y, z) in enumerate(voxel_coords):
            overlay_volume[z, y, x] = labels[i]
        # Extract the coordinates of non-zero points in overlay_volume
        z, y, x = np.nonzero(overlay_volume)
        values = overlay_volume[z, y, x]  # The corresponding label values
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot
        scatter = ax.scatter(x, y, z, c=values, cmap='jet', marker='o', s=5)
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Label Values')
        # Label axes
        ax.set_xlabel("X-axis (voxels)")
        ax.set_ylabel("Y-axis (voxels)")
        ax.set_zlabel("Z-axis (voxels)")
        plt.title("3D Overlay Volume Visualization")
        plt.show()

    def init_labels(self):
        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.ignored_labels = []
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random angle in [-5, 5] degrees

                # Create random rotations
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians

                # Generate the first vector in polar coordinates
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random theta in [-5, 5] degrees
                phi = (np.random.rand() * 2 * angle_range) - angle_range  # Random phi in [-5, 5] degrees

                # Create the first vector in Cartesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle within the range
                alpha = (np.random.rand() * 2 * angle_range) - angle_range  # Random alpha in [-5, 5] degrees

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

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
                            labels,
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
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths # [[]*5,[]*5,[]*5,[]*5,[]*5]==[[]*25]
        li += [stacked_features, labels] # [[]*27]

        return li

class Dataset_MSD_val(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, config, name=None):
        """
        Initialize parameters of the dataset here.
        """
        self.name = name
        self.config = config
        self.neighborhood_limits = []
        self.validation_labels = []
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

        # point_data_root = os.path.join(config.data_root,'Points_in_Liver_5mm') 
        point_data_root = os.path.join(config.data_root,'Points_in_Liver') 

        val_list = np.loadtxt(os.path.join(config.data_root,'val_MSD.txt'), dtype=str)
        
        val_set = []
        for item in val_list:
            val_set.append(int(re.findall(r"\d+",item)[0]))

        # val_patch_path = {'ct':[],'points':[]}
        val_patch_path = {'points':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) in val_set:
                    # val_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver','images'), 'hepaticvessel_'+re.findall(r"\d+",name)[0]+'.nii'))
                    val_patch_path['points'].append(os.path.join(root,name))
        # val_patch_path['ct'], val_patch_path['points'] = sorted(val_patch_path['ct']), sorted(val_patch_path['points'])
        val_patch_path['points'] = sorted(val_patch_path['points'])
        self.val_patch_path = val_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.val_patch_path['points'])

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        points_data = np.loadtxt(self.val_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.val_patch_path['points'][idx].split('/')[-1].split('.')[0]
        # image = sitk.ReadImage(self.val_patch_path['ct'][idx])
        # origin = image.GetOrigin()
        # spacing = image.GetSpacing()
        # direction = image.GetDirection()
        # image = sitk.GetArrayFromImage(image) # (slices,512,512)
        # D,H,W = image.shape

        # couinaud_mask = sitk.ReadImage(self.train_patch_path['couinaud'][idx])
        # couinaud_mask = sitk.GetArrayFromImage(couinaud_mask) # (slices,512,512)
        # couinaud_mask_onehot = np.eye(9)[couinaud_mask]  # (slices, 512, 512, 9)
        # couinaud_mask_onehot = couinaud_mask_onehot[:,:,:,1:] # (slices, 512, 512, 8)
        # couinaud_mask_onehot = couinaud_mask_onehot.reshape(-1,8) # (D*H*W,8)
        # liver_mask = np.zeros(couinaud_mask.shape)
        # liver_mask[liver_mask>0] = 1 # (slices,512,512)
        # liver_mask = liver_mask.reshape(-1,1) # (D*H*W,1)

        N = points_data.shape[0]
        num_samples = int(0.1 * N)
        if num_samples>50000:
            num_samples = 50000
        random_indices = np.random.choice(N, num_samples, replace=False) 
        sampled_points = points_data[random_indices] # (N',5)

        stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)  # unnormalized world coords
        'Normalization'
        stacked_points = nomalize(stacked_points)[0].astype(np.float32)   # normalized world coords
        
        stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)  # unnormalized CT values
        'CT value norm'
        stacked_features = window_level_processing(stacked_features)  # normalized CT values
        labels = sampled_points[:,4:] # (N',1)
        stack_lengths = np.array([num_samples]).astype(np.int32)

        # labels_full = points_data[:,4:] # (N,1)
        # stacked_points_full = points_data[:,:3].astype(np.float32) # (N,3)

        seg_inputs = self.segmentation_inputs(stacked_points, 
                                 stacked_features,
                                 labels,
                                 stack_lengths)

        # seg_inputs += [labels_full]
        # seg_inputs += [stacked_points_full]
        # seg_inputs += [[D,H,W]]
        seg_inputs += [case_name]
        return seg_inputs

    def init_labels(self):

        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.ignored_labels = []
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random angle in [-5, 5] degrees

                # Create random rotations
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians

                # Generate the first vector in polar coordinates
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random theta in [-5, 5] degrees
                phi = (np.random.rand() * 2 * angle_range) - angle_range  # Random phi in [-5, 5] degrees

                # Create the first vector in Cartesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle within the range
                alpha = (np.random.rand() * 2 * angle_range) - angle_range  # Random alpha in [-5, 5] degrees

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

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
                            labels,
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
        li += [stacked_features, labels]

        return li

class Dataset_MSD_test(Dataset):
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

        point_data_root = os.path.join(config.data_root,'Points_in_Liver') 
        test_list = np.loadtxt(os.path.join(config.data_root,'test_MSD.txt'), dtype=str)
        
        test_set = []
        for item in test_list:
            test_set.append(int(re.findall(r"\d+",item)[0]))

        test_patch_path = {'ct':[],'points':[],'points_Voxelidx':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) in test_set:
                    test_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver','images'), name.replace('.txt','.nii.gz')))
                    test_patch_path['points'].append(os.path.join(root,name))
                    test_patch_path['points_Voxelidx'].append(os.path.join(root.replace('Points_in_Liver','Points_in_Liver_Voxelidx'),name))
        test_patch_path['ct'], test_patch_path['points'], test_patch_path['points_Voxelidx'] = sorted(test_patch_path['ct']), sorted(test_patch_path['points']), sorted(test_patch_path['points_Voxelidx'])
        self.test_patch_path = test_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        # return len(self.test_patch_path['points'])
        return 5

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        points_data = np.loadtxt(self.test_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        points_data_Voxelidx = np.loadtxt(self.test_patch_path['points_Voxelidx'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.test_patch_path['points'][idx].split('/')[-1].split('.')[0]
        # image = sitk.ReadImage(self.test_patch_path['ct'][idx])
        # origin = image.GetOrigin()
        # spacing = image.GetSpacing()
        # direction = image.GetDirection()
        # image = sitk.GetArrayFromImage(image) # (slices,512,512)

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
                sampled_points = points_data[start_idx:end_idx,] # (N',5)
                sampled_points_Voxelidx = points_data_Voxelidx[start_idx:end_idx,]
            
            stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)
            'Normalization'
            stacked_points = nomalize(stacked_points)[0].astype(np.float32)

            stacked_points_Voxelidx = sampled_points_Voxelidx[:,:3].astype(np.int32)
            stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)
            stacked_features = window_level_processing(stacked_features)
            labels = sampled_points[:,4:] # (N',1)
            stack_lengths = np.array([stacked_points.shape[0]]).astype(np.int32)

            seg_inputs = self.segmentation_inputs(stacked_points, 
                                    stacked_features,
                                    labels,
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
                            labels,
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
        li += [stacked_features, labels]

        return li


class Dataset_IRCADB_train(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, config, name=None):
        """
        Initialize parameters of the dataset here.
        """
        self.name = name
        self.config = config
        self.neighborhood_limits = []
        self.training_labels = []
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

        point_data_root = os.path.join(config.data_root,'Points_in_Liver') 

        train_list = np.loadtxt(os.path.join(config.data_root,'train_3Dircadb.txt'), dtype=str)
        
        train_set = []
        for item in train_list:
            train_set.append(int(re.findall(r"\d+",item)[-1]))
        # print('len(train_set) is '+ str(len(train_set)))

        train_patch_path = {'ct':[],'points':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[-1]) in train_set:
                    train_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver','images'), '3Dircadb-'+str(int(re.findall(r"\d+",name)[-1]))+'.nii.gz'))
                    train_patch_path['points'].append(os.path.join(root,name))

        train_patch_path['ct'], train_patch_path['points'] = sorted(train_patch_path['ct']), sorted(train_patch_path['points'])

        self.train_patch_path = train_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.train_patch_path['points'])

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        print(f"Loading data for index: {idx}")
        points_data = np.loadtxt(self.train_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.train_patch_path['points'][idx].split('/')[-1].split('.')[0]
        image = sitk.ReadImage(self.train_patch_path['ct'][idx])
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        image = sitk.GetArrayFromImage(image) # (slices,512,512)
        image = window_level_processing(image)

        liver = sitk.ReadImage(self.train_patch_path['ct'][idx].replace('images','annotation/liver'))
        liver = sitk.GetArrayFromImage(liver)

        image = image * liver
        image[image<0] = 0

        N = points_data.shape[0]
        num_samples = int(0.1 * N)
        if num_samples>50000:
            num_samples = 50000
        random_indices = np.random.choice(N, num_samples, replace=False) 
        sampled_points = points_data[random_indices] # (N',5)
        labels = sampled_points[:,4:] # (N',1)

        stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)

        'Visualize the sampled points'
        # self.viz(stacked_points, image, labels, origin, direction, spacing)

        'point data augmentation'
        # # Set the probability for augmentation
        # augmentation_probability = 0.2
        # # Perform augmentation with a 20% chance
        # if random.random() < augmentation_probability:
        #     augmented_stacked_points = self.augmentation_transform(
        #         normalize_to_neg1_pos1(sampled_points[:, :3])
        #     )[0].astype(np.float32)
        #     augmented_stacked_points = inverse_normalize_from_neg1_pos1(
        #         augmented_stacked_points, sampled_points[:, :3]
        #     )
        # else:
        #     # No augmentation, use the original points
        #     augmented_stacked_points = stacked_points

        'Visualize the augmented sampled points'
        # self.viz(augmented_stacked_points, image, labels, origin, direction, spacing)

        stacked_points = nomalize(stacked_points)[0].astype(np.float32)
        # stacked_points = nomalize(augmented_stacked_points) 
        stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)
        stacked_features = window_level_processing(stacked_features)
        
        stack_lengths = np.array([num_samples]).astype(np.int32)

        seg_inputs = self.segmentation_inputs(stacked_points, 
                                 stacked_features,
                                 labels,
                                 stack_lengths)

        seg_inputs += [case_name]
        return seg_inputs

    def viz(self, stacked_points, image, labels, origin, direction, spacing):
        # Step 1: Subtract Origin
        adjusted_coords = stacked_points - origin  # (N, 3)
        # Step 2: Account for Direction
        direction = np.array(direction).reshape((3, 3))
        inv_direction = np.linalg.inv(direction)  # Invert the direction matrix
        direction_corrected_coords = np.dot(adjusted_coords, inv_direction)  # (N, 3)
        # Step 3: Divide by Spacing
        voxel_coords = direction_corrected_coords / spacing  # (N, 3)
        # Step 4: Round and Clip
        voxel_coords = np.round(voxel_coords).astype(int)
        voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, image.shape[2] - 1)  # x-axis
        voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, image.shape[1] - 1)  # y-axis
        voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, image.shape[0] - 1)  # z-axis
        overlay_volume = np.zeros_like(image)
        for i, (x, y, z) in enumerate(voxel_coords):
            overlay_volume[z, y, x] = labels[i]
        # Extract the coordinates of non-zero points in overlay_volume
        z, y, x = np.nonzero(overlay_volume)
        values = overlay_volume[z, y, x]  # The corresponding label values
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot
        scatter = ax.scatter(x, y, z, c=values, cmap='jet', marker='o', s=5)
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Label Values')
        # Label axes
        ax.set_xlabel("X-axis (voxels)")
        ax.set_ylabel("Y-axis (voxels)")
        ax.set_zlabel("Z-axis (voxels)")
        plt.title("3D Overlay Volume Visualization")
        plt.show()

    def init_labels(self):
        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.ignored_labels = []
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random angle in [-5, 5] degrees

                # Create random rotations
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians

                # Generate the first vector in polar coordinates
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random theta in [-5, 5] degrees
                phi = (np.random.rand() * 2 * angle_range) - angle_range  # Random phi in [-5, 5] degrees

                # Create the first vector in Cartesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle within the range
                alpha = (np.random.rand() * 2 * angle_range) - angle_range  # Random alpha in [-5, 5] degrees

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

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
                            labels,
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
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths # [[]*5,[]*5,[]*5,[]*5,[]*5]==[[]*25]
        li += [stacked_features, labels] # [[]*27]

        return li

class Dataset_IRCADB_val(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, config, name=None):
        """
        Initialize parameters of the dataset here.
        """
        self.name = name
        self.config = config
        self.neighborhood_limits = []
        self.validation_labels = []
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

        point_data_root = os.path.join(config.data_root,'Points_in_Liver') 

        val_list = np.loadtxt(os.path.join(config.data_root,'val_3Dircadb.txt'), dtype=str)
        
        val_set = []
        for item in val_list:
            val_set.append(int(re.findall(r"\d+",item)[-1]))
        # print('len(val_set) is '+ str(len(val_set)))

        val_patch_path = {'ct':[],'points':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[-1]) in val_set:
                    val_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver','images'), '3Dircadb-'+str(int(re.findall(r"\d+",name)[-1]))+'.nii'))
                    val_patch_path['points'].append(os.path.join(root,name))
        val_patch_path['ct'], val_patch_path['points'] = sorted(val_patch_path['ct']), sorted(val_patch_path['points'])
        self.val_patch_path = val_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.val_patch_path['points'])

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        points_data = np.loadtxt(self.val_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.val_patch_path['points'][idx].split('/')[-1].split('.')[0]
        image = sitk.ReadImage(self.val_patch_path['ct'][idx])
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        image = sitk.GetArrayFromImage(image) # (slices,512,512)

        N = points_data.shape[0]
        num_samples = int(0.1 * N)
        if num_samples>50000:
            num_samples = 50000
        random_indices = np.random.choice(N, num_samples, replace=False) 
        sampled_points = points_data[random_indices] # (N',5)

        stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)
        stacked_points = nomalize(stacked_points)[0].astype(np.float32)
        stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)
        stacked_features = window_level_processing(stacked_features)
        labels = sampled_points[:,4:] # (N',1)
        stack_lengths = np.array([num_samples]).astype(np.int32)

        seg_inputs = self.segmentation_inputs(stacked_points, 
                                 stacked_features,
                                 labels,
                                 stack_lengths)

        seg_inputs += [case_name]
        return seg_inputs

    def init_labels(self):

        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.ignored_labels = []
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random angle in [-5, 5] degrees

                # Create random rotations
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians

                # Generate the first vector in polar coordinates
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random theta in [-5, 5] degrees
                phi = (np.random.rand() * 2 * angle_range) - angle_range  # Random phi in [-5, 5] degrees

                # Create the first vector in Cartesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle within the range
                alpha = (np.random.rand() * 2 * angle_range) - angle_range  # Random alpha in [-5, 5] degrees

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

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
                            labels,
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
        li += [stacked_features, labels]

        return li

class Dataset_IRCADB_test(Dataset):
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

        point_data_root = os.path.join(config.data_root,'Points_in_Liver') 
        test_list = np.loadtxt(os.path.join(config.data_root,'test_3Dircadb.txt'), dtype=str)

        test_set = []
        for item in test_list:
            test_set.append(int(re.findall(r"\d+",item)[-1]))

        test_patch_path = {'ct':[],'points':[],'points_Voxelidx':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[-1]) in test_set:
                    test_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver','images'), '3Dircadb-'+str(int(re.findall(r"\d+",name)[-1]))+'.nii'))
                    test_patch_path['points'].append(os.path.join(root,name))
                    test_patch_path['points_Voxelidx'].append(os.path.join(root.replace('Points_in_Liver','Points_in_Liver_Voxelidx'),name))
        test_patch_path['ct'], test_patch_path['points'], test_patch_path['points_Voxelidx'] = sorted(test_patch_path['ct']), sorted(test_patch_path['points']), sorted(test_patch_path['points_Voxelidx'])
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
        image = sitk.ReadImage(self.test_patch_path['ct'][idx])
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        image = sitk.GetArrayFromImage(image) # (slices,512,512)

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
                sampled_points = points_data[start_idx:end_idx,] # (N',5)
                sampled_points_Voxelidx = points_data_Voxelidx[start_idx:end_idx,]
            
            stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)
            stacked_points = nomalize(stacked_points)[0].astype(np.float32)
            stacked_points_Voxelidx = sampled_points_Voxelidx[:,:3].astype(np.int32)
            stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)
            stacked_features = window_level_processing(stacked_features)
            labels = sampled_points[:,4:] # (N',1)
            stack_lengths = np.array([stacked_points.shape[0]]).astype(np.int32)

            seg_inputs = self.segmentation_inputs(stacked_points, 
                                    stacked_features,
                                    labels,
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
                            labels,
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
        li += [stacked_features, labels]

        return li


class Dataset_LiTS_train(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, config, name=None):
        """
        Initialize parameters of the dataset here.
        """
        self.name = name
        self.config = config
        self.neighborhood_limits = []
        self.training_labels = []
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

        # point_data_root = os.path.join(config.data_root,'Points_in_Liver_Norm_5mm') 
        point_data_root = os.path.join(config.data_root,'Points_in_Liver') 

        train_list = np.loadtxt(os.path.join(config.data_root,'train_LiTS.txt'), dtype=str)
        
        train_set = []
        for item in train_list:
            train_set.append(int(re.findall(r"\d+",item)[0]))

        # train_patch_path = {'ct':[],'points':[],'couinaud':[]}
        train_patch_path = {'points':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) in train_set:
                    # train_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver','images_LPI'), 'volume-'+str(int(re.findall(r"\d+",name)[0]))+'.nii'))
                    train_patch_path['points'].append(os.path.join(root,name))
                    # train_patch_path['couinaud'].append(os.path.join(root.replace('Points_in_Liver','annotation_LPI/couinaud'), 'LiTS-'+str(int(re.findall(r"\d+",name)[0]))+'.nii.gz'))

        # train_patch_path['ct'], train_patch_path['points'], train_patch_path['couinaud'] = sorted(train_patch_path['ct']), sorted(train_patch_path['points']), sorted(train_patch_path['couinaud'])
        train_patch_path['points'] = sorted(train_patch_path['points'])
        self.train_patch_path = train_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.train_patch_path['points'])

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        print(f"Loading data for index: {idx}")
        points_data = np.loadtxt(self.train_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.train_patch_path['points'][idx].split('/')[-1].split('.')[0]
        # image = sitk.ReadImage(self.train_patch_path['ct'][idx])
        # origin = image.GetOrigin()
        # spacing = image.GetSpacing()
        # direction = image.GetDirection()
        # image = sitk.GetArrayFromImage(image) # (slices,512,512)
        # image = window_level_processing(image)
        # D,H,W = image.shape

        # couinaud_mask = sitk.ReadImage(self.train_patch_path['couinaud'][idx])
        # couinaud_mask = sitk.GetArrayFromImage(couinaud_mask) # (slices,512,512)
        # couinaud_mask_onehot = np.eye(9)[couinaud_mask]  # (slices, 512, 512, 9)
        # couinaud_mask_onehot = couinaud_mask_onehot[:,:,:,1:] # (slices, 512, 512, 8)
        # couinaud_mask_onehot = couinaud_mask_onehot.reshape(-1,8) # (D*H*W,8)
        # liver_mask = np.zeros(couinaud_mask.shape)
        # liver_mask[liver_mask>0] = 1 # (slices,512,512)
        # liver_mask = liver_mask.reshape(-1,1) # (D*H*W,1)


        # liver = sitk.ReadImage(self.train_patch_path['ct'][idx].replace('images_LPI','annotation_LPI/liver').replace('volume-','LiTS-').replace('.nii','.nii.gz'))
        # liver = sitk.GetArrayFromImage(liver)
        # image = image * liver
        # image[image<0] = 0

        N = points_data.shape[0]
        num_samples = int(0.1 * N)
        if num_samples>50000:
            num_samples = 50000
        random_indices = np.random.choice(N, num_samples, replace=False) 
        sampled_points = points_data[random_indices] # (N',5)

        labels = sampled_points[:,4:] # (N',1)
        stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)

        # labels_full = points_data[:,4:] # (N,1)
        # stacked_points_full = points_data[:,:3].astype(np.float32) # (N,3)

        'Visualize the sampled points'
        # self.viz(stacked_points, image, labels, origin, direction, spacing)

        'point data augmentation'
        # # Set the probability for augmentation
        # augmentation_probability = 0.5
        # # Perform augmentation with a 50% chance
        # if random.random() < augmentation_probability:
        #     augmented_stacked_points = self.augmentation_transform(
        #         normalize_to_neg1_pos1(sampled_points[:, :3])
        #     )[0].astype(np.float32)
        #     augmented_stacked_points = inverse_normalize_from_neg1_pos1(
        #         augmented_stacked_points, sampled_points[:, :3]
        #     )
        # else:
        #     # No augmentation, use the original points
        #     augmented_stacked_points = stacked_points

        'Visualize the augmented sampled points'
        # self.viz(augmented_stacked_points, image, labels, origin, direction, spacing)

        'Normalization'
        stacked_points = nomalize(stacked_points)[0].astype(np.float32)
        # stacked_points = nomalize(augmented_stacked_points)[0].astype(np.float32)
        
        stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)
        'CT value norm'
        stacked_features = window_level_processing(stacked_features)
        
        stack_lengths = np.array([num_samples]).astype(np.int32)

        seg_inputs = self.segmentation_inputs(stacked_points, 
                                 stacked_features,
                                 labels,
                                 stack_lengths)

        # seg_inputs += [labels_full]
        # seg_inputs += [stacked_points_full]
        # seg_inputs += [[D,H,W]]
        seg_inputs += [case_name]
        return seg_inputs

    def viz(self, stacked_points, image, labels, origin, direction, spacing):
        # Step 1: Subtract Origin
        adjusted_coords = stacked_points - origin  # (N, 3)
        # Step 2: Account for Direction
        direction = np.array(direction).reshape((3, 3))
        inv_direction = np.linalg.inv(direction)  # Invert the direction matrix
        direction_corrected_coords = np.dot(adjusted_coords, inv_direction)  # (N, 3)
        # Step 3: Divide by Spacing
        voxel_coords = direction_corrected_coords / spacing  # (N, 3)
        # Step 4: Round and Clip
        voxel_coords = np.round(voxel_coords).astype(int)
        voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, image.shape[2] - 1)  # x-axis
        voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, image.shape[1] - 1)  # y-axis
        voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, image.shape[0] - 1)  # z-axis
        overlay_volume = np.zeros_like(image)
        for i, (x, y, z) in enumerate(voxel_coords):
            overlay_volume[z, y, x] = labels[i]
        # Extract the coordinates of non-zero points in overlay_volume
        z, y, x = np.nonzero(overlay_volume)
        values = overlay_volume[z, y, x]  # The corresponding label values
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot
        scatter = ax.scatter(x, y, z, c=values, cmap='jet', marker='o', s=5)
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Label Values')
        # Label axes
        ax.set_xlabel("X-axis (voxels)")
        ax.set_ylabel("Y-axis (voxels)")
        ax.set_zlabel("Z-axis (voxels)")
        plt.title("3D Overlay Volume Visualization")
        plt.show()

    def init_labels(self):
        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.ignored_labels = []
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random angle in [-5, 5] degrees

                # Create random rotations
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians

                # Generate the first vector in polar coordinates
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random theta in [-5, 5] degrees
                phi = (np.random.rand() * 2 * angle_range) - angle_range  # Random phi in [-5, 5] degrees

                # Create the first vector in Cartesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle within the range
                alpha = (np.random.rand() * 2 * angle_range) - angle_range  # Random alpha in [-5, 5] degrees

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

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
                            labels,
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
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths # [[]*5,[]*5,[]*5,[]*5,[]*5]==[[]*25]
        li += [stacked_features, labels] # [[]*27]

        return li

class Dataset_LiTS_train_Disturbance(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, config, name=None):
        """
        Initialize parameters of the dataset here.
        """
        self.name = name
        self.config = config
        self.neighborhood_limits = []
        self.training_labels = []
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

        point_data_root = os.path.join(config.data_root,'Points_in_Liver_Voxelidx') 

        train_list = np.loadtxt(os.path.join(config.data_root,'train_LiTS.txt'), dtype=str)
        
        train_set = []
        for item in train_list:
            train_set.append(int(re.findall(r"\d+",item)[0]))

        # train_patch_path = {'ct':[],'points':[],'couinaud':[]}
        train_patch_path = {'ct':[],'points':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) in train_set:
                    train_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver_Voxelidx','images_LPI'), 'volume-'+str(int(re.findall(r"\d+",name)[0]))+'.nii'))
                    train_patch_path['points'].append(os.path.join(root,name))
                    # train_patch_path['couinaud'].append(os.path.join(root.replace('Points_in_Liver_5mm','annotation_LPI/couinaud'), 'LiTS-'+str(int(re.findall(r"\d+",name)[0]))+'.nii.gz'))

        # train_patch_path['ct'], train_patch_path['points'], train_patch_path['couinaud'] = sorted(train_patch_path['ct']), sorted(train_patch_path['points']), sorted(train_patch_path['couinaud'])
        train_patch_path['ct'], train_patch_path['points'] = sorted(train_patch_path['ct']), sorted(train_patch_path['points'])
        self.train_patch_path = train_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.train_patch_path['points'])

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        print(f"Loading data for index: {idx}")
        points_data = np.loadtxt(self.train_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.train_patch_path['points'][idx].split('/')[-1].split('.')[0]
        image = sitk.ReadImage(self.train_patch_path['ct'][idx])
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        direction = np.array(direction).reshape((3, 3))
        image = sitk.GetArrayFromImage(image) # (slices,512,512)
        'CT value norm'
        image = window_level_processing(image)
        D,H,W = image.shape

        # liver = sitk.ReadImage(self.train_patch_path['ct'][idx].replace('images_LPI','annotation_LPI/liver').replace('volume-','LiTS-').replace('.nii','.nii.gz'))
        # liver = sitk.GetArrayFromImage(liver)
        # image = image * liver
        # image[image<0] = 0

        N = points_data.shape[0]
        num_samples = int(0.1 * N)
        if num_samples>50000:
            num_samples = 50000
        random_indices = np.random.choice(N, num_samples, replace=False) 
        sampled_points = points_data[random_indices] # (N',5)

        'sampled points disturbance'
        sampled_points_disturb = self.Disturbance(sampled_points, image) # voxel indices, normlizaed CT values

        'world coords transform'
        sampled_points_disturb[:,0] = sampled_points_disturb[:,0]*spacing[0]*direction[0,0]+origin[0]
        sampled_points_disturb[:,1] = sampled_points_disturb[:,1]*spacing[1]*direction[1,1]+origin[1]
        sampled_points_disturb[:,2] = sampled_points_disturb[:,2]*spacing[2]*direction[2,2]+origin[2]

        labels = sampled_points_disturb[:,4:] # (N',1)
        stacked_points = sampled_points_disturb[:,:3].astype(np.float32) # (N',3)  # unnormalized world coords

        'Normalization'
        stacked_points = nomalize(stacked_points)[0].astype(np.float32) # normalized world coords
        
        stacked_features = sampled_points_disturb[:,3:4].astype(np.float32) # (N',1)  # normalized CT values
        
        stack_lengths = np.array([num_samples]).astype(np.int32)

        seg_inputs = self.segmentation_inputs(stacked_points, 
                                 stacked_features,
                                 labels,
                                 stack_lengths)

        seg_inputs += [case_name]
        return seg_inputs

    def Disturbance(self, points, Image_arr):
        for i in range(len(points)):
            z_axis = int(points[i,2])
            if(z_axis >= Image_arr.shape[0]):
                continue
            x_d = np.random.uniform(-1,1)
            y_d = np.random.uniform(-1,1)
            z_d = np.random.uniform(-1,1)

            points[i, 0] += x_d
            points[i, 1] += y_d
            points[i, 2] += z_d

            y_min = int(math.floor(points[i, 1]))
            y_max = int(math.ceil(points[i, 1]))
            x_min = int(math.floor(points[i, 0]))
            x_max = int(math.ceil(points[i, 0]))
            if(y_min<0 or x_min<0 or y_max>=Image_arr.shape[1] or x_max>=Image_arr.shape[2]):
                continue

            points[i,3] = Image_arr[z_axis,y_min,x_min]
        return points

    def viz(self, stacked_points, image, labels, origin, direction, spacing):
        # Step 1: Subtract Origin
        adjusted_coords = stacked_points - origin  # (N, 3)
        # Step 2: Account for Direction
        direction = np.array(direction).reshape((3, 3))
        inv_direction = np.linalg.inv(direction)  # Invert the direction matrix
        direction_corrected_coords = np.dot(adjusted_coords, inv_direction)  # (N, 3)
        # Step 3: Divide by Spacing
        voxel_coords = direction_corrected_coords / spacing  # (N, 3)
        # Step 4: Round and Clip
        voxel_coords = np.round(voxel_coords).astype(int)
        voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, image.shape[2] - 1)  # x-axis
        voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, image.shape[1] - 1)  # y-axis
        voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, image.shape[0] - 1)  # z-axis
        overlay_volume = np.zeros_like(image)
        for i, (x, y, z) in enumerate(voxel_coords):
            overlay_volume[z, y, x] = labels[i]
        # Extract the coordinates of non-zero points in overlay_volume
        z, y, x = np.nonzero(overlay_volume)
        values = overlay_volume[z, y, x]  # The corresponding label values
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot
        scatter = ax.scatter(x, y, z, c=values, cmap='jet', marker='o', s=5)
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Label Values')
        # Label axes
        ax.set_xlabel("X-axis (voxels)")
        ax.set_ylabel("Y-axis (voxels)")
        ax.set_zlabel("Z-axis (voxels)")
        plt.title("3D Overlay Volume Visualization")
        plt.show()

    def init_labels(self):
        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.ignored_labels = []
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random angle in [-5, 5] degrees

                # Create random rotations
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians

                # Generate the first vector in polar coordinates
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random theta in [-5, 5] degrees
                phi = (np.random.rand() * 2 * angle_range) - angle_range  # Random phi in [-5, 5] degrees

                # Create the first vector in Cartesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle within the range
                alpha = (np.random.rand() * 2 * angle_range) - angle_range  # Random alpha in [-5, 5] degrees

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

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
                            labels,
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
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths # [[]*5,[]*5,[]*5,[]*5,[]*5]==[[]*25]
        li += [stacked_features, labels] # [[]*27]

        return li

class Dataset_LiTS_val(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, config, name=None):
        """
        Initialize parameters of the dataset here.
        """
        self.name = name
        self.config = config
        self.neighborhood_limits = []
        self.validation_labels = []
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

        # point_data_root = os.path.join(config.data_root,'Points_in_Liver_Norm_5mm') 
        point_data_root = os.path.join(config.data_root,'Points_in_Liver') 

        val_list = np.loadtxt(os.path.join(config.data_root,'val_LiTS.txt'), dtype=str)
        
        val_set = []
        for item in val_list:
            val_set.append(int(re.findall(r"\d+",item)[0]))

        # val_patch_path = {'ct':[],'points':[]}
        val_patch_path = {'points':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) in val_set:
                    # val_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver','images_LPI'), 'volume-'+str(int(re.findall(r"\d+",name)[0]))+'.nii'))
                    val_patch_path['points'].append(os.path.join(root,name))
        # val_patch_path['ct'], val_patch_path['points'] = sorted(val_patch_path['ct']), sorted(val_patch_path['points'])
        val_patch_path['points'] = sorted(val_patch_path['points'])
        self.val_patch_path = val_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.val_patch_path['points'])

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        points_data = np.loadtxt(self.val_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.val_patch_path['points'][idx].split('/')[-1].split('.')[0]
        # image = sitk.ReadImage(self.val_patch_path['ct'][idx])
        # origin = image.GetOrigin()
        # spacing = image.GetSpacing()
        # direction = image.GetDirection()
        # image = sitk.GetArrayFromImage(image) # (slices,512,512)
        # D,H,W = image.shape

        # couinaud_mask = sitk.ReadImage(self.train_patch_path['couinaud'][idx])
        # couinaud_mask = sitk.GetArrayFromImage(couinaud_mask) # (slices,512,512)
        # couinaud_mask_onehot = np.eye(9)[couinaud_mask]  # (slices, 512, 512, 9)
        # couinaud_mask_onehot = couinaud_mask_onehot[:,:,:,1:] # (slices, 512, 512, 8)
        # couinaud_mask_onehot = couinaud_mask_onehot.reshape(-1,8) # (D*H*W,8)
        # liver_mask = np.zeros(couinaud_mask.shape)
        # liver_mask[liver_mask>0] = 1 # (slices,512,512)
        # liver_mask = liver_mask.reshape(-1,1) # (D*H*W,1)

        N = points_data.shape[0]
        num_samples = int(0.1 * N)
        if num_samples>50000:
            num_samples = 50000
        random_indices = np.random.choice(N, num_samples, replace=False) 
        sampled_points = points_data[random_indices] # (N',5)

        stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)

        'Normalization'
        stacked_points = nomalize(stacked_points)[0].astype(np.float32)

        stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)
        'CT value norm'
        stacked_features = window_level_processing(stacked_features)
        labels = sampled_points[:,4:] # (N',1)
        stack_lengths = np.array([num_samples]).astype(np.int32)

        # labels_full = points_data[:,4:] # (N,1)
        # stacked_points_full = points_data[:,:3].astype(np.float32) # (N,3)

        seg_inputs = self.segmentation_inputs(stacked_points, 
                                 stacked_features,
                                 labels,
                                 stack_lengths)

        # seg_inputs += [labels_full]
        # seg_inputs += [stacked_points_full]
        # seg_inputs += [[D,H,W]]
        seg_inputs += [case_name]
        return seg_inputs

    def init_labels(self):

        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.ignored_labels = []
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random angle in [-5, 5] degrees

                # Create random rotations
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians

                # Generate the first vector in polar coordinates
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random theta in [-5, 5] degrees
                phi = (np.random.rand() * 2 * angle_range) - angle_range  # Random phi in [-5, 5] degrees

                # Create the first vector in Cartesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle within the range
                alpha = (np.random.rand() * 2 * angle_range) - angle_range  # Random alpha in [-5, 5] degrees

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

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
                            labels,
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
        li += [stacked_features, labels]

        return li

class Dataset_LiTS_test(Dataset):
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

        # point_data_root = os.path.join(config.data_root,'Points_in_Liver_Norm_5mm') 
        point_data_root = os.path.join(config.data_root,'Points_in_Liver') 
        test_list = np.loadtxt(os.path.join(config.data_root,'test_LiTS.txt'), dtype=str)
        
        test_set = []
        for item in test_list:
            test_set.append(int(re.findall(r"\d+",item)[0]))

        test_patch_path = {'ct':[],'points':[],'points_Voxelidx':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) in test_set:
                    # test_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver_Norm_5mm','Vol_LPI_resize_CT_5mm'), 'volume-'+str(int(re.findall(r"\d+",name)[0]))+'.nii'))
                    test_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver','images_LPI'), 'volume-'+str(int(re.findall(r"\d+",name)[0]))+'.nii'))
                    test_patch_path['points'].append(os.path.join(root,name))
                    # test_patch_path['points_Voxelidx'].append(os.path.join(root.replace('Points_in_Liver_Norm_5mm','Points_in_Liver_Voxelidx_5mm'),name))
                    test_patch_path['points_Voxelidx'].append(os.path.join(root.replace('Points_in_Liver','Points_in_Liver_Voxelidx'),name))
        test_patch_path['ct'], test_patch_path['points'], test_patch_path['points_Voxelidx'] = sorted(test_patch_path['ct']), sorted(test_patch_path['points']), sorted(test_patch_path['points_Voxelidx'])
        self.test_patch_path = test_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        # return len(self.test_patch_path['points'])
        return 5

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        points_data = np.loadtxt(self.test_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        points_data_Voxelidx = np.loadtxt(self.test_patch_path['points_Voxelidx'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.test_patch_path['points'][idx].split('/')[-1].split('.')[0]
        # image = sitk.ReadImage(self.test_patch_path['ct'][idx])
        # origin = image.GetOrigin()
        # spacing = image.GetSpacing()
        # direction = image.GetDirection()
        # image = sitk.GetArrayFromImage(image) # (slices,512,512)

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
                sampled_points = points_data[start_idx:end_idx,] # (N',5)
                sampled_points_Voxelidx = points_data_Voxelidx[start_idx:end_idx,]
            
            stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)
            'Normalization'
            stacked_points = nomalize(stacked_points)[0].astype(np.float32)

            stacked_points_Voxelidx = sampled_points_Voxelidx[:,:3].astype(np.int32)
            stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)
            'CT value norm'
            stacked_features = window_level_processing(stacked_features)
            labels = sampled_points[:,4:] # (N',1)
            stack_lengths = np.array([stacked_points.shape[0]]).astype(np.int32)

            seg_inputs = self.segmentation_inputs(stacked_points, 
                                    stacked_features,
                                    labels,
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
                            labels,
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
        li += [stacked_features, labels]

        return li




class Dataset_MSD_train_HCC(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, config, name=None):
        """
        Initialize parameters of the dataset here.
        """
        self.name = name
        self.config = config
        self.neighborhood_limits = []
        self.training_labels = []
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

        # point_data_root = os.path.join(config.data_root,'Points_in_Liver') 
        point_data_root = os.path.join(config.data_root,'Points_in_Liver_Norm_5mm') 

        train_list = np.loadtxt(os.path.join(config.data_root,'train_MSD_new.txt'), dtype=str)
        
        train_set = []
        for item in train_list:
            train_set.append(int(re.findall(r"\d+",item)[0]))

        train_patch_path = {'ct':[],'points':[],'couinaud':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) in train_set:
                    train_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver_Norm_5mm','Vol_resize_CT_5mm'), name.replace('.txt','.nii.gz')))
                    train_patch_path['points'].append(os.path.join(root,name))
                    train_patch_path['couinaud'].append(os.path.join(root.replace('Points_in_Liver_Norm_5mm','Vol_resize_couinaud_5mm'), name.replace('.txt','.nii.gz')))

        train_patch_path['ct'], train_patch_path['points'], train_patch_path['couinaud'] = sorted(train_patch_path['ct']), sorted(train_patch_path['points']), sorted(train_patch_path['couinaud'])

        self.train_patch_path = train_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.train_patch_path['points'])

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        print(f"Loading data for index: {idx}")
        points_data = np.loadtxt(self.train_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.train_patch_path['points'][idx].split('/')[-1].split('.')[0]
        # image = sitk.ReadImage(self.train_patch_path['ct'][idx])
        # origin = image.GetOrigin()
        # spacing = image.GetSpacing()
        # direction = image.GetDirection()
        # image = sitk.GetArrayFromImage(image) # (slices,512,512)
        # image = window_level_processing(image)
        # D,H,W = image.shape

        # couinaud_mask = sitk.ReadImage(self.train_patch_path['couinaud'][idx])
        # couinaud_mask = sitk.GetArrayFromImage(couinaud_mask) # (slices,512,512)
        # couinaud_mask_onehot = np.eye(9)[couinaud_mask]  # (slices, 512, 512, 9)
        # couinaud_mask_onehot = couinaud_mask_onehot[:,:,:,1:] # (slices, 512, 512, 8)
        # couinaud_mask_onehot = couinaud_mask_onehot.reshape(-1,8) # (D*H*W,8)
        # liver_mask = np.zeros(couinaud_mask.shape)
        # liver_mask[liver_mask>0] = 1 # (slices,512,512)
        # liver_mask = liver_mask.reshape(-1,1) # (D*H*W,1)

        # liver = sitk.ReadImage(self.train_patch_path['ct'][idx].replace('images','annotation/liver'))
        # liver = sitk.GetArrayFromImage(liver)
        # image = image * liver
        # image[image<0] = 0

        N = points_data.shape[0]
        num_samples = int(0.1 * N)
        if num_samples>50000:
            num_samples = 50000
        random_indices = np.random.choice(N, num_samples, replace=False) 
        sampled_points = points_data[random_indices] # (N',5)

        labels = sampled_points[:,4:] # (N',1)
        stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)

        # labels_full = points_data[:,4:] # (N,1)
        # stacked_points_full = points_data[:,:3].astype(np.float32) # (N,3)

        'Visualize the sampled points'
        # self.viz(stacked_points, image, labels, origin, direction, spacing)

        'point data augmentation'
        # # Set the probability for augmentation
        # augmentation_probability = 0.5
        # # Perform augmentation with a 50% chance
        # if random.random() < augmentation_probability:
        #     augmented_stacked_points = self.augmentation_transform(
        #         normalize_to_neg1_pos1(sampled_points[:, :3])
        #     )[0].astype(np.float32)
        #     augmented_stacked_points = inverse_normalize_from_neg1_pos1(
        #         augmented_stacked_points, sampled_points[:, :3]
        #     )
        # else:
        #     # No augmentation, use the original points
        #     augmented_stacked_points = stacked_points

        'Visualize the augmented sampled points'
        # self.viz(augmented_stacked_points, image, labels, origin, direction, spacing)

        'Normalization'
        # stacked_points = nomalize(stacked_points)[0].astype(np.float32)
        # stacked_points = nomalize(augmented_stacked_points)[0].astype(np.float32)

        stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)
        stacked_features = window_level_processing(stacked_features)
        
        stack_lengths = np.array([num_samples]).astype(np.int32)

        seg_inputs = self.segmentation_inputs(stacked_points, 
                                 stacked_features,
                                 labels,
                                 stack_lengths)

        # seg_inputs += [labels_full]
        # seg_inputs += [stacked_points_full]
        # seg_inputs += [[D,H,W]]
        seg_inputs += [case_name]

        ########################
        # Load target image data
        ########################
        idx_2 = random.randint(0, len(self.train_patch_path['points'])-1) 
        print(f"Loading data for index: {idx_2}")
        points_data_target = np.loadtxt(self.train_patch_path['points'][idx_2], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name_target = self.train_patch_path['points'][idx_2].split('/')[-1].split('.')[0]

        couinaud_mask_target = sitk.ReadImage(self.train_patch_path['couinaud'][idx_2])
        couinaud_mask_target = sitk.GetArrayFromImage(couinaud_mask_target).astype(np.int32) # (slices,512,512)
        
        '## crop the Couinaud mask to liver region'
        liver = couinaud_mask_target.copy()
        liver[liver>0] = 1
        edges = np.argwhere(liver == 1)
        z_min, z_max = edges[:, 0].min(), edges[:, 0].max()
        y_min, y_max = edges[:, 1].min(), edges[:, 1].max()
        x_min, x_max = edges[:, 2].min(), edges[:, 2].max()
        couinaud_mask_target = couinaud_mask_target[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        couinaud_mask_target = np.rot90(couinaud_mask_target, k=2, axes=(1, 2))

        couinaud_mask_target = F.one_hot(torch.from_numpy(couinaud_mask_target.copy()).type(torch.LongTensor), 9) # (slices,512,512,9)
        couinaud_mask_target = couinaud_mask_target.numpy()

        N = points_data_target.shape[0]
        num_samples_target = int(0.1 * N)
        if num_samples_target>50000:
            num_samples_target = 50000
        random_indices_target = np.random.choice(N, num_samples_target, replace=False) 
        sampled_points_target = points_data_target[random_indices_target] # (N',5)

        labels_target = sampled_points_target[:,4:] # (N',1)
        stacked_points_target = sampled_points_target[:,:3].astype(np.float32) # (N',3)
        'Normalization'
        # stacked_points_target = nomalize(stacked_points_target)[0].astype(np.float32)
        # stacked_points_target = 2 * stacked_points_target - 1 

        stacked_features_target = sampled_points_target[:,3:4].astype(np.float32) # (N',1)
        stacked_features_target = window_level_processing(stacked_features_target)
        
        stack_lengths_target = np.array([num_samples_target]).astype(np.int32)

        seg_inputs_target = self.segmentation_inputs(stacked_points_target, 
                                 stacked_features_target,
                                 labels_target,
                                 stack_lengths_target)

        seg_inputs_target += [couinaud_mask_target]
        seg_inputs_target += [case_name_target]

        return tuple((seg_inputs, seg_inputs_target))

    def viz(self, stacked_points, image, labels, origin, direction, spacing):
        # Step 1: Subtract Origin
        adjusted_coords = stacked_points - origin  # (N, 3)
        # Step 2: Account for Direction
        direction = np.array(direction).reshape((3, 3))
        inv_direction = np.linalg.inv(direction)  # Invert the direction matrix
        direction_corrected_coords = np.dot(adjusted_coords, inv_direction)  # (N, 3)
        # Step 3: Divide by Spacing
        voxel_coords = direction_corrected_coords / spacing  # (N, 3)
        # Step 4: Round and Clip
        voxel_coords = np.round(voxel_coords).astype(int)
        voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, image.shape[2] - 1)  # x-axis
        voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, image.shape[1] - 1)  # y-axis
        voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, image.shape[0] - 1)  # z-axis
        overlay_volume = np.zeros_like(image)
        for i, (x, y, z) in enumerate(voxel_coords):
            overlay_volume[z, y, x] = labels[i]
        # Extract the coordinates of non-zero points in overlay_volume
        z, y, x = np.nonzero(overlay_volume)
        values = overlay_volume[z, y, x]  # The corresponding label values
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot
        scatter = ax.scatter(x, y, z, c=values, cmap='jet', marker='o', s=5)
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Label Values')
        # Label axes
        ax.set_xlabel("X-axis (voxels)")
        ax.set_ylabel("Y-axis (voxels)")
        ax.set_zlabel("Z-axis (voxels)")
        plt.title("3D Overlay Volume Visualization")
        plt.show()

    def init_labels(self):
        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.ignored_labels = []
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random angle in [-5, 5] degrees

                # Create random rotations
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians

                # Generate the first vector in polar coordinates
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random theta in [-5, 5] degrees
                phi = (np.random.rand() * 2 * angle_range) - angle_range  # Random phi in [-5, 5] degrees

                # Create the first vector in Cartesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle within the range
                alpha = (np.random.rand() * 2 * angle_range) - angle_range  # Random alpha in [-5, 5] degrees

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

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
                            labels,
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
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths # [[]*5,[]*5,[]*5,[]*5,[]*5]==[[]*25]
        li += [stacked_features, labels] # [[]*27]

        return li

class Dataset_MSD_val_HCC(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, config, name=None):
        """
        Initialize parameters of the dataset here.
        """
        self.name = name
        self.config = config
        self.neighborhood_limits = []
        self.validation_labels = []
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

        point_data_root = os.path.join(config.data_root,'Points_in_Liver_Norm_5mm') 

        val_list = np.loadtxt(os.path.join(config.data_root,'val_MSD_new.txt'), dtype=str)
        
        val_set = []
        for item in val_list:
            val_set.append(int(re.findall(r"\d+",item)[0]))

        val_patch_path = {'ct':[],'points':[],'couinaud':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) in val_set:
                    val_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver_Norm_5mm','Vol_resize_CT_5mm'), name.replace('.txt','.nii.gz')))
                    val_patch_path['points'].append(os.path.join(root,name))
                    val_patch_path['couinaud'].append(os.path.join(root.replace('Points_in_Liver_Norm_5mm','Vol_resize_couinaud_5mm'), name.replace('.txt','.nii.gz')))
        val_patch_path['ct'], val_patch_path['points'], val_patch_path['couinaud'] = sorted(val_patch_path['ct']), sorted(val_patch_path['points']), sorted(val_patch_path['couinaud'])
        self.val_patch_path = val_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.val_patch_path['points'])

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        points_data = np.loadtxt(self.val_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.val_patch_path['points'][idx].split('/')[-1].split('.')[0]
        # image = sitk.ReadImage(self.val_patch_path['ct'][idx])
        # origin = image.GetOrigin()
        # spacing = image.GetSpacing()
        # direction = image.GetDirection()
        # image = sitk.GetArrayFromImage(image) # (slices,512,512)
        # D,H,W = image.shape

        # couinaud_mask = sitk.ReadImage(self.train_patch_path['couinaud'][idx])
        # couinaud_mask = sitk.GetArrayFromImage(couinaud_mask) # (slices,512,512)
        # couinaud_mask_onehot = np.eye(9)[couinaud_mask]  # (slices, 512, 512, 9)
        # couinaud_mask_onehot = couinaud_mask_onehot[:,:,:,1:] # (slices, 512, 512, 8)
        # couinaud_mask_onehot = couinaud_mask_onehot.reshape(-1,8) # (D*H*W,8)
        # liver_mask = np.zeros(couinaud_mask.shape)
        # liver_mask[liver_mask>0] = 1 # (slices,512,512)
        # liver_mask = liver_mask.reshape(-1,1) # (D*H*W,1)

        N = points_data.shape[0]
        num_samples = int(0.1 * N)
        if num_samples>50000:
            num_samples = 50000
        random_indices = np.random.choice(N, num_samples, replace=False) 
        sampled_points = points_data[random_indices] # (N',5)

        stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)
        'Normalization'
        # stacked_points = nomalize(stacked_points)[0].astype(np.float32)
        
        stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)
        stacked_features = window_level_processing(stacked_features)
        labels = sampled_points[:,4:] # (N',1)
        stack_lengths = np.array([num_samples]).astype(np.int32)

        # labels_full = points_data[:,4:] # (N,1)
        # stacked_points_full = points_data[:,:3].astype(np.float32) # (N,3)

        seg_inputs = self.segmentation_inputs(stacked_points, 
                                 stacked_features,
                                 labels,
                                 stack_lengths)

        # seg_inputs += [labels_full]
        # seg_inputs += [stacked_points_full]
        # seg_inputs += [[D,H,W]]
        seg_inputs += [case_name]

        ########################
        # Load target image data
        ########################
        idx_2 = random.randint(0, len(self.val_patch_path['points'])-1) 
        print(f"Loading data for index: {idx_2}")
        points_data_target = np.loadtxt(self.val_patch_path['points'][idx_2], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name_target = self.val_patch_path['points'][idx_2].split('/')[-1].split('.')[0]

        image_target = sitk.ReadImage(self.val_patch_path['ct'][idx_2])
        origin_target = image_target.GetOrigin()
        spacing_target = image_target.GetSpacing()
        direction_target = image_target.GetDirection()
        image_target = sitk.GetArrayFromImage(image_target) # (slices,512,512)
        image_target = window_level_processing(image_target)
        D_target,H_target,W_target = image_target.shape

        couinaud_mask_target = sitk.ReadImage(self.val_patch_path['couinaud'][idx_2])
        couinaud_mask_target = sitk.GetArrayFromImage(couinaud_mask_target).astype(np.int32) # (slices,512,512)

        '## crop the Couinaud mask to liver region'
        liver = couinaud_mask_target.copy()
        liver[liver>0] = 1
        edges = np.argwhere(liver == 1)
        z_min, z_max = edges[:, 0].min(), edges[:, 0].max()
        y_min, y_max = edges[:, 1].min(), edges[:, 1].max()
        x_min, x_max = edges[:, 2].min(), edges[:, 2].max()
        couinaud_mask_target = couinaud_mask_target[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        couinaud_mask_target = np.rot90(couinaud_mask_target, k=2, axes=(1, 2))

        couinaud_mask_target = F.one_hot(torch.from_numpy(couinaud_mask_target.copy()).type(torch.LongTensor), 9) # (slices,512,512,9)
        couinaud_mask_target = couinaud_mask_target.numpy()

        N = points_data_target.shape[0]
        num_samples_target = int(0.1 * N)
        if num_samples_target>50000:
            num_samples_target = 50000
        random_indices_target = np.random.choice(N, num_samples_target, replace=False) 
        sampled_points_target = points_data_target[random_indices_target] # (N',5)

        stacked_points_target = sampled_points_target[:,:3].astype(np.float32) # (N',3)
        'Normalization'
        # stacked_points_target = nomalize(stacked_points_target, spacing_target, origin_target, direction_target, D_target, H_target, W_target)[0].astype(np.float32)
        # stacked_points_target = 2 * stacked_points_target - 1 

        stacked_features_target = sampled_points_target[:,3:4].astype(np.float32) # (N',1)
        stacked_features_target = window_level_processing(stacked_features_target)
        labels_target = sampled_points_target[:,4:] # (N',1)
        stack_lengths_target = np.array([num_samples_target]).astype(np.int32)

        seg_inputs_target = self.segmentation_inputs(stacked_points_target, 
                                 stacked_features_target,
                                 labels_target,
                                 stack_lengths_target)

        seg_inputs_target += [couinaud_mask_target]
        seg_inputs_target += [case_name_target]
        return tuple((seg_inputs, seg_inputs_target))

    def init_labels(self):

        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.ignored_labels = []
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random angle in [-5, 5] degrees

                # Create random rotations
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians

                # Generate the first vector in polar coordinates
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random theta in [-5, 5] degrees
                phi = (np.random.rand() * 2 * angle_range) - angle_range  # Random phi in [-5, 5] degrees

                # Create the first vector in Cartesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle within the range
                alpha = (np.random.rand() * 2 * angle_range) - angle_range  # Random alpha in [-5, 5] degrees

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

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
                            labels,
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
        li += [stacked_features, labels]

        return li


class Dataset_LiTS_train_HCC(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, config, name=None):
        """
        Initialize parameters of the dataset here.
        """
        self.name = name
        self.config = config
        self.neighborhood_limits = []
        self.training_labels = []
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

        point_data_root = os.path.join(config.data_root,'Points_in_Liver_Norm_5mm') 

        train_list = np.loadtxt(os.path.join(config.data_root,'train_LiTS.txt'), dtype=str)
        
        train_set = []
        for item in train_list:
            train_set.append(int(re.findall(r"\d+",item)[0]))

        train_patch_path = {'ct':[],'points':[],'couinaud':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) in train_set:
                    train_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver_Norm_5mm','Vol_LPI_resize_CT_5mm'), 'volume-'+str(int(re.findall(r"\d+",name)[0]))+'.nii'))
                    train_patch_path['points'].append(os.path.join(root,name))
                    train_patch_path['couinaud'].append(os.path.join(root.replace('Points_in_Liver_Norm_5mm','Vol_LPI_resize_couinaud_5mm'), 'LiTS-'+str(int(re.findall(r"\d+",name)[0]))+'.nii.gz'))

        train_patch_path['ct'], train_patch_path['points'], train_patch_path['couinaud'] = sorted(train_patch_path['ct']), sorted(train_patch_path['points']), sorted(train_patch_path['couinaud'])
        self.train_patch_path = train_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.train_patch_path['points'])

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        ########################
        # Load source image data
        ########################
        print(f"Loading data for index: {idx}")
        points_data = np.loadtxt(self.train_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.train_patch_path['points'][idx].split('/')[-1].split('.')[0]
        # image = sitk.ReadImage(self.train_patch_path['ct'][idx])
        # origin = image.GetOrigin()
        # spacing = image.GetSpacing()
        # direction = image.GetDirection()
        # image = sitk.GetArrayFromImage(image) # (slices,512,512)
        # image = window_level_processing(image)
        # D,H,W = image.shape

        # couinaud_mask = sitk.ReadImage(self.train_patch_path['couinaud'][idx])
        # couinaud_mask = sitk.GetArrayFromImage(couinaud_mask) # (slices,512,512)
        # couinaud_mask_onehot = np.eye(9)[couinaud_mask]  # (slices, 512, 512, 9)
        # couinaud_mask_onehot = couinaud_mask_onehot[:,:,:,1:] # (slices, 512, 512, 8)
        # couinaud_mask_onehot = couinaud_mask_onehot.reshape(-1,8) # (D*H*W,8)
        # liver_mask = np.zeros(couinaud_mask.shape)
        # liver_mask[liver_mask>0] = 1 # (slices,512,512)
        # liver_mask = liver_mask.reshape(-1,1) # (D*H*W,1)


        # liver = sitk.ReadImage(self.train_patch_path['ct'][idx].replace('images_LPI','annotation_LPI/liver').replace('volume-','LiTS-').replace('.nii','.nii.gz'))
        # liver = sitk.GetArrayFromImage(liver)
        # image = image * liver
        # image[image<0] = 0

        N = points_data.shape[0]
        num_samples = int(0.1 * N)
        if num_samples>50000:
            num_samples = 50000
        random_indices = np.random.choice(N, num_samples, replace=False) 
        sampled_points = points_data[random_indices] # (N',5)

        labels = sampled_points[:,4:] # (N',1)
        stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)

        # labels_full = points_data[:,4:] # (N,1)
        # stacked_points_full = points_data[:,:3].astype(np.float32) # (N,3)

        'Visualize the sampled points'
        # self.viz(stacked_points, image, labels, origin, direction, spacing)

        'point data augmentation'
        # # Set the probability for augmentation
        # augmentation_probability = 0.5
        # # Perform augmentation with a 50% chance
        # if random.random() < augmentation_probability:
        #     augmented_stacked_points = self.augmentation_transform(
        #         normalize_to_neg1_pos1(sampled_points[:, :3])
        #     )[0].astype(np.float32)
        #     augmented_stacked_points = inverse_normalize_from_neg1_pos1(
        #         augmented_stacked_points, sampled_points[:, :3]
        #     )
        # else:
        #     # No augmentation, use the original points
        #     augmented_stacked_points = stacked_points

        'Visualize the augmented sampled points'
        # self.viz(augmented_stacked_points, image, labels, origin, direction, spacing)

        'Normalization'
        # stacked_points = nomalize(stacked_points)[0].astype(np.float32)
        # stacked_points = nomalize(augmented_stacked_points)[0].astype(np.float32)
        # stacked_points = 2 * stacked_points - 1 

        stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)
        stacked_features = window_level_processing(stacked_features)
        
        stack_lengths = np.array([num_samples]).astype(np.int32)

        seg_inputs = self.segmentation_inputs(stacked_points, 
                                 stacked_features,
                                 labels,
                                 stack_lengths)

        # seg_inputs += [labels_full]
        # seg_inputs += [stacked_points_full]
        # seg_inputs += [[D,H,W]]
        seg_inputs += [case_name]

        ########################
        # Load target image data
        ########################
        idx_2 = random.randint(0, len(self.train_patch_path['points'])-1) 
        print(f"Loading data for index: {idx_2}")
        points_data_target = np.loadtxt(self.train_patch_path['points'][idx_2], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name_target = self.train_patch_path['points'][idx_2].split('/')[-1].split('.')[0]

        couinaud_mask_target = sitk.ReadImage(self.train_patch_path['couinaud'][idx_2])
        couinaud_mask_target = sitk.GetArrayFromImage(couinaud_mask_target).astype(np.int32) # (slices,512,512)
        
        '## crop the Couinaud mask to liver region'
        liver = couinaud_mask_target.copy()
        liver[liver>0] = 1
        edges = np.argwhere(liver == 1)
        z_min, z_max = edges[:, 0].min(), edges[:, 0].max()
        y_min, y_max = edges[:, 1].min(), edges[:, 1].max()
        x_min, x_max = edges[:, 2].min(), edges[:, 2].max()
        couinaud_mask_target = couinaud_mask_target[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        couinaud_mask_target = np.rot90(couinaud_mask_target, k=2, axes=(1, 2))

        couinaud_mask_target = F.one_hot(torch.from_numpy(couinaud_mask_target.copy()).type(torch.LongTensor), 9) # (slices,512,512,9)
        couinaud_mask_target = couinaud_mask_target.numpy()

        N = points_data_target.shape[0]
        num_samples_target = int(0.1 * N)
        if num_samples_target>50000:
            num_samples_target = 50000
        random_indices_target = np.random.choice(N, num_samples_target, replace=False) 
        sampled_points_target = points_data_target[random_indices_target] # (N',5)

        labels_target = sampled_points_target[:,4:] # (N',1)
        stacked_points_target = sampled_points_target[:,:3].astype(np.float32) # (N',3)
        'Normalization'
        # stacked_points_target = nomalize(stacked_points_target)[0].astype(np.float32)
        # stacked_points_target = 2 * stacked_points_target - 1 

        stacked_features_target = sampled_points_target[:,3:4].astype(np.float32) # (N',1)
        stacked_features_target = window_level_processing(stacked_features_target)
        
        stack_lengths_target = np.array([num_samples_target]).astype(np.int32)

        seg_inputs_target = self.segmentation_inputs(stacked_points_target, 
                                 stacked_features_target,
                                 labels_target,
                                 stack_lengths_target)

        seg_inputs_target += [couinaud_mask_target]
        seg_inputs_target += [case_name_target]

        return tuple((seg_inputs, seg_inputs_target))

    def viz(self, stacked_points, image, labels, origin, direction, spacing):
        # Step 1: Subtract Origin
        adjusted_coords = stacked_points - origin  # (N, 3)
        # Step 2: Account for Direction
        direction = np.array(direction).reshape((3, 3))
        inv_direction = np.linalg.inv(direction)  # Invert the direction matrix
        direction_corrected_coords = np.dot(adjusted_coords, inv_direction)  # (N, 3)
        # Step 3: Divide by Spacing
        voxel_coords = direction_corrected_coords / spacing  # (N, 3)
        # Step 4: Round and Clip
        voxel_coords = np.round(voxel_coords).astype(int)
        voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, image.shape[2] - 1)  # x-axis
        voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, image.shape[1] - 1)  # y-axis
        voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, image.shape[0] - 1)  # z-axis
        overlay_volume = np.zeros_like(image)
        for i, (x, y, z) in enumerate(voxel_coords):
            overlay_volume[z, y, x] = labels[i]
        # Extract the coordinates of non-zero points in overlay_volume
        z, y, x = np.nonzero(overlay_volume)
        values = overlay_volume[z, y, x]  # The corresponding label values
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot
        scatter = ax.scatter(x, y, z, c=values, cmap='jet', marker='o', s=5)
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Label Values')
        # Label axes
        ax.set_xlabel("X-axis (voxels)")
        ax.set_ylabel("Y-axis (voxels)")
        ax.set_zlabel("Z-axis (voxels)")
        plt.title("3D Overlay Volume Visualization")
        plt.show()

    def init_labels(self):
        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.ignored_labels = []
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random angle in [-5, 5] degrees

                # Create random rotations
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians

                # Generate the first vector in polar coordinates
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random theta in [-5, 5] degrees
                phi = (np.random.rand() * 2 * angle_range) - angle_range  # Random phi in [-5, 5] degrees

                # Create the first vector in Cartesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle within the range
                alpha = (np.random.rand() * 2 * angle_range) - angle_range  # Random alpha in [-5, 5] degrees

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

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
                            labels,
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
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths # [[]*5,[]*5,[]*5,[]*5,[]*5]==[[]*25]
        li += [stacked_features, labels] # [[]*27]

        return li

class Dataset_LiTS_val_HCC(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, config, name=None):
        """
        Initialize parameters of the dataset here.
        """
        self.name = name
        self.config = config
        self.neighborhood_limits = []
        self.validation_labels = []
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

        point_data_root = os.path.join(config.data_root,'Points_in_Liver_Norm_5mm') 

        val_list = np.loadtxt(os.path.join(config.data_root,'val_LiTS.txt'), dtype=str)
        
        val_set = []
        for item in val_list:
            val_set.append(int(re.findall(r"\d+",item)[0]))

        val_patch_path = {'ct':[],'points':[],'couinaud':[]}
        for root, dirs, files in os.walk(point_data_root): 
            for name in files:
                if int(re.findall(r"\d+",name)[0]) in val_set:
                    val_patch_path['ct'].append(os.path.join(root.replace('Points_in_Liver_Norm_5mm','Vol_LPI_resize_CT_5mm'), 'volume-'+str(int(re.findall(r"\d+",name)[0]))+'.nii'))
                    val_patch_path['points'].append(os.path.join(root,name))
                    val_patch_path['couinaud'].append(os.path.join(root.replace('Points_in_Liver_Norm_5mm','Vol_LPI_resize_couinaud_5mm'), 'LiTS-'+str(int(re.findall(r"\d+",name)[0]))+'.nii.gz'))
        val_patch_path['ct'], val_patch_path['points'], val_patch_path['couinaud'] = sorted(val_patch_path['ct']), sorted(val_patch_path['points']), sorted(val_patch_path['couinaud'])
        self.val_patch_path = val_patch_path

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.val_patch_path['points'])

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """
        ########################
        # Load source image data
        ########################
        points_data = np.loadtxt(self.val_patch_path['points'][idx], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name = self.val_patch_path['points'][idx].split('/')[-1].split('.')[0]
        image = sitk.ReadImage(self.val_patch_path['ct'][idx])
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        image = sitk.GetArrayFromImage(image) # (slices,512,512)
        D,H,W = image.shape

        # couinaud_mask = sitk.ReadImage(self.val_patch_path['couinaud'][idx])
        # couinaud_mask = sitk.GetArrayFromImage(couinaud_mask) # (slices,512,512)
        # couinaud_mask_onehot = np.eye(9)[couinaud_mask]  # (slices, 512, 512, 9)
        # couinaud_mask_onehot = couinaud_mask_onehot[:,:,:,1:] # (slices, 512, 512, 8)
        # couinaud_mask_onehot = couinaud_mask_onehot.reshape(-1,8) # (D*H*W,8)
        # liver_mask = np.zeros(couinaud_mask.shape)
        # liver_mask[liver_mask>0] = 1 # (slices,512,512)
        # liver_mask = liver_mask.reshape(-1,1) # (D*H*W,1)

        N = points_data.shape[0]
        num_samples = int(0.1 * N)
        if num_samples>50000:
            num_samples = 50000
        random_indices = np.random.choice(N, num_samples, replace=False) 
        sampled_points = points_data[random_indices] # (N',5)

        stacked_points = sampled_points[:,:3].astype(np.float32) # (N',3)
        'Normalization'
        # stacked_points = nomalize(stacked_points, spacing, origin, direction, D, H, W)[0].astype(np.float32)
        # stacked_points = 2 * stacked_points - 1 

        stacked_features = sampled_points[:,3:4].astype(np.float32) # (N',1)
        stacked_features = window_level_processing(stacked_features)
        labels = sampled_points[:,4:] # (N',1)
        stack_lengths = np.array([num_samples]).astype(np.int32)

        # labels_full = points_data[:,4:] # (N,1)
        # stacked_points_full = points_data[:,:3].astype(np.float32) # (N,3)

        seg_inputs = self.segmentation_inputs(stacked_points, 
                                 stacked_features,
                                 labels,
                                 stack_lengths)

        # seg_inputs += [labels_full]
        # seg_inputs += [stacked_points_full]
        # seg_inputs += [[D,H,W]]
        seg_inputs += [case_name]

        ########################
        # Load target image data
        ########################
        idx_2 = random.randint(0, len(self.val_patch_path['points'])-1) 
        print(f"Loading data for index: {idx_2}")
        points_data_target = np.loadtxt(self.val_patch_path['points'][idx_2], dtype=float, delimiter=",", skiprows=0, usecols=None,
                                unpack=False) # (N,5)
        case_name_target = self.val_patch_path['points'][idx_2].split('/')[-1].split('.')[0]

        image_target = sitk.ReadImage(self.val_patch_path['ct'][idx_2])
        origin_target = image_target.GetOrigin()
        spacing_target = image_target.GetSpacing()
        direction_target = image_target.GetDirection()
        image_target = sitk.GetArrayFromImage(image_target) # (slices,512,512)
        image_target = window_level_processing(image_target)
        D_target,H_target,W_target = image_target.shape

        couinaud_mask_target = sitk.ReadImage(self.val_patch_path['couinaud'][idx_2])
        couinaud_mask_target = sitk.GetArrayFromImage(couinaud_mask_target).astype(np.int32) # (slices,512,512)

        '## crop the Couinaud mask to liver region'
        liver = couinaud_mask_target.copy()
        liver[liver>0] = 1
        edges = np.argwhere(liver == 1)
        z_min, z_max = edges[:, 0].min(), edges[:, 0].max()
        y_min, y_max = edges[:, 1].min(), edges[:, 1].max()
        x_min, x_max = edges[:, 2].min(), edges[:, 2].max()
        couinaud_mask_target = couinaud_mask_target[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        couinaud_mask_target = np.rot90(couinaud_mask_target, k=2, axes=(1, 2))

        couinaud_mask_target = F.one_hot(torch.from_numpy(couinaud_mask_target.copy()).type(torch.LongTensor), 9) # (slices,512,512,9)
        couinaud_mask_target = couinaud_mask_target.numpy()

        N = points_data_target.shape[0]
        num_samples_target = int(0.1 * N)
        if num_samples_target>50000:
            num_samples_target = 50000
        random_indices_target = np.random.choice(N, num_samples_target, replace=False) 
        sampled_points_target = points_data_target[random_indices_target] # (N',5)

        stacked_points_target = sampled_points_target[:,:3].astype(np.float32) # (N',3)
        'Normalization'
        # stacked_points_target = nomalize(stacked_points_target, spacing_target, origin_target, direction_target, D_target, H_target, W_target)[0].astype(np.float32)
        # stacked_points_target = 2 * stacked_points_target - 1 

        stacked_features_target = sampled_points_target[:,3:4].astype(np.float32) # (N',1)
        stacked_features_target = window_level_processing(stacked_features_target)
        labels_target = sampled_points_target[:,4:] # (N',1)
        stack_lengths_target = np.array([num_samples_target]).astype(np.int32)

        seg_inputs_target = self.segmentation_inputs(stacked_points_target, 
                                 stacked_features_target,
                                 labels_target,
                                 stack_lengths_target)

        seg_inputs_target += [couinaud_mask_target]
        seg_inputs_target += [case_name_target]
        return tuple((seg_inputs, seg_inputs_target))
    
    def init_labels(self):

        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.ignored_labels = []
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random angle in [-5, 5] degrees

                # Create random rotations
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Define rotation angle range in radians (-5 to 5 degrees)
                angle_range = np.deg2rad(self.config.augment_rotation_range)  # Convert 5 degrees to radians

                # Generate the first vector in polar coordinates
                theta = (np.random.rand() * 2 * angle_range) - angle_range  # Random theta in [-5, 5] degrees
                phi = (np.random.rand() * 2 * angle_range) - angle_range  # Random phi in [-5, 5] degrees

                # Create the first vector in Cartesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle within the range
                alpha = (np.random.rand() * 2 * angle_range) - angle_range  # Random alpha in [-5, 5] degrees

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

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
                            labels,
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
        li += [stacked_features, labels]

        return li

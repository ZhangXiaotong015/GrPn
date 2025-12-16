import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import torch
from skimage.measure import label as la
import re
import shutil

def dir_transform(data_root=None, LPI_data_save_root=None, liver_root=None, LPI_liver_save_root=None): # transform the original direction to LPI

    def convert_to_lpi(image, current_orientation):
        """
        Converts an image from RPI or RAI orientation to LPI.

        Parameters:
            image (sitk.Image): The input image in RPI or RAI orientation.
            current_orientation (str): The current orientation of the image ("RPI" or "RAI").
        
        Returns:
            sitk.Image: The reoriented image in LPI orientation.
        """
        if current_orientation not in ["RPI", "RAI"]:
            raise ValueError("Unsupported orientation. Only 'RPI' and 'RAI' are supported.")
        
        # Determine which axes to flip based on the current orientation
        if current_orientation == "RPI":
            flip_axes = [True, False, False]  # Flip only the X-axis
        elif current_orientation == "RAI":
            flip_axes = [True, True, False]   # Flip X-axis and Z-axis
        
        # Flip the image along the specified axes
        flipped_image = sitk.Flip(image, flip_axes)
        
        # Adjust the origin to account for the flipping
        origin = list(flipped_image.GetOrigin())
        size = flipped_image.GetSize()
        spacing = flipped_image.GetSpacing()
        
        if flip_axes[0]:  # Adjust origin for X-axis flip
            origin[0] += (size[0] - 1) * spacing[0]
        if flip_axes[2]:  # Adjust origin for Z-axis flip
            origin[2] += (size[2] - 1) * spacing[2]
        
        flipped_image.SetOrigin(tuple(origin))
        
        # The direction matrix is already updated by sitk.Flip, so no further adjustments needed.
        return flipped_image

    LPI_dir = (-1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0)
    RPI_dir = (1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0)
    RAI_dir = (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)

    hflip_cases = list(np.arange(28,48))

    for root, dirs, files in os.walk(liver_root): 
        for name in files:
            liver = sitk.ReadImage(os.path.join(root, name))
            origin = liver.GetOrigin()
            spacing = liver.GetSpacing()
            direction = liver.GetDirection()

            if direction==RPI_dir:
                
                current_orientation = 'RPI'

                reoriented_liver = convert_to_lpi(liver, current_orientation)
                os.makedirs(LPI_liver_save_root, exist_ok=True)
                output_path = os.path.join(LPI_liver_save_root, name)
                sitk.WriteImage(reoriented_liver, output_path)

            elif direction==RAI_dir:

                current_orientation = 'RAI'

                reoriented_liver = convert_to_lpi(liver, current_orientation)
                os.makedirs(LPI_liver_save_root, exist_ok=True)
                output_path = os.path.join(LPI_liver_save_root, name)
                sitk.WriteImage(reoriented_liver, output_path)



    for root, dirs, files in os.walk(data_root): 
        for name in files:
            image = sitk.ReadImage(os.path.join(root, name))
            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            direction = image.GetDirection()

            if direction==RPI_dir:

                current_orientation = 'RPI'

                reoriented_image = convert_to_lpi(image, current_orientation)
                os.makedirs(LPI_data_save_root, exist_ok=True)
                output_path = os.path.join(LPI_data_save_root, name)  
                sitk.WriteImage(reoriented_image, output_path)

            elif direction==RAI_dir:

                current_orientation = 'RAI'

                reoriented_image = convert_to_lpi(image, current_orientation)
                os.makedirs(LPI_data_save_root, exist_ok=True)
                output_path = os.path.join(LPI_data_save_root, name)  
                sitk.WriteImage(reoriented_image, output_path)



def convert2point(data_root=None, liver_root=None, save_root=None, save_root_Voxelidx=None):

    # for root, dirs, files in os.walk(data_root): 
    for name, liver_name in zip(sorted(os.listdir(data_root)), sorted(os.listdir(liver_root))):
        image = sitk.ReadImage(os.path.join(data_root, name))
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        image = sitk.GetArrayFromImage(image) # (slices,512,512)
        print(spacing)

        liver = sitk.ReadImage(os.path.join(liver_root, liver_name))
        liver = sitk.GetArrayFromImage(liver)

        loc_img, num = la(liver, background=0, return_num=True, connectivity=2)
        max_label = 0
        max_num = 0
        for i in range(1, num + 1):
            if np.sum(loc_img == i) > max_num:
                max_num = np.sum(loc_img == i)
                max_label = i
        mcr = (loc_img == max_label)
        mcr = mcr + 0
        z_true, y_true, x_true = np.where(mcr)
        box = np.array([[np.min(z_true), np.max(z_true)], [np.min(y_true), np.max(y_true)],
                        [np.min(x_true), np.max(x_true)]])

        z_min, z_max = box[0]
        y_min, y_max = box[1]
        x_min, x_max = box[2]

        Spacing_arr = np.array(spacing)
        Origin_arr = np.array(origin)
        Direction_arr = np.array(direction).reshape((3, 3))

        print(image.shape)

        point_list = [[x * Spacing_arr[0]*Direction_arr[0,0] + Origin_arr[0], 
                    y * Spacing_arr[1]*Direction_arr[1,1] + Origin_arr[1], 
                    z * Spacing_arr[2]*Direction_arr[2,2] + Origin_arr[2], 
                    image[z][y][x]
                    ]  for x in range(x_min,x_max) for y in range(y_min,y_max) for z in range(z_min,z_max) if
            (liver[z][y][x] != 0)]

        point_list_Voxelidx = [[x, y, z, 
                    image[z][y][x]
                    ]  for x in range(x_min,x_max) for y in range(y_min,y_max) for z in range(z_min,z_max) if
            (liver[z][y][x] != 0)]

        # print(len(point_list))
        os.makedirs(save_root, exist_ok=True)
        df = pd.DataFrame(point_list, columns=None)
        df.to_csv(os.path.join(save_root, name.split('.')[0] + ".txt"), header=None,
                    index=False)
        # print(len(point_list_Voxelidx))
        os.makedirs(save_root_Voxelidx, exist_ok=True)
        df = pd.DataFrame(point_list_Voxelidx, columns=None)
        df.to_csv(os.path.join(save_root_Voxelidx, name.split('.')[0] + ".txt"), header=None,
                    index=False)
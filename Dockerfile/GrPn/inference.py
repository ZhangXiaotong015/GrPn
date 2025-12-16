
# Common libs
import signal
import os
import numpy as np
import sys
import re
from scipy.ndimage import zoom
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label
import torch
import torch.nn as nn
from os import makedirs, listdir
from os.path import exists, join
import time
import json
import argparse
import pandas as pd
import SimpleITK as sitk
# from calflops import calculate_flops

# Dataset
# from datasets.S3DIS import *
from datasets.dataset import Dataset_test
from torch.utils.data import DataLoader

from utils.config import Config
from models.architectures import Net

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion, metrics
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KDTree


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='', help='Test name')
parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
parser.add_argument('--model', type=str, default=os.getenv("MODEL_PATH","/app/model_weights/chkp_MSD_median_interplanar_1mm.tar"), help='Chosen model.')
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--data_CT', type=str, default='')
parser.add_argument('--data_liver', type=str, default='')
parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--data_root_voxelidx', type=str, default='')
parser.add_argument('--LPI_volume_dir', type=str, default='')
parser.add_argument('--out_dir', type=str, default='')
parser.add_argument('--first_subsampling_dl', type=float, default=-1, help='Size of the first subsampling grid')
parser.add_argument('--fea_size', nargs='+', type=int, required=True, help='Feature sizes required by DGMN')
parser.add_argument('--voxel_resolution', nargs='+', type=int, required=True, help='Voxel resolutions required by re-voxelization and de-voxelization')

args = parser.parse_args()

# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class Config(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    name = ''

    # Dataset name
    dataset = ''
    data_root = ''
    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None
    batch_size = 1
    label_values = [1,2,3,4,5,6,7,8]
    ignored_labels = []

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 4

    # DGMN parameters
    embed_dims = [64,128,256,512]
    num_heads = [1,2,4,8]
    # fea_size = [32,16,8,4] # points_grid_sample_unit: 0.015625
    # fea_size = [64,32,16,8] # points_grid_sample_unit: 0.0078125
    fea_size = []
    mlp_ratios = [4,4,4,4]
    qkv_bias = False
    qk_scale = None
    drop_rate = 0
    attn_drop_rate = 0
    drop_path = 0
    norm_layer = nn.LayerNorm

    # Re-voxelization parameters
    # voxel_resolution = [32,16,8,4] # points_grid_sample_unit: 0.015625
    # voxel_resolution = [64,32,16,8] # points_grid_sample_unit: 0.0078125
    voxel_resolution = []
    normalize = True
    eps = 0

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['adapt_simple',
                    'adapt_resnetb',
                    'revoxelization',
                    'dgmn3d',
                    'devoxelization',
                    'adapt_resnetb_strided',
                    'adapt_resnetb',
                    'adapt_resnetb',
                    'revoxelization',
                    'dgmn3d',
                    'devoxelization',
                    'adapt_resnetb_strided',
                    'adapt_resnetb',
                    'adapt_resnetb',
                    'revoxelization',
                    'dgmn3d',
                    'devoxelization',
                    'adapt_resnetb_strided',
                    'adapt_resnetb',
                    'adapt_resnetb',
                    'revoxelization',
                    'dgmn3d',
                    'devoxelization',
                    'adapt_resnetb_strided',
                    'adapt_resnetb',
                    'adapt_resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    # convolutional feature
    adaptive_feature = 'xyz'
    first_adaptive_feature = 'xyz_joint'

    # Radius of the input sphere
    # in_radius = 1.5

    # Size of the first subsampling grid in meter
    # first_subsampling_dl = 0.03
    # first_subsampling_dl = 0.015625
    # first_subsampling_dl = 0.0078125
    first_subsampling_dl = -1

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Choice of input features
    first_features_dim = 64 #128
    in_features_dim = 1

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 200

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 50) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 6

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 200

    # Number of epoch between each checkpoint
    checkpoint_gap = 1 #10

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [False, False, False] # (x,y,z)
    augment_rotation = 'vertical'
    augment_rotation_range = 5 # [-5,5] degs
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    # augment_color = 0.8

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = ''
    saving_ply = False

class BatchData:
    def __init__(self, points, neighbors, pools, upsamples, lengths, features, case_name, points_Voxelidx):
        self.points = points
        self.neighbors = neighbors
        self.pools = pools
        self.upsamples = upsamples
        self.lengths = lengths
        self.features = features
        self.case_name = case_name
        self.points_Voxelidx = points_Voxelidx

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)

        return self

def collate_test(batch):
    L = (len(batch[0][0]) - 3) // 5
    batch_list = []
    for i in range(len(batch[0])):
        ind = 0
        points = [torch.from_numpy(nparray) for nparray in batch[0][i][ind:ind+L]]
        ind += L
        neighbors = [torch.from_numpy(nparray) for nparray in batch[0][i][ind:ind+L]]
        ind += L
        pools = [torch.from_numpy(nparray) for nparray in batch[0][i][ind:ind+L]]
        ind += L
        upsamples = [torch.from_numpy(nparray) for nparray in batch[0][i][ind:ind+L]]
        ind += L
        lengths = [torch.from_numpy(nparray) for nparray in batch[0][i][ind:ind+L]]
        ind += L
        features = torch.from_numpy(batch[0][i][ind])
        ind += 1
        case_name = batch[0][i][ind]
        ind += 1
        points_Voxelidx = torch.from_numpy(batch[0][i][ind]).type(torch.int32)
        batch_list.append(BatchData(points, neighbors, pools, upsamples, lengths, features, case_name, points_Voxelidx))
    return batch_list

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#
def infer():

    ###############################
    # Choose the model to visualize
    ###############################

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    ############################
    # Initialize the environment
    ############################

    ###############
    # Previous chkp
    ###############

    chosen_chkp = args.model
    print('Load pretrained: {}'.format(chosen_chkp))

    # Initialize configuration class
    config = Config()
    config.name = args.name
    config.dataset = args.dataset
    config.first_subsampling_dl = args.first_subsampling_dl
    config.fea_size = args.fea_size
    config.voxel_resolution = args.voxel_resolution
    config.data_root = args.data_root
    config.data_root_voxelidx = args.data_root_voxelidx
    config.saving_path = args.out_dir
    config.data_CT = args.data_CT
    config.data_liver = args.data_liver

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.
    #config.augment_symmetries = [False, False, False]
    #config.augment_rotation = 'none'
    #config.augment_scale_min = 0.99
    #config.augment_scale_max = 1.01
    #config.augment_noise = 0.0001
    #config.augment_color = 1.0

    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    #config.batch_num = 3
    #config.in_radius = 4
    config.validation_size = 200
    config.input_threads = 10
    

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initiate dataset
    test_dataset = Dataset_test(config, name='testing')

    # Data loader
    test_loader = DataLoader(test_dataset,
                            batch_size=config.batch_size,
                            num_workers=0,#config.input_threads,
                            collate_fn=collate_test,
                            pin_memory=True
                            )

    # Calibrate samplers
    # test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = Net(config, test_dataset.label_values, test_dataset.ignored_labels)

    # Define a visualizer class
    tester = ModelTester(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Training
    tester.cloud_segmentation_test(net, test_loader, config)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path=None, on_gpu=True):

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device('cpu' if args.gpu_idx < 0 else 'cuda:{}'.format(args.gpu_idx))
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        # Test saving path
        self.record_file = None
        if config.saving:

            self.test_path = args.out_dir
                
            if not exists(self.test_path):
                makedirs(self.test_path)

            print('Test path: {}'.format(self.test_path))

            if not exists(join(self.test_path, 'predictions')):
                makedirs(join(self.test_path, 'predictions'))
            
            self.record_file = open(join(self.test_path, 'TestInfo.txt'), 'w')
        else:
            self.test_path = None

        self.record(str(args))

        ##########################
        # Load previous checkpoint
        ##########################

        self.chkp_path = chkp_path
        checkpoint = torch.load(chkp_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def record(self, info):
        print(info)
        if self.record_file:
            self.record_file.write(info + '\n')
            self.record_file.flush()

    def get_batch_acc(self, uout, label):

        """soft dice score"""
        eps = 1e-7
        uout = torch.Tensor(uout)
        label = torch.Tensor(label)

        #print("type(uout), uout.shape, type(label), label.shape:", type(uout), uout.shape, type(label), label.shape)
        iflat = uout.view(-1) .float()
        tflat = label.view(-1).float()
        'Only evaluate model performance in the liver area'
        iflat = iflat[tflat > 0]
        tflat = tflat[tflat > 0]

        intersection = (iflat * tflat).sum()
        dice_0 = 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

        return dice_0

    def hole_filling(self, bw, hole_min, hole_max, fill_2d=True):
        bw = bw > 0
        if len(bw.shape) == 2:
            background_lab = label(~bw, connectivity=1)
            fill_out = np.copy(background_lab)
            component_sizes = np.bincount(background_lab.ravel())
            too_big = component_sizes > hole_max
            too_big_mask = too_big[background_lab]
            fill_out[too_big_mask] = 0
            too_small = component_sizes < hole_min
            too_small_mask = too_small[background_lab]
            fill_out[too_small_mask] = 0
        elif len(bw.shape) == 3:
            if fill_2d:
                fill_out = np.zeros_like(bw)
                for zz in range(bw.shape[1]):
                    background_lab = label(~bw[:, zz, :], connectivity=1)   # 1表示4连通， ~bw[zz, :, :]1变为0， 0变为1
                    # 标记背景和孔洞， target区域标记为0
                    out = np.copy(background_lab)
                    # plt.imshow(bw[:, :, 87])
                    # plt.show()
                    component_sizes = np.bincount(background_lab.ravel())
                    # 求各个类别的个数
                    too_big = component_sizes > hole_max
                    too_big_mask = too_big[background_lab]

                    out[too_big_mask] = 0
                    too_small = component_sizes < hole_min
                    too_small_mask = too_small[background_lab]
                    out[too_small_mask] = 0
                    # 大于最大孔洞和小于最小孔洞的都标记为0， 所以背景部分被标记为0了。只剩下符合规则的孔洞
                    fill_out[:, zz, :] = out
                    # 只有符合规则的孔洞区域是1， 背景及target都是0
            else:
                background_lab = label(~bw, connectivity=1)
                fill_out = np.copy(background_lab)
                component_sizes = np.bincount(background_lab.ravel())
                too_big = component_sizes > hole_max
                too_big_mask = too_big[background_lab]
                fill_out[too_big_mask] = 0
                too_small = component_sizes < hole_min
                too_small_mask = too_small[background_lab]
                fill_out[too_small_mask] = 0
        else:
            print('error')
            return

        return np.logical_or(bw, fill_out)  # 或运算，孔洞的地方是1，原来target的地方也是1

    def reprocessing(self, label):

        flag_5_8 = True
        flag_2_3 = True
        for i in range (label.shape[0]):

            l = label[i,:,:]
            num_2 = np.sum(l == 2)
            num_3 = np.sum(l == 3)
            num_5 = np.sum(l == 5)
            num_8 = np.sum(l == 8)
            num_6 = np.sum(l == 6)
            num_7 = np.sum(l == 7)

            if  flag_2_3:
                if  num_3 >= num_2:
                    l[l == 2] = 3
                else:
                    flag_2_3 = False
                    l[l == 3] = 2
            else:
                l[l == 3] = 2

            if flag_5_8:
                if  num_5 >= num_8:
                    l[l == 8] = 5
                else:
                    l[l == 5] = 8
                    l[l == 6] = 7
                    flag_5_8 = False
                if  num_6 >= num_7:
                    l[l == 7] = 6
                else:
                    l[l == 6] = 7
                    l[l == 5] = 8
                    flag_5_8 = False
            else:
                l[l == 5] = 8
                l[l == 6] = 7


        num = 10000
        
        new_label_1 = np.zeros(label.shape)
        new_label_1[label == 1] =1
        new_label_1 = self.hole_filling(new_label_1,0,num,False)
        #
        new_label_2 = np.zeros(label.shape)
        new_label_2[label == 2] =1
        new_label_2 = self.hole_filling(new_label_2,0,num,False)
        #
        new_label_3 = np.zeros(label.shape)
        new_label_3[label == 3] =1
        new_label_3 = self.hole_filling(new_label_3,0,num,False)
        #
        new_label_4 = np.zeros(label.shape)
        new_label_4[label == 4] =1
        new_label_4 = self.hole_filling(new_label_4,0,num,False)
        #
        new_label_5 = np.zeros(label.shape)
        new_label_5[label == 5] =1
        new_label_5 = self.hole_filling(new_label_5,0,num,False)
        #
        new_label_6 = np.zeros(label.shape)
        new_label_6[label == 6] =1
        new_label_6 = self.hole_filling(new_label_6,0,num,False)
        #
        new_label_7 = np.zeros(label.shape)
        new_label_7[label == 7] =1
        new_label_7 = self.hole_filling(new_label_7,0,num,False)
        #
        new_label_8 = np.zeros(label.shape)
        new_label_8[label == 8] =1
        new_label_8 = self.hole_filling(new_label_8,0,num,False)
        #
        label[new_label_1 == 1] = 1
        label[new_label_2 == 1] = 2
        label[new_label_3 == 1] = 3
        label[new_label_4 == 1] = 4
        label[new_label_5 == 1] = 5
        label[new_label_6 == 1] = 6
        label[new_label_7 == 1] = 7
        label[new_label_8 == 1] = 8

        return label

    def inverse_resize_couinaud(self, resized_couinaud, target_num_slices):
        """
        Restore the Couinaud mask to the original resolution along the z-axis.

        Parameters:
            resized_couinaud (SimpleITK.Image): The resized Couinaud mask as a SimpleITK.Image object.

        Returns:
            numpy.ndarray: The interpolated Couinaud mask with original z-spacing (D, H, W).
        """
        # Read metadata from the input image
        origin = resized_couinaud.GetOrigin()
        direction = resized_couinaud.GetDirection()

        # Convert to numpy array
        couinaud_numpy = sitk.GetArrayFromImage(resized_couinaud).transpose(1, 2, 0)  # (H, W, D)
        non_zero_slices = np.any(couinaud_numpy > 0, axis=(0, 1))
        cropped_couinaud = couinaud_numpy[:, :, non_zero_slices]
        original_num_slices = cropped_couinaud.shape[-1]

        # Calculate the scaling factor for the z-axis
        scale_factor_z = target_num_slices / original_num_slices

        # Use nearest-neighbor interpolation for labels or cubic interpolation for smooth scaling
        # If the mask contains discrete classes (0-8), stick with order=0
        couinaud_original = zoom(cropped_couinaud, (1, 1, scale_factor_z), order=0)  # Nearest-neighbor for categorical data

        # Convert back to (D, H, W)
        couinaud_original = couinaud_original.transpose(2, 0, 1)

        return couinaud_original

    def cloud_segmentation_test(self, net, test_loader, config, num_votes=100, debug=False):
        """
        Test method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes


        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_min = -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        data_iter = iter(test_loader)

        '## Start test loop'
        print('Point prediction...')
        total_time_ms = 0.0
        num_cases = 0
        while True:
            print('Initialize workers')
            try:
                print("Attempting to load batch")
                batch = next(data_iter)  # CT and seg label
                print("Batch loaded successfully")
            except StopIteration:
                break


            stacked_probs = []
            s_points = []
            s_points_Voxelidx = []
            lengths = []
            for sub_i in range(len(batch)):
                if 'cuda' in self.device.type:
                    batch[sub_i].to(self.device)
                print(f'{sub_i+1} / {len(batch)}')
                # Forward pass
                with torch.no_grad():
                    'Start the inference'
                    torch.cuda.synchronize()  # Wait for GPU to finish
                    self.starter.record()

                    outputs = net(batch[sub_i], config)

                    self.ender.record()
                    torch.cuda.synchronize()
                    infer_time = self.starter.elapsed_time(self.ender)
                    total_time_ms += infer_time

                # Get probs and labels
                stacked_probs_sub = softmax(outputs)
                s_points_sub = batch[sub_i].points[0]
                s_points_Voxelidx_sub = batch[sub_i].points_Voxelidx
                lengths_sub = batch[sub_i].lengths[0]
                stacked_probs.append(stacked_probs_sub)
                s_points.append(s_points_sub)
                s_points_Voxelidx.append(s_points_Voxelidx_sub)
                lengths.append(lengths_sub)

            num_cases += 1

            stacked_probs = torch.cat(stacked_probs,dim=0).cpu().detach().numpy()
            s_points = torch.cat(s_points,dim=0).cpu().numpy()
            s_points_Voxelidx = torch.cat(s_points_Voxelidx,dim=0).cpu().numpy()
            lengths = torch.cat(lengths,dim=0).cpu().numpy()
            torch.cuda.synchronize(self.device)

            probs = stacked_probs
            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                if label_value in test_loader.dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

            case_name = batch[-1].case_name

            # Save predictions
            print('Saving clouds')

            # Get file
            points = s_points_Voxelidx

            # Save plys
            test_name = join(self.test_path, 'predictions', case_name+'.txt')
            PreTxt = np.concatenate((points, preds[:,None]), axis=1) # (N,4)

            df = pd.DataFrame(PreTxt.tolist())
            df.to_csv(test_name, header=None, index=False) 

        self.record(f'Total inference time: {total_time_ms:.2f} ms')
        self.record(f'Average inference time per case: {total_time_ms / num_cases:.2f} ms')

        '## Points to masks'
        print('Convert point predictions to image domain...')
        Txt_file = join(self.test_path, 'predictions')
        save_file = join(self.test_path, 'predictions_nii')
        liver_file = config.data_liver
        image_nii_file = config.data_CT
        self.points2img(Txt_file, save_file, liver_file, image_nii_file)

        '## Fill holes in the saved mask'
        print('Holes filling...')
        pred_nii_file = join(self.test_path, 'predictions_nii')
        save_fill_file = join(self.test_path, 'predictions_fillHoles_nii')
        liver_file = config.data_liver
        self.fill_holes_in_mask(pred_nii_file, save_fill_file, liver_file)

        return


    def points2img(self, Txt_file, save_file, liver_file, image_nii_file):
        Txt_folder = sorted(os.listdir(Txt_file))
        liver_folder = sorted(os.listdir(liver_file))
        l = 0

        cnt = 0
        for file_name,liver_name in zip(Txt_folder,liver_folder):

            print(file_name)
            file_path = os.path.join(Txt_file,file_name)
            
            liver_path = os.path.join(liver_file,liver_name)
            save_point_path = os.path.join(save_file, file_name.split('.')[0]+'.nii.gz')     
            os.makedirs(save_file, exist_ok=True)

            point_arr = np.loadtxt(file_path,delimiter=",")
            image_arr = point_arr

            liver_Image = sitk.ReadImage(liver_path)
            liver_arr = sitk.GetArrayFromImage(liver_Image)

            arr = np.zeros(liver_arr.shape)

            image_arr, point_arr = image_arr.astype(int), point_arr.astype(int)
            for i in range(len(image_arr)):
                arr[image_arr[i, 2], image_arr[i, 1], image_arr[i, 0]] = point_arr[i, 3]

            save_Image = sitk.GetImageFromArray(arr)
            save_Image.SetDirection(liver_Image.GetDirection())
            save_Image.SetOrigin(liver_Image.GetOrigin())
            save_Image.SetSpacing(liver_Image.GetSpacing())
            sitk.WriteImage(save_Image, save_point_path)

            l += 1

            arr[liver_arr == 0] = 0

            all_uout = np.zeros((liver_arr.shape[0], 8, liver_arr.shape[1], liver_arr.shape[2]))
            for i in range(1, 9):
                all_uout[:, i - 1, :, :][arr == i] = 1

            cnt +=1

    def fill_holes_in_mask(self, pred_nii_file, save_fill_file, liver_file):
        """
        Fills holes in a mask with values in the range [0,1,2,...,N].
        
        Parameters:
            mask (np.ndarray): Input mask of shape (H, W, ...) with integer values.

        Returns:
            np.ndarray: Mask with filled holes.
        """
        cnt = 0
        pred_nii_folder = sorted(os.listdir(pred_nii_file))
        liver_nii_folder = sorted(os.listdir(liver_file))

        for file_name,liver_name in zip(pred_nii_folder,liver_nii_folder):

            print(file_name)

            mask = sitk.ReadImage(os.path.join(pred_nii_file, file_name))
            origin = mask.GetOrigin()
            spacing = mask.GetSpacing()
            direction = mask.GetDirection()
            mask = sitk.GetArrayFromImage(mask) # (slices,512,512)

            liver_path = os.path.join(liver_file, liver_name)
            liver_Image = sitk.ReadImage(liver_path)
            liver_arr = sitk.GetArrayFromImage(liver_Image)

            filled_mask = self.reprocessing(mask)

            save_Image = sitk.GetImageFromArray(filled_mask)
            save_Image.SetDirection(direction)
            save_Image.SetOrigin(origin)
            save_Image.SetSpacing(spacing)
            os.makedirs(save_fill_file, exist_ok=True)
            save_fill_path = os.path.join(save_fill_file, file_name)
            sitk.WriteImage(save_Image, save_fill_path)

            arr = filled_mask.copy()
            arr[liver_arr == 0] = 0

            all_uout = np.zeros((liver_arr.shape[0], 8, liver_arr.shape[1], liver_arr.shape[2]))
            for i in range(1, 9):
                all_uout[:, i - 1, :, :][arr == i] = 1

            cnt +=1

        return filled_mask



if __name__ == '__main__':

    print('Data preparation...')
    ####################################################################################################
    from data_processing import dir_transform,convert2point

    LPI_data_save_root = os.path.join(args.LPI_volume_dir, 'LPI_CT_volume')
    os.makedirs(LPI_data_save_root, exist_ok=True)
    LPI_liver_save_root = os.path.join(args.LPI_volume_dir, 'LPI_liver_volume')
    os.makedirs(LPI_liver_save_root, exist_ok=True)

    dir_transform(data_root=args.data_CT, LPI_data_save_root=LPI_data_save_root, liver_root=args.data_liver, LPI_liver_save_root=LPI_liver_save_root)

    convert2point(data_root=LPI_data_save_root, liver_root=LPI_liver_save_root, save_root=args.data_root, save_root_Voxelidx=args.data_root_voxelidx)

    print('Inference...')
    ####################################################################################################   
    infer()

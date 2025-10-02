
# Common libs
import signal
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys
import argparse

# Dataset
# from datasets.S3DIS import *
from datasets.dataset import Dataset_LiTS_train, Dataset_LiTS_val
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.config import Config
from models.architectures import Net

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion, metrics, AverageMeter
from utils.config import Config
from sklearn.neighbors import KDTree
torch.autograd.set_detect_anomaly(True)

TRAIN_NAME = __file__.split('.')[0]

# ----------------------------------------------------------------------------------------------------------------------
#
#           Arguments 
#       \******************/
#

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='', help='description')
parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--first_subsampling_dl', type=float, default=-1, help='Size of the first subsampling grid')
parser.add_argument('--fea_size', nargs='+', type=int, required=True, help='Feature sizes required by DGMN')
parser.add_argument('--voxel_resolution', nargs='+', type=int, required=True, help='Voxel resolutions required by re-voxelization and de-voxelization')

args = parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class LiTSConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    name = ''

    # Dataset name
    dataset = '' #'LiTS'
    data_root = '' #'/data1/xzhang2/liver_couinaud_segmentation/data/LiTS_Couinaud'
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
    architecture = [
                    'adapt_simple',
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
                    'unary'
                    ]

    # convolutional feature
    adaptive_feature = 'xyz'
    first_adaptive_feature = 'xyz_joint'

    # Radius of the input sphere
    # in_radius = 1.5

    # Size of the first subsampling grid in meter
    # first_subsampling_dl = 0.03 # train_1
    # first_subsampling_dl = 0.015625 # train_2
    # first_subsampling_dl = 0.0078125 # train_3
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
    max_epoch = 400

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 50) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # # Number of batch
    # batch_num = 6

    # # Number of steps per epochs
    # epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = -1

    # Number of epoch between each checkpoint
    checkpoint_gap = 10

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
    saving_path = '/home/xzhang2/data1/liver_couinaud_segmentation/model/GRCNN/outputs'
    saving_ply = False


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

class BatchData:
    def __init__(self, points, neighbors, pools, upsamples, lengths, features, labels, case_name):
        self.points = points
        self.neighbors = neighbors
        self.pools = pools
        self.upsamples = upsamples
        self.lengths = lengths
        self.features = features
        self.labels = labels
        # self.labels_full = labels_full
        # self.points_full = points_full
        # self.image_size = image_size
        self.case_name = case_name

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        # self.labels_full = self.labels_full.to(device)
        # self.points_full = self.points_full.to(device)
        return self
    
def LiTS_collate_train(batch):
    L = (len(batch[0]) - 2) // 5
    # L = (len(batch[0]) - 5) // 5
    ind = 0
    points = [torch.from_numpy(nparray) for nparray in batch[0][ind:ind+L]]
    ind += L
    neighbors = [torch.from_numpy(nparray) for nparray in batch[0][ind:ind+L]]
    ind += L
    pools = [torch.from_numpy(nparray) for nparray in batch[0][ind:ind+L]]
    ind += L
    upsamples = [torch.from_numpy(nparray) for nparray in batch[0][ind:ind+L]]
    ind += L
    lengths = [torch.from_numpy(nparray) for nparray in batch[0][ind:ind+L]]
    ind += L
    features = torch.from_numpy(batch[0][ind])
    ind += 1
    labels = torch.from_numpy(batch[0][ind]).type(torch.LongTensor)
    ind += 1
    # labels_full = torch.from_numpy(batch[0][ind]).type(torch.LongTensor)
    # ind += 1
    # points_full = torch.from_numpy(batch[0][ind]).type(torch.float32)
    # ind += 1
    # image_size = batch[0][ind]
    # ind += 1
    case_name = batch[0][ind]
    return BatchData(points, neighbors, pools, upsamples, lengths, features, labels, case_name)

def LiTS_collate_val(batch):
    L = (len(batch[0]) - 2) // 5
    # L = (len(batch[0]) - 5) // 5
    ind = 0
    points = [torch.from_numpy(nparray) for nparray in batch[0][ind:ind+L]]
    ind += L
    neighbors = [torch.from_numpy(nparray) for nparray in batch[0][ind:ind+L]]
    ind += L
    pools = [torch.from_numpy(nparray) for nparray in batch[0][ind:ind+L]]
    ind += L
    upsamples = [torch.from_numpy(nparray) for nparray in batch[0][ind:ind+L]]
    ind += L
    lengths = [torch.from_numpy(nparray) for nparray in batch[0][ind:ind+L]]
    ind += L
    features = torch.from_numpy(batch[0][ind])
    ind += 1
    labels = torch.from_numpy(batch[0][ind]).type(torch.LongTensor)
    ind += 1
    # labels_full = torch.from_numpy(batch[0][ind]).type(torch.LongTensor)
    # ind += 1
    # points_full = torch.from_numpy(batch[0][ind]).type(torch.float32)
    # ind += 1
    # image_size = batch[0][ind]
    # ind += 1
    case_name = batch[0][ind]
    return BatchData(points, neighbors, pools, upsamples, lengths, features, labels, case_name)


def main():

    ############################
    # Initialize the environment
    ############################

    # Set GPU visible device
    print('USE GPU: {}'.format(args.gpu_idx))

    ###############
    # Previous chkp
    ###############

    # Choose here if you want to start training from a previous snapshot (None for new training)
    previous_training_path = ''

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = LiTSConfig()
    config.name = args.name
    config.dataset = args.dataset
    config.first_subsampling_dl = args.first_subsampling_dl
    config.fea_size = args.fea_size
    config.voxel_resolution = args.voxel_resolution
    config.data_root = args.data_root

    if previous_training_path:
        config.load(os.path.join('results', previous_training_path))
        config.saving_path = None

    # Initialize datasets
    training_dataset = Dataset_LiTS_train(config, name='training')
    val_dataset = Dataset_LiTS_val(config, name='validation')

    # Initialize samplers
    # training_sampler = S3DISSampler(training_dataset)
    # test_sampler = S3DISSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                            batch_size=config.batch_size,
                            num_workers=0,#config.input_threads,
                            collate_fn=LiTS_collate_train,
                            pin_memory=True
                            )
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            num_workers=0,#config.input_threads,
                            collate_fn=LiTS_collate_val,
                            pin_memory=True
                            )

    # Calibrate samplers
    # training_sampler.calibration(training_loader, verbose=True)
    # test_sampler.calibration(test_loader, verbose=True)

    # Optional debug functions
    # debug_timing(training_dataset, training_loader)
    # debug_timing(test_dataset, test_loader)
    # debug_upsampling(training_dataset, training_loader)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = Net(config, config.label_values, config.ignored_labels)

    debug = True
    if debug:
        print('\n*************************************\n')
        print(net)
        print('\n*************************************\n')
        print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')

    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    trainer.train(net, training_loader, val_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer
#       \***************/
#

class ModelTrainer:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path=None, finetune=False, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        ############
        # Parameters
        ############
        self.writer = SummaryWriter(log_dir=os.path.join(config.saving_path, "logs", config.name))
        os.makedirs(os.path.join(config.saving_path, "logs", config.name), exist_ok=True)

        self.lossesMeter = AverageMeter(name='TrainMeter total loss ')
        # self.lossesMeter_intra = AverageMeter(name='TrainMeter intra loss ')
        # self.lossesMeter_inter = AverageMeter(name='TrainMeter inter loss ')
        self.precisionMeter = AverageMeter(name='TrainMeter precision')
        self.recallMeter = AverageMeter(name='TrainMeter recall')
        self.fscoreMeter = AverageMeter(name='TrainMeter f1')
        self.iouMeter = AverageMeter(name='TrainMeter IoU')
        self.accMeter = AverageMeter(name='TrainMeter accuracy')

        self.precisionMeter_per_class = {}
        self.recallMeter_per_class = {}
        self.fscoreMeter_per_class = {}
        self.iouMeter_per_class = {}
        for class_i in range(config.num_classes):
            self.precisionMeter_per_class.update(
                {f'class_{class_i}': AverageMeter(name=f'TrainMeter precision for class_{class_i}')}
            )
            self.recallMeter_per_class.update(
                {f'class_{class_i}': AverageMeter(name=f'TrainMeter recall for class_{class_i}')}
            )
            self.fscoreMeter_per_class.update(
                {f'class_{class_i}': AverageMeter(name=f'TrainMeter f1 for class_{class_i}')}
            )
            self.iouMeter_per_class.update(
                {f'class_{class_i}': AverageMeter(name=f'TrainMeter IoU for class_{class_i}')}
            )


        self.lossesMeter_val = AverageMeter(name='ValMeter total loss ')
        self.precisionMeter_val = AverageMeter(name='ValMeter precision')
        self.recallMeter_val = AverageMeter(name='ValMeter recall')
        self.fscoreMeter_val = AverageMeter(name='ValMeter f1')
        self.iouMeter_val = AverageMeter(name='ValMeter IoU')
        self.accMeter_val = AverageMeter(name='ValMeter accuracy')

        self.precisionMeter_val_per_class = {}
        self.recallMeter_val_per_class = {}
        self.fscoreMeter_val_per_class = {}
        self.iouMeter_val_per_class = {}
        for class_i in range(config.num_classes):
            self.precisionMeter_val_per_class.update(
                {f'class_{class_i}': AverageMeter(name=f'ValMeter precision for class_{class_i}')}
            )
            self.recallMeter_val_per_class.update(
                {f'class_{class_i}': AverageMeter(name=f'ValMeter recall for class_{class_i}')}
            )
            self.fscoreMeter_val_per_class.update(
                {f'class_{class_i}': AverageMeter(name=f'ValMeter f1 for class_{class_i}')}
            )
            self.iouMeter_val_per_class.update(
                {f'class_{class_i}': AverageMeter(name=f'ValMeter IoU for class_{class_i}')}
            )

        # Epoch index
        self.epoch = 0
        self.step = 0

        # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in net.named_parameters() if 'offset' in k]
        other_params = [v for k, v in net.named_parameters() if 'offset' not in k]
        deform_lr = config.learning_rate * config.deform_lr_factor
        self.optimizer = torch.optim.SGD([{'params': other_params},
                                          {'params': deform_params, 'lr': deform_lr}],
                                         lr=config.learning_rate,
                                         momentum=config.momentum,
                                         weight_decay=config.weight_decay)

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device('cpu' if args.gpu_idx < 0 else 'cuda:{}'.format(args.gpu_idx))
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        if (chkp_path is not None):
            if finetune:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                net.train()
                print("Model restored and ready for finetuning.")
            else:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                net.train()
                print("Model and training state restored.")

        # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                config.saving_path = join('results', TRAIN_NAME if args.name == '' else args.name)
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, net, training_loader, val_loader, config):
        """
        Train the model on a particular dataset.
        """

        ################
        # Initialization
        ################

        if config.saving:
            # Training log file
            os.makedirs(os.path.join(config.saving_path, 'logger', config.name), exist_ok=True)
            with open(join(config.saving_path, 'logger', config.name, 'training.txt'), "w") as file:
                file.write('epochs steps out_loss offset_loss train_accuracy time\n')

            # Killing file (simply delete this file when you want to stop the training)
            # PID_file = join(config.saving_path, 'running_PID.txt')
            # if not exists(PID_file):
            #     with open(PID_file, "w") as file:
            #         file.write('Launched with PyCharm')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints', config.name)
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None
            PID_file = None

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        print(f"Dataset length: {len(training_loader.dataset)}")

        # Start training loop
        data_iter = iter(training_loader)
        self.steps_per_epoch = len(training_loader)
        self.step = 0

        predictions = []
        targets = []
        loss_list = []
        # intra_loss_list = []
        # inter_loss_list = []
        training_loader.dataset.training_labels = []
        # Number of classes including ignored labels
        nc_tot = training_loader.dataset.num_classes
        # Number of classes predicted by the model
        nc_model = config.num_classes
        softmax = torch.nn.Softmax(1)

        # for epoch in range(config.max_epoch):
        while (
            self.epoch <= config.max_epoch
        ):

            # Remove File for kill signal
            if self.epoch == config.max_epoch - 1 and exists(PID_file):
                remove(PID_file)

            
            # for batch in training_loader:
            try:
                print("Attempting to load batch")
                batch = next(data_iter)  # CT and seg label
                print("Batch loaded successfully")
            except StopIteration:
                print("End of dataset reached, reinitializing data loader.")
                data_iter = iter(training_loader)
                batch = next(data_iter)
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise  # Re-raise the exception

            # Check kill signal (running_PID.txt deleted)
            # if config.saving and not exists(PID_file):
            #     continue

            ##################
            # Processing batch
            ##################

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            # outputs, out_feat = net(batch, config) # (N,8)
            outputs = net(batch, config) # (N,8)

            # loss, intra_loss, inter_loss = net.loss(outputs, batch.labels, out_feat)
            loss = net.loss(outputs, batch.labels)
            acc = net.accuracy(outputs, batch.labels)
            # loss = net.loss(outputs, batch.labels_full)
            # acc = net.accuracy(outputs, batch.labels_full)

            predictions.append(softmax(outputs).cpu().detach().numpy())
            targets.append(batch.labels.cpu().detach().numpy())
            # targets.append(batch.labels_full.cpu().detach().numpy())

            training_loader.dataset.training_labels += [batch.labels.detach().cpu().numpy()]
            loss_list.append(loss.cpu().detach().numpy())
            # intra_loss_list.append(intra_loss.cpu().detach().numpy())
            # inter_loss_list.append(inter_loss.cpu().detach().numpy())

            t += [time.time()]

            # Backward + optimize
            loss.backward()

            if config.grad_clip_norm > 0:
                #torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
            self.optimizer.step()
            torch.cuda.synchronize(self.device)

            t += [time.time()]

            # Average timing
            if self.step < 2:
                mean_dt = np.array(t[1:]) - np.array(t[:-1])
            else:
                mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > 10.0:
                last_display = t[-1]
                message = 'e{:03d}-i{:04d} => L={:.3f} acc={:3.0f}% / t(ms): {:5.1f} {:5.1f} {:5.1f})'
                print(message.format(self.epoch, self.step,
                                        loss.item(),
                                        100*acc,
                                        1000 * mean_dt[0],
                                        1000 * mean_dt[1],
                                        1000 * mean_dt[2]))

            # Log file
            if config.saving:
                with open(join(config.saving_path, 'logger', config.name, 'training.txt'), "a") as file:
                    message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                    file.write(message.format(self.epoch,
                                                self.step,
                                                net.output_loss,
                                                net.reg_loss,
                                                acc,
                                                t[-1] - t0))


            self.step += 1

            ##############
            # End of epoch
            ##############
            if self.step % self.steps_per_epoch ==0:
                self.train_proportions = np.zeros(nc_model, dtype=np.float32)
                i = 0
                for label_value in training_loader.dataset.label_values:
                    if label_value not in training_loader.dataset.ignored_labels:
                        self.train_proportions[i] = np.sum([np.sum(labels_ == label_value)
                                                        for labels_ in training_loader.dataset.training_labels])
                        i += 1
                
                # Confusions for our subparts of validation set
                Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
                for i, (probs, truth) in enumerate(zip(predictions, targets)):
                    # Predicted labels
                    preds = training_loader.dataset.label_values[np.argmax(probs, axis=1)]
                    # Confusions
                    Confs[i, :, :] = fast_confusion(truth, preds, training_loader.dataset.label_values).astype(np.int32)
                # Sum all confusions
                C = np.sum(Confs, axis=0).astype(np.float32)
                # Balance with real training proportions
                C *= np.expand_dims(self.train_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                # metrics calculation
                # IoUs = IoU_from_confusions(C)
                # mIoU = np.mean(IoUs)
                precision, recall, F1_score, IoUs, accuracy = metrics(C)
                precision_avg = precision.mean() # average precision over 8 liver segments
                recall_avg = recall.mean() # average recall over 8 liver segments
                F1_avg = F1_score.mean() # average F1-score over 8 liver segments
                IoU_avg = IoUs.mean() # average IoU over 8 liver segments
                loss_avg = np.array(loss_list).mean() # average loss over steps of one epoch
                # intra_loss_avg = np.array(intra_loss_list).mean()
                # inter_loss_avg = np.array(inter_loss_list).mean()

                self.lossesMeter.update(loss_avg.item())
                # self.lossesMeter_intra.update(intra_loss_avg.item())
                # self.lossesMeter_inter.update(inter_loss_avg.item())
                self.precisionMeter.update(precision_avg.item())
                self.recallMeter.update(recall_avg.item())
                self.fscoreMeter.update(F1_avg.item())
                self.iouMeter.update(IoU_avg.item())
                self.accMeter.update(accuracy.item())

                self.writer.add_scalar("train/loss", scalar_value=self.lossesMeter.avg, global_step=self.epoch+1)
                # self.writer.add_scalar("train/loss_intra", scalar_value=self.lossesMeter_intra.avg, global_step=self.epoch+1)
                # self.writer.add_scalar("train/loss_inter", scalar_value=self.lossesMeter_inter.avg, global_step=self.epoch+1)
                self.writer.add_scalar("train_metrics/precision", scalar_value=self.precisionMeter.avg, global_step=self.epoch+1)
                self.writer.add_scalar("train_metrics/recall", scalar_value=self.recallMeter.avg, global_step=self.epoch+1)
                self.writer.add_scalar("train_metrics/f1", scalar_value=self.fscoreMeter.avg, global_step=self.epoch+1)
                self.writer.add_scalar("train_metrics/IoU", scalar_value=self.iouMeter.avg, global_step=self.epoch+1)
                self.writer.add_scalar("train_metrics/accracy", scalar_value=self.accMeter.avg, global_step=self.epoch+1)

                
                for class_idx in range(nc_model):  # Assuming `nc_model` is the number of classes
                    self.precisionMeter_per_class[f'class_{class_idx}'].update(precision[class_idx])
                    self.recallMeter_per_class[f'class_{class_idx}'].update(recall[class_idx])
                    self.fscoreMeter_per_class[f'class_{class_idx}'].update(F1_score[class_idx])
                    self.iouMeter_per_class[f'class_{class_idx}'].update(IoUs[class_idx])
                
                precision_dict = {f'class_{class_idx + 1}': self.precisionMeter_per_class[f'class_{class_idx}'].avg for class_idx in range(nc_model)}
                recall_dict = {f'class_{class_idx + 1}': self.recallMeter_per_class[f'class_{class_idx}'].avg for class_idx in range(nc_model)}
                f1_dict = {f'class_{class_idx + 1}': self.fscoreMeter_per_class[f'class_{class_idx}'].avg for class_idx in range(nc_model)}
                iou_dict = {f'class_{class_idx + 1}': self.iouMeter_per_class[f'class_{class_idx}'].avg for class_idx in range(nc_model)}
                self.writer.add_scalars("train_metrics/precision_per_class", precision_dict, self.epoch + 1)
                self.writer.add_scalars("train_metrics/recall_per_class", recall_dict, self.epoch + 1)
                self.writer.add_scalars("train_metrics/F1_per_class", f1_dict, self.epoch + 1)
                self.writer.add_scalars("train_metrics/IoU_per_class", iou_dict, self.epoch + 1)

                training_loader.dataset.training_labels = []
                predictions = []
                targets = []
                loss_list = []
                # intra_loss_list = []
                # inter_loss_list = []

                # Check kill signal (running_PID.txt deleted)
                # if config.saving and not exists(PID_file):
                #     break

                # Update learning rate
                if self.epoch in config.lr_decays:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= config.lr_decays[self.epoch]

                # Saving
                if config.saving:
                    # Get current state dict
                    save_dict = {'epoch': self.epoch,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'saving_path': config.saving_path}

                    # Save current state of the network (for restoring purposes)
                    checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                    torch.save(save_dict, checkpoint_path)

                    # Save checkpoints occasionally
                    if (self.epoch + 1) % config.checkpoint_gap == 0:
                        checkpoint_path = join(checkpoint_directory, 'epoch_{:04d}.tar'.format(self.epoch))
                        torch.save(save_dict, checkpoint_path)

                # Validation
                net.eval()
                self.validation(net, val_loader, config)
                net.train()

                # Update epoch
                self.epoch += 1

        print('Finished Training')
        return

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def validation(self, net, val_loader, config: Config):

        self.cloud_segmentation_validation(net, val_loader, config)


    def cloud_segmentation_validation(self, net, val_loader, config, debug=False):
        """
        Validation method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Do not validate if dataset has no validation cloud
        # if val_loader.dataset.validation_split not in val_loader.dataset.all_splits:
        #     return

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        #print(nc_tot)
        #print(nc_model)



        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        loss_list = []
        val_loader.dataset.validation_labels = []
        point_coords_list = []
        case_name_list = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)


        t1 = time.time()
        config.validation_size = len(val_loader)

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = net(batch, config) # (N,8)
                loss = net.loss(outputs, batch.labels)

            # Get probs and labels
            stacked_probs = softmax(outputs).cpu().detach().numpy()
            labels = batch.labels.cpu().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            points = batch.points[0].cpu().numpy()
            case_name = batch.case_name
            # in_inds = batch.input_inds.cpu().numpy()
            # cloud_inds = batch.cloud_inds.cpu().numpy()
            torch.cuda.synchronize(self.device)

            # Initiate global prediction over validation clouds
            val_loader.dataset.validation_labels += [labels]
            if not hasattr(self, 'validation_probs'):
                # self.validation_probs = [np.zeros((l.shape[0], nc_model))
                #                          for l in val_loader.dataset.input_labels]
                self.val_proportions = np.zeros(nc_model, dtype=np.float32)
                i = 0
                for label_value in val_loader.dataset.label_values:
                    if label_value not in val_loader.dataset.ignored_labels:
                        self.val_proportions[i] = np.sum([np.sum(labels_ == label_value)
                                                        for labels_ in val_loader.dataset.validation_labels])
                        i += 1

            # Get predictions and labels per instance
            # ***************************************
            case_name_list.append(case_name)
            loss_list.append(loss.detach().cpu().numpy())

            i0 = 0
            for b_i, length in enumerate(lengths):

                # Get prediction
                target = labels[i0:i0 + length] # (N,1)
                probs = stacked_probs[i0:i0 + length] # (N,8)
                point_coords = points[i0:i0 + length] # (N,3)
                # inds = in_inds[i0:i0 + length]
                # c_i = cloud_inds[b_i]

                # Update current probs in whole cloud
                # self.validation_probs[c_i][inds] = val_smooth * self.validation_probs[c_i][inds] \
                #                                    + (1 - val_smooth) * probs

                # Stack all prediction for this epoch
                predictions.append(probs)
                targets.append(target)
                point_coords_list.append(point_coords)

                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 5.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * i / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):

            # Insert false columns for ignored labels
            # for l_ind, label_value in enumerate(val_loader.dataset.label_values):
            #     if label_value in val_loader.dataset.ignored_labels:
            #         probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)


        t3 = time.time()

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        # for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
        #     if label_value in val_loader.dataset.ignored_labels:
        #         C = np.delete(C, l_ind, axis=0)
        #         C = np.delete(C, l_ind, axis=1)

        # Balance with real validation proportions
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)


        t4 = time.time()

        # metrics calculation
        # IoUs = IoU_from_confusions(C)
        # mIoU = np.mean(IoUs)
        precision, recall, F1_score, IoUs, accuracy = metrics(C)
        precision_avg = precision.mean() # average precision over 8 liver segments
        recall_avg = recall.mean() # average recall over 8 liver segments
        F1_avg = F1_score.mean() # average F1-score over 8 liver segments
        IoU_avg = IoUs.mean() # average IoU over 8 liver segments
        loss_avg = np.array(loss_list).mean() # average loss over steps of one validation epoch

        self.lossesMeter_val.update(loss_avg.item())
        self.precisionMeter_val.update(precision_avg.item())
        self.recallMeter_val.update(recall_avg.item())
        self.fscoreMeter_val.update(F1_avg.item())
        self.iouMeter_val.update(IoU_avg.item())
        self.accMeter_val.update(accuracy.item())

        self.writer.add_scalar("validation/loss", scalar_value=self.lossesMeter_val.avg, global_step=self.epoch+1)
        self.writer.add_scalar("validation_metrics/precision", scalar_value=self.precisionMeter_val.avg, global_step=self.epoch+1)
        self.writer.add_scalar("validation_metrics/recall", scalar_value=self.recallMeter_val.avg, global_step=self.epoch+1)
        self.writer.add_scalar("validation_metrics/f1", scalar_value=self.fscoreMeter_val.avg, global_step=self.epoch+1)
        self.writer.add_scalar("validation_metrics/IoU", scalar_value=self.iouMeter_val.avg, global_step=self.epoch+1)
        self.writer.add_scalar("validation_metrics/accracy", scalar_value=self.accMeter_val.avg, global_step=self.epoch+1)

        for class_idx in range(nc_model):  # Assuming `nc_model` is the number of classes
            self.precisionMeter_val_per_class[f'class_{class_idx}'].update(precision[class_idx])
            self.recallMeter_val_per_class[f'class_{class_idx}'].update(recall[class_idx])
            self.fscoreMeter_val_per_class[f'class_{class_idx}'].update(F1_score[class_idx])
            self.iouMeter_val_per_class[f'class_{class_idx}'].update(IoUs[class_idx])
        
        precision_dict = {f'class_{class_idx + 1}': self.precisionMeter_val_per_class[f'class_{class_idx}'].avg for class_idx in range(nc_model)}
        recall_dict = {f'class_{class_idx + 1}': self.recallMeter_val_per_class[f'class_{class_idx}'].avg for class_idx in range(nc_model)}
        f1_dict = {f'class_{class_idx + 1}': self.fscoreMeter_val_per_class[f'class_{class_idx}'].avg for class_idx in range(nc_model)}
        iou_dict = {f'class_{class_idx + 1}': self.iouMeter_val_per_class[f'class_{class_idx}'].avg for class_idx in range(nc_model)}
        self.writer.add_scalars("validation_metrics/precision_per_class", precision_dict, self.epoch + 1)
        self.writer.add_scalars("validation_metrics/recall_per_class", recall_dict, self.epoch + 1)
        self.writer.add_scalars("validation_metrics/F1_per_class", f1_dict, self.epoch + 1)
        self.writer.add_scalars("validation_metrics/IoU_per_class", iou_dict, self.epoch + 1)

        t5 = time.time()

        # Saving (optionnal)
        if config.saving:

            # Name of saving file
            test_file = join(config.saving_path, 'logger', config.name, 'val_IoUs.txt')

            # Line to write:
            line = '#{:d} || '.format(self.epoch)
            line += '{:.3f} | '.format(np.mean(IoUs)) # mean IoU
            for IoU in IoUs:
                line += '{:.3f} '.format(IoU)
            line += '\n'

            # Write in file
            if exists(test_file):
                with open(test_file, "a") as text_file:
                    text_file.write(line)
            else:
                with open(test_file, "w") as text_file:
                    text_file.write(line)

            # Save potentials
            # pot_path = join(config.saving_path, 'potentials')
            # if not exists(pot_path):
            #     makedirs(pot_path)
            # files = val_loader.dataset.files
            # for i, file_path in enumerate(files):
            #     pot_points = np.array(val_loader.dataset.pot_trees[i].data, copy=False)
            #     cloud_name = file_path.split('/')[-1]
            #     pot_name = join(pot_path, cloud_name)
            #     pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
            #     write_ply(pot_name,
            #               [pot_points.astype(np.float32), pots],
            #               ['x', 'y', 'z', 'pots'])

        t6 = time.time()

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} mean IoU = {:.1f}%'.format(config.dataset, mIoU))

        # Save predicted cloud occasionally
        if config.saving and (self.epoch + 1) % config.checkpoint_gap == 0 and config.saving_ply:
            val_path = join(config.saving_path, 'val_preds_epoch_{:d}'.format(self.epoch), config.name)
            if not exists(val_path):
                makedirs(val_path)
            # files = val_loader.dataset.files
            for i, sub_probs in enumerate(predictions):

                # Get points
                # points = val_loader.dataset.load_evaluation_points(file_path)

                # Get probs on our own ply points
                # sub_probs = self.validation_probs[i]

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

                # Get the predicted labels
                sub_preds = val_loader.dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]

                # Reproject preds on the evaluations points
                # preds = (sub_preds[val_loader.dataset.test_proj[i]]).astype(np.int32)

                # Path of saved validation file
                # cloud_name = file_path.split('/')[-1]
                cloud_name = case_name_list[i]
                val_name = join(val_path, cloud_name)

                # Save file
                labels = val_loader.dataset.validation_labels[i].astype(np.int32)
                write_ply(val_name,
                          [point_coords_list[i], sub_preds, targets[i]],
                          ['x', 'y', 'z', 'preds', 'class'])

        # Display timings
        t7 = time.time()
        if debug:
            print('\n************************\n')
            print('Validation timings:')
            print('Init ...... {:.1f}s'.format(t1 - t0))
            print('Loop ...... {:.1f}s'.format(t2 - t1))
            print('Confs ..... {:.1f}s'.format(t3 - t2))
            print('Confs bis . {:.1f}s'.format(t4 - t3))
            print('IoU ....... {:.1f}s'.format(t5 - t4))
            print('Save1 ..... {:.1f}s'.format(t6 - t5))
            print('Save2 ..... {:.1f}s'.format(t7 - t6))
            print('\n************************\n')

        return


if __name__ == '__main__':
    main()

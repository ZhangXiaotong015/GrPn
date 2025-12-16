
from models.blocks import *
import numpy as np
import torch
import os
import nibabel
# from modules.hcc.hcc import Matcher

def save_nifti(img, img_path):
    pair_img = nibabel.Nifti1Pair(img,np.eye(4))
    nibabel.save(pair_img,img_path)

def save_feat(grid_feat, save_path):
    grid_feat = grid_feat.detach() # (1,C,D,H,W)
    grid_feat = grid_feat.squeeze(0)
    grid_feat_max = torch.max(grid_feat, dim=0)[0]
    grid_feat_sum = torch.sum(grid_feat, dim=0)
    grid_feat_max = grid_feat_max.cpu().numpy() # (D,H,W)
    grid_feat_sum = grid_feat_sum.cpu().numpy() # (D,H,W)
    save_nifti(grid_feat_max, save_path)
    save_nifti(grid_feat_sum, save_path.replace('_max.nii.gz','_sum.nii.gz'))

class Net(nn.Module):
    """
    Class defining Segmentation Network
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(Net, self).__init__()

        ############
        # Parameters
        ############
        self.config = config
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        dgmn_cnt = 0
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config,
                                                    dgmn_cnt))
            # Count the order of DGMN3D
            if 'devoxelization' in block:
                dgmn_cnt +=1

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)

        ############
        # HCC Branch
        ############
        # self.hcc_matcher = Matcher(imside=64)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1, reduction='none')
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        ######################################
        # Extra loss for sharpening boundaries
        ######################################
        # self.ceter_loss = CenterLoss()


        # self.grid_feat_viz = torch.zeros((1,64,32,32,32)).cuda()
        # self.case_name_previous = ''

        return

    def forward(self, batch, config):
    # def forward(self, points_0, points_1, points_2, points_3, points_4, features, 
    #             pools_0, pools_1, pools_2, pools_3, pools_4, 
    #             neighbors_0, neighbors_1, neighbors_2, neighbors_3, neighbors_4,
    #             upsamples_0, upsamples_1, upsamples_2, upsamples_3, upsamples_4):#, config):
    #     inputs = {}
    #     inputs['points_0'] = points_0 # batch.points[0]
    #     inputs['points_1'] = points_1 # batch.points[1]
    #     inputs['points_2'] = points_2 # batch.points[2]
    #     inputs['points_3'] = points_3 # batch.points[3]
    #     inputs['points_4'] = points_4 # batch.points[4]
    #     inputs['features'] = features # batch.features
    #     inputs['pools_0'] = pools_0 # batch.pools[0]
    #     inputs['pools_1'] = pools_1 # batch.pools[1]
    #     inputs['pools_2'] = pools_2 # batch.pools[2]
    #     inputs['pools_3'] = pools_3 # batch.pools[3]
    #     inputs['pools_4'] = pools_4 # batch.pools[4]
    #     inputs['neighbors_0'] = neighbors_0 # batch.neighbors[0]
    #     inputs['neighbors_1'] = neighbors_1 # batch.neighbors[1]
    #     inputs['neighbors_2'] = neighbors_2 # batch.neighbors[2]
    #     inputs['neighbors_3'] = neighbors_3 # batch.neighbors[3]
    #     inputs['neighbors_4'] = neighbors_4 # batch.neighbors[4]
    #     inputs['upsamples_0'] = upsamples_0 # batch.upsamples[0]
    #     inputs['upsamples_1'] = upsamples_1 # batch.upsamples[1]
    #     inputs['upsamples_2'] = upsamples_2 # batch.upsamples[2]
    #     inputs['upsamples_3'] = upsamples_3 # batch.upsamples[3]
    #     inputs['upsamples_4'] = upsamples_4 # batch.upsamples[4]

        # Get input features
        x = batch.features.clone().detach()
        # x = inputs['features'].clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        # skip_x_HCC = []
        # voxel_x = []
        point_feat = []
        dgmn_cnt = 0
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)

            if self.config.architecture[block_i] == 'revoxelization':
                point_feat.append(x)
                x, voxel_coords = block_op(x, batch) # x:(1,C,D,H,W) voxel_coords:(1,3,N)
                # x, voxel_coords = block_op(x, inputs)
                D = x.shape[-3]
                H = x.shape[-2]
                W = x.shape[-1]
                x = x.view(x.shape[0], x.shape[1], -1).permute(0,2,1) # (1,N,C)
                # voxel_x.append(x)
                # if 'dgmn3d' not in config.architecture:
                #     skip_x_HCC.append(x)

            elif self.config.architecture[block_i] == 'dgmn3d':
                x = block_op(x, D, H, W) # (1,N_grid,C)
                # skip_x_HCC.append(x)

            elif self.config.architecture[block_i] == 'devoxelization':
                x = x.permute(0,2,1).reshape(x.shape[0],x.shape[2],D,H,W) # (1,C,D,H,W)
                # if dgmn_cnt==0: ## grid feature visualization
                #     os.makedirs(os.path.join(self.config.saving_path, 'test', config.name, 'grid_feat_viz'), exist_ok=True)
                #     if self.case_name_previous!=batch.case_name and self.case_name_previous!='':
                #         save_feat(self.grid_feat_viz, os.path.join(self.config.saving_path, 'test', config.name, 'grid_feat_viz', batch.case_name+'_max.nii.gz'))
                #         self.grid_feat_viz = torch.zeros((1,64,32,32,32)).cuda()
                #     self.grid_feat_viz = self.grid_feat_viz + x.detach()
                #     self.case_name_previous = batch.case_name

                x = block_op(x, voxel_coords) # (1,C,N)
                x = x.squeeze(0).permute(1,0) # (N,C)
                x = x + point_feat[dgmn_cnt] # (N,C)
                dgmn_cnt +=1

            else:
                x = block_op(x, batch)
                # x = block_op(x, inputs)

        ## points interpolation decoder is the same as the one used in AGCNN
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
            # x = block_op(x, inputs)
        # out_feat = x.clone()
        ## Head of network
        x = self.head_mlp(x)
        x = self.head_softmax(x)

        # if self.training:
        #     return x, out_feat
        # else:
        return x

        ################################
        # Hypercolumn correlation branch
        # for target image
        ################################
        # # Get input features
        # x_target = batch.features_target.clone().detach()
        # skip_x_HCC_target = []
        # point_feat_target = []
        # feat_size_list = []
        # dgmn_cnt = 0
        # for block_i, block_op in enumerate(self.encoder_blocks):

        #     if config.architecture[block_i] == 'revoxelization':
        #         point_feat_target.append(x_target)
        #         x_target, voxel_coords_target = block_op(x_target, batch, 'target') # x:(1,C,D,H,W) voxel_coords:(1,3,N)
        #         D = x_target.shape[-3]
        #         H = x_target.shape[-2]
        #         W = x_target.shape[-1]
        #         x_target = x_target.view(x_target.shape[0], x_target.shape[1], -1).permute(0,2,1) # (1,N,C)
        #         if 'dgmn3d' not in config.architecture:
        #             skip_x_HCC_target.append(x_target)
        #             feat_size_list.append([D, H, W])
        #             if len(skip_x_HCC_target)==4:
        #                 break

        #     elif config.architecture[block_i] == 'dgmn3d':
        #         x_target = block_op(x_target, D, H, W) # (1,N_grid,C)
        #         skip_x_HCC_target.append(x_target)
        #         feat_size_list.append([D, H, W])
        #         if len(skip_x_HCC_target)==4:
        #             break

        #     elif config.architecture[block_i] == 'devoxelization':
        #         x_target = x_target.permute(0,2,1).reshape(x_target.shape[0],x_target.shape[2],D,H,W) # (1,C,D,H,W)
        #         x_target = block_op(x_target, voxel_coords_target) # (1,C,N)
        #         x_target = x_target.squeeze(0).permute(1,0) # (N,C)
        #         x_target = x_target + point_feat_target[dgmn_cnt] # (N,C)
        #         dgmn_cnt +=1

        #     else:
        #         x_target = block_op(x_target, batch, 'target')

        # if self.training:
        #     loss_match, log_dict = self.hcc_matcher(skip_x_HCC, skip_x_HCC_target, batch, feat_size_list)
        #     return x, loss_match, log_dict
        # else:
        #     return x
        

    def loss(self, outputs, labels, output_feat=None):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0) # (1,8,N)
        target = target.unsqueeze(0).squeeze(-1) # (1,N)

        # Cross entropy loss
        ce_loss = self.criterion(outputs, target)

        ## Dice loss
        # num_seg = outputs.shape[1]
        # pred = torch.softmax(outputs,dim=1)
        # y_pred = pred # (1,8,N)
        # organ_target = torch.zeros((target.size(0), num_seg, target.size(1))) # (1,8,N)
        # for organ_index in range(num_seg):
        #     temp_target = torch.zeros(target.shape)
        #     temp_target[target == organ_index+1] = 1
        #     organ_target[:, organ_index, :] = temp_target
        # organ_target = organ_target.cuda()
        # dice = 0.0
        # for organ_index in range(num_seg):
        #     iflat = (y_pred[:, organ_index, :].contiguous().view(-1)).float()
        #     tflat = organ_target[:, organ_index, :].contiguous().view(-1)
        #     intersection = (iflat * tflat).sum()
        #     dice += 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum())

        # dice_loss = 1 - dice / (num_seg)

        ## extra loss
        # if self.training:
        #     '## Extra loss for sharpening boundaries'
        #     intra_loss, inter_loss = self.ceter_loss(output_feat, target.squeeze(0), ce_loss) # intra-class loss, inter-class loss

        #     alpha = 0.0
        #     # alpha = 0.1
        #     # alpha = 0.9
        #     # alpha = 1.0
        #     extra_loss = 0.01*(alpha*intra_loss + (1-alpha)*inter_loss)

        #     self.output_loss = ce_loss.mean() + extra_loss
        #     return self.output_loss, intra_loss, inter_loss
        # else:
        #     self.output_loss = ce_loss.mean()
        #     return self.output_loss

        self.output_loss = ce_loss.mean()
        return self.output_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target.squeeze(-1)).sum().item()

        return correct / total




criterion_cld = nn.CrossEntropyLoss().cuda()     # 会自动加上softmax 
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=8, feat_dim=64, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    def forward(self, x, labels, loss1):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).   (N,64)
            labels: ground truth labels with shape (batch_size).   (N)
        """
        # print("x.shape, labels.shape:", x.shape, labels.shape)
        # print("self.centers.shape:", self.centers.shape)
        ce_loss_1 = loss1.view(-1)
        ind_1_sorted = np.argsort(ce_loss_1.cpu().data).cuda()  # 从小到大排列，然后输出下标列表
        ce_loss_1_sorted = ce_loss_1[ind_1_sorted]
        # print("ce_loss_1_sorted:", ce_loss_1_sorted)     # 此时的loss为从小到大的排列顺序 
        num_remember = int(0.9 * len(ce_loss_1_sorted))

        ind_1_update = ind_1_sorted[num_remember:]
        
        # print("hard_x.shape, hard_labels.shape, ce_loss_1-ind_1_update-:", hard_x.shape, hard_labels.shape, ce_loss_1[ind_1_update])

        batch_size = x.size(0)
        # print(torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes).shape)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        """distmat = 1 * distmat - 2 * (x @ self.centers.t())"""
        distmat.addmm_(1, -2, x, self.centers.t())     # 加速计算欧氏距离，因为distmat已经是两个向量的平方和了，但中心是学习出来的
        # 注意上面的中心值是学习出来的，不是计算出的特征中心（为啥不可以计算出特征中心呢？浪费时间效率麽？）
        classes = torch.arange(1, self.num_classes+1).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        # print("batch_size, self.centers.shape, x.shape, cld_lab.shape:", batch_size, self.centers.shape, x.shape, labels[:,0].shape, labels[:8,0])

        # 按理说这里应该得到血管那些点和label，先不用计算血管来看看 
        hard_x = x[ind_1_update] 
        hard_labels = labels[ind_1_update]
        affnity = torch.mm(hard_x, self.centers.t())     # 其实也可以换为中心点自身的乘积，表示自身的相似性，使得中心点feature互相具有差异 
        CLD_loss = criterion_cld(affnity.div_(1), hard_labels[:,0])

        # print("affnity, classes:", affnity, classes)
        # CLD_loss = criterion_cld(affnity.div_(1), classes)
        # print("CLD_loss:", CLD_loss)
        return loss, CLD_loss
        """其实上面可以通过CEloss先得到loss较大的那些点，然后仅对那些点进行距离上的拉近以及鉴别，使得模型更好训练！！！"""
























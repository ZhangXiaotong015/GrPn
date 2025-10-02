# GrPn
The official implementation of 'Skip priors and add graph-based anatomical information, for point-based Couinaud segmentation'.

# Training
1. Training with MSD dataset:
```
python "/home/data1/liver_couinaud_segmentation/model/GRCNN/train_MSD.py" \
        --name 'MSD_res_64_64_64' \
        --dataset 'MSD' \
        --data_root '/data1/liver_couinaud_segmentation/data/MSD_Couinaud' \
        --first_subsampling_dl 0.0078125 \
        --fea_size 64 32 16 8 \
        --voxel_resolution 64 32 16 8
```
2. Training with LiTS dataset:
```
python "/home/data1/liver_couinaud_segmentation/model/GRCNN/train_LiTS.py" \
        --name 'LiTS_32_32_32' \
        --dataset 'LiTS' \
        --data_root '/data1/liver_couinaud_segmentation/data/LiTS_Couinaud' \
        --first_subsampling_dl 0.015625 \
        --fea_size 32 16 8 4 \
        --voxel_resolution 32 16 8 4
```

# Inference
1. Inference with MSD dataset:
```
python "/home/data1/liver_couinaud_segmentation/model/GRCNN/test_MSD.py" \
        --name 'Infer_MSD_res_64_64_64' \
        --dataset 'MSD' \
        --data_root '/data1/liver_couinaud_segmentation/data/MSD_Couinaud' \
        --first_subsampling_dl 0.0078125 \
        --fea_size 64 32 16 8 \
        --voxel_resolution 64 32 16 8
```
2. Inference with LiTS dataset:
```
python "/home/data1/liver_couinaud_segmentation/model/GRCNN/test_LiTS.py" \
        --name 'Infer_LiTS_res_32_32_32' \
        --dataset 'LiTS' \
        --data_root '/data1/liver_couinaud_segmentation/data/LiTS_Couinaud' \
        --first_subsampling_dl 0.015625 \
        --fea_size 32 16 8 4 \
        --voxel_resolution 32 16 8 4
```

# Citation
If you use this work, please cite:
```
@article{zhang2025skip,
  title={Skip priors and add graph-based anatomical information, for point-based Couinaud segmentation},
  author={Zhang, Xiaotong and Broersen, Alexander and van Erp, Gonnie and Pintea, Silvia L and Dijkstra, Jouke},
  journal={arXiv preprint arXiv:2508.01785},
  year={2025}
}
```

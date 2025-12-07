# GrPn
The official implementation of [Skip priors and add graph-based anatomical information, for point-based Couinaud segmentation](https://link.springer.com/chapter/10.1007/978-3-032-06103-4_13).
You can find the full article at [this link](https://arxiv.org/pdf/2508.01785).

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
@inproceedings{10.1007/978-3-032-06103-4_13,
author = {Zhang, Xiaotong and Broersen, Alexander and van Erp, Gonnie C. M. and Pintea, Silvia L. and Dijkstra, Jouke},
title = {Skip Priors and Add Graph-Based Anatomical Information, for Point-Based Couinaud Segmentation},
year = {2025},
isbn = {978-3-032-06102-7},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-032-06103-4_13},
doi = {10.1007/978-3-032-06103-4_13},
booktitle = {Reconstruction and Imaging Motion Estimation, and Graphs in Biomedical Image Analysis: First International Workshop, RIME 2025, and 7th International Workshop, GRAIL 2025, Daejeon, South Korea, September 27, 2025, Proceedings},
pages = {131–140},
numpages = {10},
keywords = {Couinaud segmentation, 3D graph reasoning, Point net},
location = {Daejeon, Korea (Republic of)}
}
```

# Acknowledgement
```
This work was performed using the compute resources from the Academic Leiden Interdisciplinary Cluster Environment (ALICE) provided by Leiden University.
```

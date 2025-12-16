#!/bin/bash
OUTPUT_DIR="/home/xzhang/inference_project/GrPn/output/Pre-ablation/Portal"
data_CT_point="/home/xzhang/inference_project/GrPn/data/CT_point"
data_CT_point_voxelidx="/home/xzhang/inference_project/GrPn/data/CT_point_voxelidx"
LPI_volume_dir="/home/xzhang/inference_project/GrPn/data/LPI"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$data_CT_point"
mkdir -p "$data_CT_point_voxelidx"
mkdir -p "$LPI_volume_dir"
docker run --rm --gpus "device=0" \
           --tmpfs /dev/shm:rw,noexec,nosuid,size=256m \
    --mount type=bind,src="$OUTPUT_DIR",dst=/output \
    --mount type=bind,src=/mnt/e/WSL/TestData/CouinaudSeg/Pre_CT/portal,dst=/data_CT,readonly \
    --mount type=bind,src=/mnt/e/WSL/TestData/CouinaudSeg/Pre_liverMask/Portal,dst=/data_liver,readonly \
    --mount type=bind,src="$data_CT_point",dst=/data_CT_point \
    --mount type=bind,src="$data_CT_point_voxelidx",dst=/data_CT_point_voxelidx \
    --mount type=bind,src="$LPI_volume_dir",dst=/LPI_volume_dir \
    -e MODEL_PATH=/app/model_weights/chkp_LiTS_median_interplanar_1mm.tar \
    couinaud_seg:latest \
    --out_dir /output \
    --data_CT /data_CT \
    --data_liver /data_liver \
    --data_root /data_CT_point \
    --data_root_voxelidx /data_CT_point_voxelidx \
    --LPI_volume_dir /LPI_volume_dir \
    --first_subsampling_dl 0.015625 \
    --fea_size 32 16 8 4 \
    --voxel_resolution 32 16 8 4

## Interactive
# OUTPUT_DIR="/home/xzhang/inference_project/GrPn/output/Pre-ablation/Portal"
# data_CT_point="/home/xzhang/inference_project/GrPn/data/CT_point"
# data_CT_point_voxelidx="/home/xzhang/inference_project/GrPn/data/CT_point_voxelidx"
# LPI_volume_dir="/home/xzhang/inference_project/GrPn/data/LPI"
# mkdir -p "$OUTPUT_DIR"
# mkdir -p "$data_CT_point"
# mkdir -p "$data_CT_point_voxelidx"
# mkdir -p "$LPI_volume_dir"
# docker run -it --rm --gpus "device=0" --tmpfs /dev/shm:rw,noexec,nosuid,size=256m \
#     --mount type=bind,src="$OUTPUT_DIR",dst=/output \
#     --mount type=bind,src=/mnt/e/WSL/TestData/CouinaudSeg/Pre_CT/portal,dst=/data_CT,readonly \
#     --mount type=bind,src=/mnt/e/WSL/TestData/CouinaudSeg/Pre_liverMask/Portal,dst=/data_liver,readonly \
#     --mount type=bind,src="$data_CT_point",dst=/data_CT_point \
#     --mount type=bind,src="$data_CT_point_voxelidx",dst=/data_CT_point_voxelidx \
#     --mount type=bind,src="$LPI_volume_dir",dst=/LPI_volume_dir \
#     -e MODEL_PATH=/app/model_weights/chkp_LiTS_median_interplanar_1mm.tar \
#     couinaud_seg:latest \
#     bash

##### All mounted paths must be exist!!!


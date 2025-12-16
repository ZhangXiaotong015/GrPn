echo "Using GPUs:"
nvidia-smi || echo "No GPU visible"

# --- Base directory for all Apptainer data ---
BASE=/data1/xzhang2/docker_archive

# --- 1. Set Apptainer tmp & cache to your large persistent directory ---
# export APPTAINER_TMPDIR=$BASE/apptainer_tmp
# export APPTAINER_CACHEDIR=$BASE/apptainer_cache
export TMPDIR=/tmp
export APPTAINER_TMPDIR=/tmp
export APPTAINER_CACHEDIR=/tmp

mkdir -p "$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_CACHEDIR"
# chmod 700 "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# --- 2. Where to store your .sif ---
export IMAGEDIR=$BASE/apptainer_images
mkdir -p "$IMAGEDIR"
# chmod 700 "$IMAGEDIR"

# --- 3. Path to your Docker archive (.tar) ---
TARFILE=$BASE/couinaud_seg.tar

# --- 4. Build SIF only if not already present ---
if [ ! -f "$IMAGEDIR/couinaud_seg.sif" ]; then
    echo "Building SIF from $TARFILE ..."
    apptainer build "$IMAGEDIR/couinaud_seg.sif" docker-archive://$TARFILE
fi

# --- 5. Prepare output and data directories ---
echo "Running container..."

OUT_BASE=/data1/xzhang2/docker_archive/CouinaudSeg_GrPn

OUTPUT_DIR="$OUT_BASE/output/Pre-ablation/Portal"
data_CT_point="$OUT_BASE/data/CT_point"
data_CT_point_voxelidx="$OUT_BASE/data/CT_point_voxelidx"
LPI_volume_dir="$OUT_BASE/data/LPI"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$data_CT_point"
mkdir -p "$data_CT_point_voxelidx"
mkdir -p "$LPI_volume_dir"

DATA_CT=/data1/xzhang2/docker_archive/data/CouinaudSeg/Pre_CT/portal
DATA_LIVER=/data1/xzhang2/docker_archive/data/CouinaudSeg/Pre_liverMask/portal

# --- 6. Run topkMIP container on GPU ---
apptainer run --nv \
    --bind "$OUTPUT_DIR:/output" \
    --bind "$DATA_CT:/data_CT:ro" \
    --bind "$DATA_LIVER:/data_liver:ro" \
    --bind "$data_CT_point:/data_CT_point" \
    --bind "$data_CT_point_voxelidx:/data_CT_point_voxelidx" \
    --bind "$LPI_volume_dir:/LPI_volume_dir" \
    --env MODEL_PATH=/app/model_weights/chkp_LiTS_median_interplanar_1mm.tar \
    "$IMAGEDIR/couinaud_seg.sif" \
        --out_dir /output \
        --data_CT /data_CT \
        --data_liver /data_liver \
        --data_root /data_CT_point \
        --data_root_voxelidx /data_CT_point_voxelidx \
        --LPI_volume_dir /LPI_volume_dir \
        --first_subsampling_dl 0.015625 \
        --fea_size 32 16 8 4 \
        --voxel_resolution 32 16 8 4




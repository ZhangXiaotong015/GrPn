FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -y tzdata

# Python3.8 is already default in Ubuntu 20.04
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-distutils python3-pip

RUN python3 -m pip install --upgrade pip

# CUDA include path
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64
ENV CPATH=$CUDA_HOME/include
ENV C_INCLUDE_PATH=$CUDA_HOME/include
ENV CPLUS_INCLUDE_PATH=$CUDA_HOME/include
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build libeigen3-dev \
    ffmpeg libsm6 libxext6 libgl1 vim && \
    rm -rf /var/lib/apt/lists/*

# Install requirements
WORKDIR /app
COPY requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy project
COPY inference.py /app/inference.py
COPY data_processing.py /app/data_processing.py
COPY models/ /app/models/
COPY datasets/ /app/datasets/
COPY modules/ /app/modules/
COPY utils/ /app/utils/
COPY cpp_wrappers/ /app/cpp_wrappers/
COPY model_weights/ /app/model_weights/

# Compile wrappers
RUN cd /app/cpp_wrappers && chmod +x compile_wrappers.sh && bash compile_wrappers.sh

# Compile DCN (Deformable Convolution)
RUN cd /app/modules/dgmn/dcn && \
    python3 setup.py build_ext --inplace


# CMD ["python3", "/app/inference.py"]
ENTRYPOINT ["python3", "/app/inference.py"]

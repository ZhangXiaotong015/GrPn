/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer ****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer ********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

// modify from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
#include <stdio.h>
#include <math.h>
#include <float.h>

using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename scalar_t>
__device__ scalar_t deformable_im2col_trilinear_3d(const scalar_t *bottom_data, 
                                                const int data_width, 
                                                const int data_height, 
                                                const int depth,
                                                const int height, 
                                                const int width, 
                                                scalar_t d, scalar_t h, scalar_t w) 
{
    int d_low = floor(d);
    int h_low = floor(h);
    int w_low = floor(w);
    int d_high = d_low + 1;
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    scalar_t ld = d - d_low, lh = h - h_low, lw = w - w_low;
    scalar_t hd = 1 - ld, hh = 1 - lh, hw = 1 - lw;

    scalar_t v000 = 0, v001 = 0, v010 = 0, v011 = 0;
    scalar_t v100 = 0, v101 = 0, v110 = 0, v111 = 0;

    if (d_low >= 0 && h_low >= 0 && w_low >= 0)
        v000 = bottom_data[(d_low * data_height + h_low) * data_width + w_low];
    if (d_low >= 0 && h_low >= 0 && w_high <= width - 1)
        v001 = bottom_data[(d_low * data_height + h_low) * data_width + w_high];
    if (d_low >= 0 && h_high <= height - 1 && w_low >= 0)
        v010 = bottom_data[(d_low * data_height + h_high) * data_width + w_low];
    if (d_low >= 0 && h_high <= height - 1 && w_high <= width - 1)
        v011 = bottom_data[(d_low * data_height + h_high) * data_width + w_high];
    if (d_high <= depth - 1 && h_low >= 0 && w_low >= 0)
        v100 = bottom_data[(d_high * data_height + h_low) * data_width + w_low];
    if (d_high <= depth - 1 && h_low >= 0 && w_high <= width - 1)
        v101 = bottom_data[(d_high * data_height + h_low) * data_width + w_high];
    if (d_high <= depth - 1 && h_high <= height - 1 && w_low >= 0)
        v110 = bottom_data[(d_high * data_height + h_high) * data_width + w_low];
    if (d_high <= depth - 1 && h_high <= height - 1 && w_high <= width - 1)
        v111 = bottom_data[(d_high * data_height + h_high) * data_width + w_high];

    scalar_t w000 = hd * hh * hw, w001 = hd * hh * lw;
    scalar_t w010 = hd * lh * hw, w011 = hd * lh * lw;
    scalar_t w100 = ld * hh * hw, w101 = ld * hh * lw;
    scalar_t w110 = ld * lh * hw, w111 = ld * lh * lw;

    scalar_t val = w000 * v000 + w001 * v001 + w010 * v010 + w011 * v011 +
                   w100 * v100 + w101 * v101 + w110 * v110 + w111 * v111;

    return val;
}

template <typename scalar_t>
__device__ scalar_t get_gradient_weight_3d(scalar_t argmax_d, scalar_t argmax_h, scalar_t argmax_w,
                                           const int d, const int h, const int w, 
                                           const int depth, const int height, const int width) 
{
    if (argmax_d <= -1 || argmax_d >= depth || 
        argmax_h <= -1 || argmax_h >= height || 
        argmax_w <= -1 || argmax_w >= width) 
    {
        return 0;
    }

    int argmax_d_low = floor(argmax_d);
    int argmax_h_low = floor(argmax_h);
    int argmax_w_low = floor(argmax_w);
    int argmax_d_high = argmax_d_low + 1;
    int argmax_h_high = argmax_h_low + 1;
    int argmax_w_high = argmax_w_low + 1;

    scalar_t weight = 0;

    if (d == argmax_d_low && h == argmax_h_low && w == argmax_w_low)
        weight = (d + 1 - argmax_d) * (h + 1 - argmax_h) * (w + 1 - argmax_w);
    if (d == argmax_d_low && h == argmax_h_low && w == argmax_w_high)
        weight = (d + 1 - argmax_d) * (h + 1 - argmax_h) * (argmax_w + 1 - w);
    if (d == argmax_d_low && h == argmax_h_high && w == argmax_w_low)
        weight = (d + 1 - argmax_d) * (argmax_h + 1 - h) * (w + 1 - argmax_w);
    if (d == argmax_d_low && h == argmax_h_high && w == argmax_w_high)
        weight = (d + 1 - argmax_d) * (argmax_h + 1 - h) * (argmax_w + 1 - w);
    if (d == argmax_d_high && h == argmax_h_low && w == argmax_w_low)
        weight = (argmax_d + 1 - d) * (h + 1 - argmax_h) * (w + 1 - argmax_w);
    if (d == argmax_d_high && h == argmax_h_low && w == argmax_w_high)
        weight = (argmax_d + 1 - d) * (h + 1 - argmax_h) * (argmax_w + 1 - w);
    if (d == argmax_d_high && h == argmax_h_high && w == argmax_w_low)
        weight = (argmax_d + 1 - d) * (argmax_h + 1 - h) * (w + 1 - argmax_w);
    if (d == argmax_d_high && h == argmax_h_high && w == argmax_w_high)
        weight = (argmax_d + 1 - d) * (argmax_h + 1 - h) * (argmax_w + 1 - w);

    return weight;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight_3d(scalar_t argmax_h, scalar_t argmax_w, scalar_t argmax_d,
                                            const int height, const int width, const int depth, 
                                            const scalar_t *im_data, const int data_width, const int data_depth,
                                            const int bp_dir)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width || argmax_d <= -1 || argmax_d >= depth)
  {
    // Empty, out of bounds
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_d_low = floor(argmax_d);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;
  int argmax_d_high = argmax_d_low + 1;

  scalar_t weight = 0;

  if (bp_dir == 0) // Backpropagation direction 0 (forward pass)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_d_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) * (argmax_d_low + 1 - argmax_d) * im_data[argmax_h_low * data_width * data_depth + argmax_w_low * data_depth + argmax_d_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1 && argmax_d_low >= 0)
      weight += -1 * (argmax_w - argmax_w_low) * (argmax_d_low + 1 - argmax_d) * im_data[argmax_h_low * data_width * data_depth + argmax_w_high * data_depth + argmax_d_low];
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_d_high <= depth - 1)
      weight += -1 * (argmax_w_low + 1 - argmax_w) * (argmax_d - argmax_d_low) * im_data[argmax_h_low * data_width * data_depth + argmax_w_low * data_depth + argmax_d_high];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1 && argmax_d_high <= depth - 1)
      weight += -1 * (argmax_w - argmax_w_low) * (argmax_d - argmax_d_low) * im_data[argmax_h_low * data_width * data_depth + argmax_w_high * data_depth + argmax_d_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0 && argmax_d_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) * (argmax_d_low + 1 - argmax_d) * im_data[argmax_h_high * data_width * data_depth + argmax_w_low * data_depth + argmax_d_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1 && argmax_d_low >= 0)
      weight += (argmax_w - argmax_w_low) * (argmax_d_low + 1 - argmax_d) * im_data[argmax_h_high * data_width * data_depth + argmax_w_high * data_depth + argmax_d_low];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0 && argmax_d_high <= depth - 1)
      weight += (argmax_w_low + 1 - argmax_w) * (argmax_d - argmax_d_low) * im_data[argmax_h_high * data_width * data_depth + argmax_w_low * data_depth + argmax_d_high];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1 && argmax_d_high <= depth - 1)
      weight += (argmax_w - argmax_w_low) * (argmax_d - argmax_d_low) * im_data[argmax_h_high * data_width * data_depth + argmax_w_high * data_depth + argmax_d_high];
  }
  else if (bp_dir == 1) // Backpropagation direction 1 (reverse pass)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_d_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * (argmax_w_low + 1 - argmax_w) * (argmax_d_low + 1 - argmax_d) * im_data[argmax_h_low * data_width * data_depth + argmax_w_low * data_depth + argmax_d_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1 && argmax_d_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * (argmax_w - argmax_w_low) * (argmax_d_low + 1 - argmax_d) * im_data[argmax_h_low * data_width * data_depth + argmax_w_high * data_depth + argmax_d_low];
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_d_high <= depth - 1)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * (argmax_w_low + 1 - argmax_w) * (argmax_d - argmax_d_low) * im_data[argmax_h_low * data_width * data_depth + argmax_w_low * data_depth + argmax_d_high];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1 && argmax_d_high <= depth - 1)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * (argmax_w - argmax_w_low) * (argmax_d - argmax_d_low) * im_data[argmax_h_low * data_width * data_depth + argmax_w_high * data_depth + argmax_d_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0 && argmax_d_low >= 0)
      weight += (argmax_h - argmax_h_low) * (argmax_w_low + 1 - argmax_w) * (argmax_d_low + 1 - argmax_d) * im_data[argmax_h_high * data_width * data_depth + argmax_w_low * data_depth + argmax_d_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1 && argmax_d_low >= 0)
      weight += (argmax_h - argmax_h_low) * (argmax_w - argmax_w_low) * (argmax_d_low + 1 - argmax_d) * im_data[argmax_h_high * data_width * data_depth + argmax_w_high * data_depth + argmax_d_low];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0 && argmax_d_high <= depth - 1)
      weight += (argmax_h - argmax_h_low) * (argmax_w_low + 1 - argmax_w) * (argmax_d - argmax_d_low) * im_data[argmax_h_high * data_width * data_depth + argmax_w_low * data_depth + argmax_d_high];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1 && argmax_d_high <= depth - 1)
      weight += (argmax_h - argmax_h_low) * (argmax_w - argmax_w_low) * (argmax_d - argmax_d_low) * im_data[argmax_h_high * data_width * data_depth + argmax_w_high * data_depth + argmax_d_high];
  }

  return weight;
}

template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel_3d(const int n, const scalar_t *data_im, const scalar_t *data_offset,
                                               const int height, const int width, const int depth, 
                                               const int kernel_h, const int kernel_w, const int kernel_d,
                                               const int pad_h, const int pad_w, const int pad_d,
                                               const int stride_h, const int stride_w, const int stride_d,
                                               const int dilation_h, const int dilation_w, const int dilation_d,
                                               const int channel_per_deformable_group,
                                               const int batch_size, const int num_channels, const int deformable_group,
                                               const int height_col, const int width_col, const int depth_col,
                                               scalar_t *data_col) 
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // index of the output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int d_col = (index / width_col / height_col) % depth_col;
    const int b_col = (index / width_col / height_col / depth_col) % batch_size;
    const int c_im = (index / width_col / height_col / depth_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w * kernel_d;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    const int d_in = d_col * stride_d - pad_d;

    scalar_t *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col * depth_col + w_col * depth_col + d_col;
    
    const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width * depth;
    const scalar_t *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 3 * kernel_h * kernel_w * kernel_d * height_col * width_col * depth_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        for (int k = 0; k < kernel_d; ++k)
        {
          const int data_offset_h_ptr = ((3 * (i * kernel_w * kernel_d + j * kernel_d + k)) * height_col + h_col) * width_col * depth_col + w_col * depth_col + d_col;
          const int data_offset_w_ptr = ((3 * (i * kernel_w * kernel_d + j * kernel_d + k) + 1) * height_col + h_col) * width_col * depth_col + w_col * depth_col + d_col;
          const int data_offset_d_ptr = ((3 * (i * kernel_w * kernel_d + j * kernel_d + k) + 2) * height_col + h_col) * width_col * depth_col + w_col * depth_col + d_col;
          
          const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
          const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
          const scalar_t offset_d = data_offset_ptr[data_offset_d_ptr];
          
          scalar_t val = static_cast<scalar_t>(0);

          const scalar_t h_im = h_in + i * dilation_h + offset_h;
          const scalar_t w_im = w_in + j * dilation_w + offset_w;
          const scalar_t d_im = d_in + k * dilation_d + offset_d;

          if (h_im > -1 && w_im > -1 && d_im > -1 && h_im < height && w_im < width && d_im < depth)
          {
            val = deformable_im2col_trilinear_3d(data_im_ptr, width, height, depth, height, width, h_im, w_im, d_im);
          }
          
          *data_col_ptr = val;
          data_col_ptr += batch_size * height_col * width_col * depth_col;
        }
      }
    }
  }
}

void deformable_im2col_3d(
    const at::Tensor data_im, const at::Tensor data_offset, const int channels,
    const int depth, const int height, const int width, const int ksize_d, const int ksize_h, const int ksize_w,
    const int pad_d, const int pad_h, const int pad_w, const int stride_d, const int stride_h, const int stride_w,
    const int dilation_d, const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, at::Tensor data_col) 
{
    // Calculate the output dimensions based on 3D inputs
    int depth_col = (depth + 2 * pad_d - (dilation_d * (ksize_d - 1) + 1)) / stride_d + 1;
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    
    // Total number of kernels to process
    int num_kernels = channels * depth_col * height_col * width_col * parallel_imgs;
    int channel_per_deformable_group = channels / deformable_group;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_im.type(), "deformable_im2col_3d_gpu", ([&] {
            const scalar_t *data_im_ = data_im.data<scalar_t>();
            const scalar_t *data_offset_ = data_offset.data<scalar_t>();
            scalar_t *data_col_ = data_col.data<scalar_t>();

            deformable_im2col_gpu_kernel_3d<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>( 
                num_kernels, data_im_, data_offset_, depth, height, width, ksize_d, ksize_h, ksize_w,
                pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, dilation_d, dilation_h, dilation_w,
                channel_per_deformable_group, parallel_imgs, channels, deformable_group,
                depth_col, height_col, width_col, data_col_); 
        }));

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in deformable_im2col_3d: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
__global__ void deformable_col2im_gpu_kernel_3d(
    const int n, const scalar_t *data_col, const scalar_t *data_offset,
    const int channels, const int height, const int width, const int depth,
    const int kernel_h, const int kernel_w, const int kernel_d,
    const int pad_h, const int pad_w, const int pad_d,
    const int stride_h, const int stride_w, const int stride_d,
    const int dilation_h, const int dilation_w, const int dilation_d,
    const int channel_per_deformable_group,
    const int batch_size, const int deformable_group,
    const int height_col, const int width_col, const int depth_col,
    scalar_t *grad_im)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        const int j = (index / depth_col / width_col / height_col / batch_size) % kernel_w;
        const int i = (index / depth_col / width_col / height_col / batch_size / kernel_w) % kernel_h;
        const int k = (index / depth_col / width_col / height_col / batch_size / kernel_w / kernel_h) % kernel_d;
        const int c = index / depth_col / width_col / height_col / batch_size / kernel_w / kernel_h / kernel_d;
        const int deformable_group_index = c / channel_per_deformable_group;

        // Compute the start and end of the output location
        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        int d_out = (index / width_col / height_col) % depth_col;
        int b = (index / width_col / height_col / depth_col) % batch_size;
        int w_in = w_out * stride_w - pad_w;
        int h_in = h_out * stride_h - pad_h;
        int d_in = d_out * stride_d - pad_d;

        // Accessing offset data
        const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) *
                                                        3 * kernel_h * kernel_w * kernel_d * height_col * width_col * depth_col;
        const int data_offset_h_ptr = ((3 * (i * kernel_w * kernel_d + j * kernel_d + k)) * height_col + h_out) * width_col * depth_col + w_out * depth_col + d_out;
        const int data_offset_w_ptr = ((3 * (i * kernel_w * kernel_d + j * kernel_d + k) + 1) * height_col + h_out) * width_col * depth_col + w_out * depth_col + d_out;
        const int data_offset_d_ptr = ((3 * (i * kernel_w * kernel_d + j * kernel_d + k) + 2) * height_col + h_out) * width_col * depth_col + w_out * depth_col + d_out;
        
        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        const scalar_t offset_d = data_offset_ptr[data_offset_d_ptr];
        
        const scalar_t cur_inv_h_data = h_in + i * dilation_h + offset_h;
        const scalar_t cur_inv_w_data = w_in + j * dilation_w + offset_w;
        const scalar_t cur_inv_d_data = d_in + k * dilation_d + offset_d;

        const scalar_t cur_top_grad = data_col[index];
        const int cur_h = (int)cur_inv_h_data;
        const int cur_w = (int)cur_inv_w_data;
        const int cur_d = (int)cur_inv_d_data;

        for (int dz = -2; dz <= 2; dz++)
        {
            for (int dy = -2; dy <= 2; dy++)
            {
                for (int dx = -2; dx <= 2; dx++)
                {
                    if (cur_h + dy >= 0 && cur_h + dy < height &&
                        cur_w + dx >= 0 && cur_w + dx < width &&
                        cur_d + dz >= 0 && cur_d + dz < depth &&
                        abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
                        abs(cur_inv_w_data - (cur_w + dx)) < 1 &&
                        abs(cur_inv_d_data - (cur_d + dz)) < 1)
                    {
                        int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width * depth + (cur_w + dx) * depth + (cur_d + dz);
                        scalar_t weight = get_gradient_weight_3d(cur_inv_h_data, cur_inv_w_data, cur_inv_d_data, cur_h + dy, cur_w + dx, cur_d + dz, height, width, depth);
                        atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
                    }
                }
            }
        }
    }
}

void deformable_col2im_3d(
    const at::Tensor data_col, const at::Tensor data_offset, const int channels,
    const int height, const int width, const int depth, const int ksize_h,
    const int ksize_w, const int ksize_d, const int pad_h, const int pad_w, const int pad_d,
    const int stride_h, const int stride_w, const int stride_d,
    const int dilation_h, const int dilation_w, const int dilation_d,
    const int parallel_imgs, const int deformable_group,
    at::Tensor grad_im)
{
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int depth_col = (depth + 2 * pad_d - (dilation_d * (ksize_d - 1) + 1)) / stride_d + 1;
    
    int num_kernels = channels * ksize_h * ksize_w * ksize_d * height_col * width_col * depth_col * parallel_imgs;
    int channel_per_deformable_group = channels / deformable_group;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_col.type(), "deformable_col2im_gpu_3d", ([&] {
            const scalar_t *data_col_ = data_col.data<scalar_t>();
            const scalar_t *data_offset_ = data_offset.data<scalar_t>();
            scalar_t *grad_im_ = grad_im.data<scalar_t>();

            deformable_col2im_gpu_kernel_3d<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                num_kernels, data_col_, data_offset_, channels, height, width, depth,
                ksize_h, ksize_w, ksize_d, pad_h, pad_w, pad_d,
                stride_h, stride_w, stride_d, dilation_h, dilation_w, dilation_d,
                channel_per_deformable_group, parallel_imgs, deformable_group,
                height_col, width_col, depth_col, grad_im_);
        }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in deformable_col2im_3d: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
__global__ void deformable_col2im_coord_gpu_kernel_3d(const int n, const scalar_t *data_col,
                                                     const scalar_t *data_im, const scalar_t *data_offset,
                                                     const int channels, const int height, const int width,
                                                     const int depth,
                                                     const int kernel_h, const int kernel_w, const int kernel_d,
                                                     const int pad_h, const int pad_w, const int pad_d,
                                                     const int stride_h, const int stride_w, const int stride_d,
                                                     const int dilation_h, const int dilation_w, const int dilation_d,
                                                     const int channel_per_deformable_group,
                                                     const int batch_size, const int offset_channels, const int deformable_group,
                                                     const int height_col, const int width_col, const int depth_col,
                                                     scalar_t *grad_offset)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        scalar_t val = 0;
        int w = index % width_col;
        int h = (index / width_col) % height_col;
        int d = (index / (width_col * height_col)) % depth_col;
        int c = (index / (width_col * height_col * depth_col)) % offset_channels;
        int b = (index / (width_col * height_col * depth_col)) / offset_channels;

        const int deformable_group_index = c / (2 * kernel_h * kernel_w * kernel_d);
        const int col_step = kernel_h * kernel_w * kernel_d;
        int cnt = 0;

        const scalar_t *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group *
                                                    batch_size * width_col * height_col * depth_col;
        const scalar_t *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) *
                                                  channel_per_deformable_group / kernel_h / kernel_w / kernel_d * height * width * depth;
        const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 *
                                                        kernel_h * kernel_w * kernel_d * height_col * width_col * depth_col;

        const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w * kernel_d;

        for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
        {
            const int col_pos = (((col_c * batch_size + b) * height_col + h) * width_col + w) * depth_col + d;
            const int bp_dir = offset_c % 2;

            int j = (col_pos / (width_col * height_col * depth_col * batch_size)) % kernel_w;
            int i = (col_pos / (width_col * height_col * depth_col * batch_size * kernel_w)) % kernel_h;
            int k = (col_pos / (width_col * height_col * depth_col * batch_size * kernel_w * kernel_h)) % kernel_d;

            int w_out = col_pos % width_col;
            int h_out = (col_pos / width_col) % height_col;
            int d_out = (col_pos / (width_col * height_col)) % depth_col;

            int w_in = w_out * stride_w - pad_w;
            int h_in = h_out * stride_h - pad_h;
            int d_in = d_out * stride_d - pad_d;

            const int data_offset_h_ptr = (((2 * (i * kernel_w * kernel_d + j * kernel_d + k)) * height_col + h_out) * width_col * depth_col + w_out * depth_col + d_out);
            const int data_offset_w_ptr = (((2 * (i * kernel_w * kernel_d + j * kernel_d + k) + 1) * height_col + h_out) * width_col * depth_col + w_out * depth_col + d_out);
            const int data_offset_d_ptr = (((2 * (i * kernel_w * kernel_d + j * kernel_d + k) + 2) * height_col + h_out) * width_col * depth_col + w_out * depth_col + d_out);

            const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
            const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
            const scalar_t offset_d = data_offset_ptr[data_offset_d_ptr];

            scalar_t inv_h = h_in + i * dilation_h + offset_h;
            scalar_t inv_w = w_in + j * dilation_w + offset_w;
            scalar_t inv_d = d_in + k * dilation_d + offset_d;

            if (inv_h <= -1 || inv_w <= -1 || inv_d <= -1 || inv_h >= height || inv_w >= width || inv_d >= depth)
            {
                inv_h = inv_w = inv_d = -2;
            }

            const scalar_t weight = get_coordinate_weight_3d(
                inv_h, inv_w, inv_d,
                height, width, depth,
                data_im_ptr + cnt * height * width * depth, width, depth, bp_dir);

            val += weight * data_col_ptr[col_pos];
            cnt += 1;
        }

        grad_offset[index] = val;
    }
}

void deformable_col2im_coord_3d(
    const at::Tensor data_col, const at::Tensor data_im, const at::Tensor data_offset,
    const int channels, const int height, const int width, const int depth,
    const int ksize_h, const int ksize_w, const int ksize_d,
    const int pad_h, const int pad_w, const int pad_d,
    const int stride_h, const int stride_w, const int stride_d,
    const int dilation_h, const int dilation_w, const int dilation_d,
    const int parallel_imgs, const int deformable_group, at::Tensor grad_offset)
{
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int depth_col = (depth + 2 * pad_d - (dilation_d * (ksize_d - 1) + 1)) / stride_d + 1;

    int num_kernels = height_col * width_col * depth_col * 2 * ksize_h * ksize_w * ksize_d * deformable_group * parallel_imgs;
    int channel_per_deformable_group = channels * ksize_h * ksize_w * ksize_d / deformable_group;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_col.type(), "deformable_col2im_coord_gpu_3d", ([&] {
            const scalar_t *data_col_ = data_col.data<scalar_t>();
            const scalar_t *data_im_ = data_im.data<scalar_t>();
            const scalar_t *data_offset_ = data_offset.data<scalar_t>();
            scalar_t *grad_offset_ = grad_offset.data<scalar_t>();

            deformable_col2im_coord_gpu_kernel_3d<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>( 
                num_kernels, data_col_, data_im_, data_offset_, channels, height, width, depth,
                ksize_h, ksize_w, ksize_d, pad_h, pad_w, pad_d, stride_h, stride_w, stride_d, 
                dilation_h, dilation_w, dilation_d, channel_per_deformable_group, parallel_imgs, 
                2 * ksize_h * ksize_w * ksize_d * deformable_group, deformable_group,
                height_col, width_col, depth_col, grad_offset_);
        }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in deformable_col2im_coord_3d: %s\n", cudaGetErrorString(err));
    }
}

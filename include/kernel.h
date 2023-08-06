/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "common.h"

const int NMS_THREADS_PER_BLOCK = sizeof(uint64_t) * 8;
const int THREADS_FOR_VOXEL = 28;

#define DIVUP(x, y) (x + y - 1) / y

cudaError_t voxelizationLaunch(const float *points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float voxel_x_size, float voxel_y_size, float voxel_z_size,
        int grid_y_size, int grid_x_size, int feature_num,
	int max_voxels, int max_points_voxel,
        unsigned int *hash_table,
	unsigned int *num_points_per_voxel, float *voxel_features,
	unsigned int *voxel_indices, unsigned int *real_voxel_num,
        cudaStream_t stream = 0);

cudaError_t featureExtractionLaunch(float *voxels_temp_,
	unsigned int *num_points_per_voxel,
        const unsigned int real_voxel_num, int max_points_per_voxel,
	int feature_num, half *voxel_features, cudaStream_t stream_ = 0);


#endif
#ifndef COMMON_H_
#define COMMON_H_

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <numeric>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>

const unsigned int MAX_DET_NUM = 1000;           // nms_pre_max_size = 1000;
const unsigned int DET_CHANNEL = 11;
const unsigned int MAX_POINTS_NUM = 300000;
const unsigned int NUM_TASKS = 6;

#define checkCudaErrors(op)                                                                  \
  {                                                                                          \
    auto status = ((op));                                                                    \
    if (status != 0) {                                                                       \
      std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                << ":" << __LINE__ << " error status: " << status << std::endl;              \
      abort();                                                                               \
    }                                                                                        \
  }


class Params
{
  public:
    const unsigned int task_num_stride[NUM_TASKS] = { 0, 1, 3, 5, 6, 8, };

    const float out_size_factor = 8;
    const float voxel_size[2] = { 0.075, 0.075, };
    const float pc_range[2] = { -54, -54, };
    const float score_threshold = 0.1;
    const float post_center_range[6] = { -61.2, -61.2, -10.0, 61.2, 61.2, 10.0, };

    const float min_x_range = -54;
    const float max_x_range = 54;
    const float min_y_range = -54;
    const float max_y_range = 54;
    const float min_z_range = -5.0;
    const float max_z_range = 3.0;
    
    // the size of a pillar
    const float pillar_x_size = 0.075;
    const float pillar_y_size = 0.075;
    const float pillar_z_size = 0.2;
    const int max_points_per_voxel = 10;

    const unsigned int max_voxels = 160000;
    const unsigned int feature_num = 5;

    Params() {};

    int getGridXSize() {
      return (int)std::round((max_x_range - min_x_range) / pillar_x_size);
    }
    int getGridYSize() {
      return (int)std::round((max_y_range - min_y_range) / pillar_y_size);
    }
    int getGridZSize() {
      return (int)std::round((max_z_range - min_z_range) / pillar_z_size);
    }
};

#endif
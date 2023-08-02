#include "kernel.h"

__device__ inline uint64_t hash(uint64_t k) {
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k;
}

__device__ inline void insertHashTable(const uint32_t key, uint32_t *value,
		const uint32_t hash_size, uint32_t *hash_table) {
  
  // The applies hash function on the key ( which is voxel_offset ).
  // The hash table is an array with all keys stored first followed by values.
  // For ex : [k1,k2,k3...,v1,v2,v3..]
  // The hash value is used to find the slot in the hash_map using mod.
  // Divide by 2 because it stores both keys and values.
  // For slot, we apply atomic compare and swap function.
  // It returns the current key in the slot. 3 possibilities :
  // 1. If empty, that means the insertion was successfull. We then insert corresponding value
  //    which is the unique voxel index.
  // 2. If key matches current key, it means this voxel offset is already in hash table.
  // 3. If key is another key, that is a collision, which means a different key with same hash value eixsts in table.
  //    Apply linear probing to solve this problem. Check in the next slot.
  uint64_t hash_value = hash(key);
  uint32_t slot = hash_value % (hash_size / 2)/*key, value*/;
  uint32_t empty_key = UINT32_MAX;
  while (true) {
     uint32_t pre_key = atomicCAS(hash_table + slot, empty_key, key);
     if (pre_key == empty_key) {
       hash_table[slot + hash_size / 2 /*offset*/] = atomicAdd(value, 1);
       break;
     } else if (pre_key == key) {
       break;
     }
     slot = (slot + 1) % (hash_size / 2);
  }
}

__device__ inline uint32_t lookupHashTable(const uint32_t key, const uint32_t hash_size, const uint32_t *hash_table) {
  uint64_t hash_value = hash(key);
  uint32_t slot = hash_value % (hash_size / 2)/*key, value*/;
  uint32_t empty_key = UINT32_MAX;
  int cnt = 0;
  while (cnt < 100 /* need to be adjusted according to data*/) {
    cnt++;
    if (hash_table[slot] == key) {
      return hash_table[slot + hash_size / 2];
    } else if (hash_table[slot] == empty_key) {
      return empty_key;
    } else {
      slot = (slot + 1) % (hash_size / 2);
    }
  }
  return empty_key;
}

__global__ void buildHashKernel(const float *points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float voxel_x_size, float voxel_y_size, float voxel_z_size,
        int grid_y_size, int grid_x_size, int feature_num,
	unsigned int *hash_table, unsigned int *real_voxel_num) {
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= points_size) {
    return;
  }
  
  float px = points[feature_num * point_idx];
  float py = points[feature_num * point_idx + 1];
  float pz = points[feature_num * point_idx + 2];

  if( px < min_x_range || px >= max_x_range || py < min_y_range || py >= max_y_range
    || pz < min_z_range || pz >= max_z_range) {
    return;
  }

  unsigned int voxel_idx = floorf((px - min_x_range) / voxel_x_size);
  unsigned int voxel_idy = floorf((py - min_y_range) / voxel_y_size);
  unsigned int voxel_idz = floorf((pz - min_z_range) / voxel_z_size);
  unsigned int voxel_offset = voxel_idz * grid_y_size * grid_x_size
	                    + voxel_idy * grid_x_size
                            + voxel_idx;
  insertHashTable(voxel_offset, real_voxel_num, points_size * 2 * 2, hash_table);
}

__global__ void voxelizationKernel(const float *points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float voxel_x_size, float voxel_y_size, float voxel_z_size,
        int grid_y_size, int grid_x_size, int feature_num, int max_voxels,
        int max_points_per_voxel,
	unsigned int *hash_table, unsigned int *num_points_per_voxel,
	float *voxels_temp, unsigned int *voxel_indices, unsigned int *real_voxel_num) {
  
  // give every point to a thread. Find the index of the current point within this kernel.
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= points_size) {
    return;
  }
  
  // points is the array of points in point cloud storing point features (x,y,z,intensity,t) 
  // in a serialized formaat. We access px,py,pz of this point now.
  // feature_num is 5
  float px = points[feature_num * point_idx];
  float py = points[feature_num * point_idx + 1];
  float pz = points[feature_num * point_idx + 2];

  // If the point is outside the range along (x,y,z) dim then just stop further 
  // processing and return.
  if( px < min_x_range || px >= max_x_range || py < min_y_range || py >= max_y_range
    || pz < min_z_range || pz >= max_z_range) {
    return;
  }

  // Now find the voxel id for this point using the usual voxel conversion logic.
  unsigned int voxel_idx = floorf((px - min_x_range) / voxel_x_size);
  unsigned int voxel_idy = floorf((py - min_y_range) / voxel_y_size);
  unsigned int voxel_idz = floorf((pz - min_z_range) / voxel_z_size);
  // Now find the voxel offset, which is the index of the voxel if all voxels are flattened.
  unsigned int voxel_offset = voxel_idz * grid_y_size * grid_x_size
	                    + voxel_idy * grid_x_size
                            + voxel_idx;
  
  // The hash table we built earlier contains the mapping from each unique voxel_offset 
  // to the corresponding voxel id. Lets say, in 2D space we have 100x100 voxels but only
  // 200 voxels are occupied. Then instead of storing and searching information in 100x100 
  // space, using hashmaps allows us to lookup only in the constrained 200 sized space. 
  // This makes processing of voxel information faster, in cases when point cloud is sparse.
  // This makes it efficient in both memory and time compexity.
  // scatter to voxels
  unsigned int voxel_id = lookupHashTable(voxel_offset, points_size * 2 * 2, hash_table);
  // If the current voxel id is greater than max_voxels, simply return.
  if (voxel_id >= max_voxels) {
    return;
  }
  
  // num_points_per_voxel is an array of size of total valid voxels,
  // and stores count of how many points per voxel exist.
  // here we increment the number of points in current voxel id by 1.
  unsigned int current_num = atomicAdd(num_points_per_voxel + voxel_id, 1);

  // Now we copy the current points features (x,y,z,i,t) to the voxel_temp array 
  // in its correct position. voxel_temp array stores all points features of all voxels,
  // in a serialized fashion in order of the voxel_id. For for example, for first voxel, ( if max points per voxel = 3),
  // voxel_temp stores [x11,y11,z11,i11,t11,x12,y12,z12,i12,t12,x13,y13,z13,i13,t13]. where i featureij indicates voxelid and j indiciates pointid i
  // in that voxel. We define src_offset and dst_offset using point_idx and voxel_id respectively.
  // Here is where the assignment of points from point array to voxel_temp array happens. 
  if (current_num < max_points_per_voxel) {
    unsigned int dst_offset = voxel_id * (feature_num * max_points_per_voxel) + current_num * feature_num;
    unsigned int src_offset = point_idx * feature_num;
    for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
      voxels_temp[dst_offset + feature_idx] = points[src_offset + feature_idx];
    }
    // 
    // now only deal with batch_size = 1
    // since not sure what the input format will be if batch size > 1
    // voxel_indices stores 3D voxel poistions (b,x,y,z) where b is batch_size.
    // The voxel_indices stores this information serially in order of the voxel_id.
    uint4 idx = {0, voxel_idz, voxel_idy, voxel_idx};
    ((uint4 *)voxel_indices)[voxel_id] = idx;

  }
}

__global__ void featureExtractionKernel(float *voxels_temp,
		unsigned int *num_points_per_voxel,
		int max_points_per_voxel, int feature_num, half *voxel_features) {
  // Apply feature extraction on voxels by using one thread for each voxel
  // Get voxel_id using block and thread id.
  int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // If number_points_per_voxel is greater than max_points_per_voxel, clip the value
  num_points_per_voxel[voxel_idx] = num_points_per_voxel[voxel_idx] > max_points_per_voxel ?
	                                          max_points_per_voxel :  num_points_per_voxel[voxel_idx];
  // Get number of points for this particular voxel
  int valid_points_num = num_points_per_voxel[voxel_idx];

  // To extract points features for this voxel from voxel_temp,
  // calculate offset (index offset from index 0 in voxel_temp)
  int offset = voxel_idx * max_points_per_voxel * feature_num;
  // Now the goal is to take average for each feature (x,y,z,i,t) of every point in this voxel.
  // For each voxel in voxel_temp, the first point feature info is updated by summing up
  // info for all other points. 
  // offset is id of voxel, 
  for (int feature_idx = 0; feature_idx< feature_num; ++feature_idx) {
    for (int point_idx = 0; point_idx < valid_points_num - 1; ++point_idx) {
      voxels_temp[offset + feature_idx] += voxels_temp[offset + (point_idx + 1) * feature_num + feature_idx];
    }
    voxels_temp[offset + feature_idx] /= valid_points_num;
  }

  // The voxel_features array stores the averaged out voxel features
  // from voxel_temp in a contiguous manner. The features are converted to "half"
  // for memory efficiency.
  // move to be continuous
  for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
    int dst_offset = voxel_idx * feature_num;
    int src_offset = voxel_idx * feature_num * max_points_per_voxel;
    voxel_features[dst_offset + feature_idx] = __float2half(voxels_temp[src_offset + feature_idx]);
  }
}

cudaError_t featureExtractionLaunch(float *voxels_temp, unsigned int *num_points_per_voxel,
        const unsigned int real_voxel_num, int max_points_per_voxel, int feature_num,
	half *voxel_features, cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((real_voxel_num + threadNum - 1) / threadNum);
  dim3 threads(threadNum);
  featureExtractionKernel<<<blocks, threads, 0, stream>>>
    (voxels_temp, num_points_per_voxel,
        max_points_per_voxel, feature_num, voxel_features);
  cudaError_t err = cudaGetLastError();
  return err;
}

cudaError_t voxelizationLaunch(const float *points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float voxel_x_size, float voxel_y_size, float voxel_z_size,
        int grid_y_size, int grid_x_size, int feature_num, int max_voxels,
	int max_points_per_voxel,
	unsigned int *hash_table, unsigned int *num_points_per_voxel,
	float *voxel_features, unsigned int *voxel_indices,
	unsigned int *real_voxel_num, cudaStream_t stream)
{
  // how many threads in each block
  int threadNum = THREADS_FOR_VOXEL;
  // how many blocks needed if each point gets on thread.
  dim3 blocks((points_size+threadNum-1)/threadNum);
  // how many threads in each block
  dim3 threads(threadNum);
  // how many blocks needed to launch the kernel, how many threads in each block,
  // how many bytes for dynamic shared memory  ( zero here), cuda stream
  buildHashKernel<<<blocks, threads, 0, stream>>>
    (points, points_size,
        min_x_range, max_x_range,
        min_y_range, max_y_range,
        min_z_range, max_z_range,
        voxel_x_size, voxel_y_size, voxel_z_size,
        grid_y_size, grid_x_size, feature_num, hash_table,
	real_voxel_num);
  voxelizationKernel<<<blocks, threads, 0, stream>>>
    (points, points_size,
        min_x_range, max_x_range,
        min_y_range, max_y_range,
        min_z_range, max_z_range,
        voxel_x_size, voxel_y_size, voxel_z_size,
        grid_y_size, grid_x_size, feature_num, max_voxels,
        max_points_per_voxel, hash_table,
	num_points_per_voxel, voxel_features, voxel_indices, real_voxel_num);
  cudaError_t err = cudaGetLastError();
  return err;
}
#ifndef VOXELIZER_CPU_H
#define VOXELIZER_CPU_H

#include <vector>
#include <array> // Added for std::array
#include "common.h"

// Define a structure to represent a point in the point cloud
struct Point {
    float x;
    float y;
    float z;
    float intensity; // Assuming intensity is a feature of the point
};

// Define a structure to represent a voxel
struct Voxel {
    std::vector<Point> points;
};

class VoxelizerCPU {
private:
    Params params_;

public:
    VoxelizerCPU();
    ~VoxelizerCPU();
    std::vector<Voxel> voxelization(const float* points, size_t num_points);
    std::vector<std::array<float, 4>> calculateVoxelFeatures(const std::vector<Voxel>& voxels);
};

#endif // VOXELIZER_CPU_H

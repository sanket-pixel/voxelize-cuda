#include "VoxelizerCPU.hpp"

VoxelizerCPU::VoxelizerCPU() {}

VoxelizerCPU::~VoxelizerCPU() {}

std::vector<Voxel> VoxelizerCPU::voxelization(const float* points_data, size_t num_points) {
    std::vector<Point> points;

    // Convert the points_data to a vector of Point
    for (size_t i = 0; i < num_points; ++i) {
        Point point;
        int offset = params_.feature_num * i;
        point.x = points_data[offset];
        point.y = points_data[offset + 1];
        point.z = points_data[offset + 2];
        point.intensity = points_data[offset + 3];
        points.push_back(point);
    }

    // Calculate the number of voxels in each dimension
    int num_voxels_x = params_.getGridXSize();
    int num_voxels_y = params_.getGridYSize();
    int num_voxels_z = params_.getGridZSize();

    // Create a 3D grid of voxels
    std::vector<std::vector<std::vector<Voxel>>> voxel_grid(num_voxels_x, std::vector<std::vector<Voxel>>(num_voxels_y, std::vector<Voxel>(num_voxels_z)));

    // Loop through each point in the point cloud
    for (const Point& point : points) {
        // Calculate the voxel indices for this point
        int voxel_x = (point.x - params_.min_x_range) / params_.pillar_x_size;
        int voxel_y = (point.y - params_.min_y_range) / params_.pillar_y_size;
        int voxel_z = (point.z - params_.min_z_range) / params_.pillar_z_size;

        // Check if the voxel indices are within the grid bounds
        if (voxel_x >= 0 && voxel_x < num_voxels_x &&
            voxel_y >= 0 && voxel_y < num_voxels_y &&
            voxel_z >= 0 && voxel_z < num_voxels_z) {
            // Add the point to the corresponding voxel in the grid
            voxel_grid[voxel_x][voxel_y][voxel_z].points.push_back(point);
        }
    }

    // Flatten the voxel grid to get a 1D array of voxels
    std::vector<Voxel> voxels;
    for (int x = 0; x < num_voxels_x; ++x) {
        for (int y = 0; y < num_voxels_y; ++y) {
            for (int z = 0; z < num_voxels_z; ++z) {
                // Check if the voxel is not empty and does not exceed the maximum number of points per voxel
                if (!voxel_grid[x][y][z].points.empty() && voxel_grid[x][y][z].points.size() <= params_.max_points_per_voxel) {
                    // Add the voxel to the list of voxels
                    voxels.push_back(voxel_grid[x][y][z]);
                }
            }
        }
    }

    // Return the list of voxels
    return voxels;
}



std::vector<std::array<float, 4>> VoxelizerCPU::calculateVoxelFeatures(const std::vector<Voxel>& voxels) {
    std::vector<std::array<float, 4>> voxel_features;
    voxel_features.reserve(voxels.size());

    for (const Voxel& voxel : voxels) {
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;
        float sum_intensity = 0.0f;

        size_t num_points_in_voxel = voxel.points.size();
        for (const Point& point : voxel.points) {
            sum_x += point.x;
            sum_y += point.y;
            sum_z += point.z;
            sum_intensity += point.intensity;
        }

        if (num_points_in_voxel > 0) {
            sum_x /= static_cast<float>(num_points_in_voxel);
            sum_y /= static_cast<float>(num_points_in_voxel);
            sum_z /= static_cast<float>(num_points_in_voxel);
            sum_intensity /= static_cast<float>(num_points_in_voxel);
        }

        std::array<float, 4> feature_array = { sum_x, sum_y, sum_z, sum_intensity };
        voxel_features.push_back(feature_array);
    }

    return voxel_features;
}

#include "visualizer.hpp"

Visualizer::Visualizer(const std::string& id, int num_points, float* data, int dim)
    : id(id), num_points(num_points), data(data), dim(dim) {}

Visualizer::~Visualizer(){

}

void Visualizer::initialize() {
    cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    viewer.reset(new pcl::visualization::CloudViewer(id));
}

void Visualizer::populate_cloud() {
    cloud->width = num_points;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    for (size_t i = 0; i < num_points; ++i) {
        int offset = dim * i;
        cloud->points[i].x = data[offset];
        cloud->points[i].y = data[offset + 1];
        cloud->points[i].z = data[offset + 2];
        cloud->points[i].intensity = data[offset + 3];
    }
}

void Visualizer::show_cloud(){

    viewer->showCloud(cloud);
    while (!viewer->wasStopped()) {}

}

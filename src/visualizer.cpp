#include "visualizer.hpp"
#include <chrono>
#include <thread>



Visualizer::Visualizer(const std::string& id, int dim)
    : id(id), dim(dim) {
            viewer.reset(new pcl::visualization::PCLVisualizer(id));

    }

Visualizer::~Visualizer(){

}

void Visualizer::initialize() {
    cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);

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
        // viewer->showCloud(cloud);
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_color_handler(cloud, "intensity");

        viewer->addPointCloud<pcl::PointXYZI>(cloud, intensity_color_handler, "cloud");

        // Update the viewer
        viewer->spinOnce(100); // Show the point cloud for 10 milliseconds

        // Remove the point cloud from the viewer
        viewer->removePointCloud("cloud");

        // viewer->close();
        
}


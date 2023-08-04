#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>


class Visualizer{
    private :
        pcl::visualization::CloudViewer::Ptr viewer;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
        std::string id="";
        int num_points=0;
        float* data;
        int dim=5;

    public:
        Visualizer(const std::string& id, const int num_points, float* data, const int dim);
        ~Visualizer();
        void initialize();
        void populate_cloud();
        void show_cloud();

};
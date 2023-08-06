#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>


class Visualizer{
    private :
        pcl::visualization::PCLVisualizer::Ptr viewer;
        std::string id="";
        int dim=5;

    public:
        int num_points=0;
        float* data;
        Visualizer(const std::string& id, const int dim);
        ~Visualizer();
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
        void initialize();
        void populate_cloud();
        void show_cloud();

};
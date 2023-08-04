#include <sstream>
#include <fstream>
#include <stdio.h>
#include <fstream>
#include <memory>
#include <chrono>
#include <dirent.h>
#include "common.h"
#include "preprocess.h"
#include "VoxelizerCPU.hpp"
#include "visualizer.hpp"
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <cuda_fp16.h>

void GetDeviceInfo()
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
        printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    printf("\n");
}

bool hasEnding(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

int getFolderFile(const char *path, std::vector<std::string>& files, const char *suffix = ".bin")
{
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path)) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            std::string file = ent->d_name;
            if(hasEnding(file, suffix)){
                files.push_back(file.substr(0, file.length()-4));
            }
        }
        closedir(dir);
    } else {
        printf("No such folder: %s.", path);
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}

int loadData(const char *file, void **data, unsigned int *length)
{
    std::fstream dataFile(file, std::ifstream::in);

    if (!dataFile.is_open()) {
        std::cout << "Can't open files: "<< file<<std::endl;
        return -1;
    }

    unsigned int len = 0;
    dataFile.seekg (0, dataFile.end);
    len = dataFile.tellg();
    dataFile.seekg (0, dataFile.beg);

    char *buffer = new char[len];
    if (buffer==NULL) {
        std::cout << "Can't malloc buffer."<<std::endl;
        dataFile.close();
        exit(EXIT_FAILURE);
    }

    dataFile.read(buffer, len);
    dataFile.close();

    *data = (void*)buffer;
    *length = len;
    return 0;  
}


static bool startswith(const char *s, const char *with, const char **last)
{
    while (*s++ == *with++)
    {
        if (*s == 0 || *with == 0)
            break;
    }
    if (*with == 0)
        *last = s + 1;
    return *with == 0;
}

static void help()
{
    printf(
        "Usage: \n"
        "    ./centerpoint_infer ../data/test/\n"
        "    Run centerpoint(voxelnet) inference with data under ../data/test/\n"
        "    Optional: --verbose, enable verbose log level\n"
    );
    exit(EXIT_SUCCESS);
}




int main(int argc, const char **argv)
{
    if (argc < 2)
        help();

    const char *value = nullptr;
    bool verbose = false;
    bool visualize = false;
    bool cpu=false;
    for (int i = 2; i < argc; ++i) {
        if (startswith(argv[i], "--verbose", &value)) {
            verbose = true;
        } else if(startswith(argv[i], "--visualize", &value)){
            visualize = true;
        }
        else if(startswith(argv[i], "--cpu", &value)){
            cpu = true;
        }
        else{
            help();
        }
    }

    const char *data_folder  = argv[1];

    GetDeviceInfo();

    std::vector<std::string> files;
    getFolderFile(data_folder, files);

    std::cout << "Total " << files.size() << std::endl;

    Params params;
    cudaStream_t stream = NULL;
    checkCudaErrors(cudaStreamCreate(&stream));

    std::shared_ptr<PreProcessCuda> pre_;
    pre_.reset(new PreProcessCuda());
    pre_->alloc_resource();

    half* d_voxel_features;
    unsigned int* d_voxel_indices;
    std::vector<int> sparse_shape;

    float *d_points = nullptr;    
    checkCudaErrors(cudaMalloc((void **)&d_points, MAX_POINTS_NUM * params.feature_num * sizeof(float)));
    for (const auto & file : files)
    {
        std::string dataFile = data_folder + file + ".bin";

        std::cout << "\n<<<<<<<<<<<" <<std::endl;
        std::cout << "load file: "<< dataFile <<std::endl;

        unsigned int length = 0;
        void *pc_data = NULL;

        loadData(dataFile.c_str() , &pc_data, &length);
        size_t points_num = length / (params.feature_num * sizeof(float)) ;
        std::cout << "find points num: " << points_num << std::endl;
        
        float* point_data = static_cast<float*>(pc_data);
        if (cpu) {
            // Create the VoxelizerCPU object
            VoxelizerCPU voxelizer;

            // Measure CPU voxelization time
            auto start_cpu_voxel = std::chrono::high_resolution_clock::now();
            std::vector<Voxel> voxels = voxelizer.voxelization(point_data, points_num);
            std::vector<std::array<float, 4>> voxel_features = voxelizer.calculateVoxelFeatures(voxels);
            auto end_cpu_voxel = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> cpu_voxel_time = end_cpu_voxel - start_cpu_voxel;
            std::cout << "Voxelization (CPU) : " << cpu_voxel_time.count() << " ms" << std::endl;
            std::cout << "Voxel Count (CPU): " << voxels.size() << std::endl;
        }
        if (visualize){

            Visualizer point_cloud_visualizer("Point Cloud Viewer", points_num, point_data, params.feature_num);
            point_cloud_visualizer.initialize();
            point_cloud_visualizer.populate_cloud();
            point_cloud_visualizer.show_cloud();
        }

        
        checkCudaErrors(cudaMemcpy(d_points, pc_data, length, cudaMemcpyHostToDevice));
        auto start_gpu_voxel = std::chrono::high_resolution_clock::now();
        pre_->generateVoxels((float *)d_points, points_num, stream);
        unsigned int valid_num = pre_->getOutput(&d_voxel_features, &d_voxel_indices, sparse_shape);
        half* h_voxel_features = new half[valid_num * params.feature_num];
        unsigned int* h_voxel_indices = new unsigned int[valid_num * 4];
        checkCudaErrors(cudaMemcpy(h_voxel_features, d_voxel_features, valid_num * params.feature_num * sizeof(half), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_voxel_indices, d_voxel_indices, valid_num * 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        
        auto end_gpu_voxel = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> gpu_voxel_time = end_gpu_voxel - start_gpu_voxel;
        std::cout << "Voxelization (GPU) : " << gpu_voxel_time.count() << " ms" << std::endl;    
       
        if (visualize){
            float* h_float_array = new float[valid_num * params.feature_num];
            // Convert the half array to float array element by element
            for (size_t i = 0; i < valid_num; ++i) {
                h_float_array[i] = __half2float(h_voxel_features[i]);
            }

            Visualizer voxel_visualizer("Voxel Cloud Viewer", valid_num, h_float_array, params.feature_num);
            voxel_visualizer.initialize();
            voxel_visualizer.populate_cloud();
            voxel_visualizer.show_cloud();
        }



        std::cout << "voxel count : " << valid_num << std::endl;
        std::cout << ">>>>>>>>>>>" <<std::endl;
    }

    // centerpoint.perf_report();
    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaStreamDestroy(stream));
    return 0;
}
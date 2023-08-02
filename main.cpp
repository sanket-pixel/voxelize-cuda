#include <sstream>
#include <fstream>
#include <stdio.h>
#include <fstream>
#include <memory>
#include <chrono>
#include <dirent.h>
#include "common.h"
#include "preprocess.h"
#include <GL/glut.h> 
#include <vector>

std::vector<std::tuple<float, float, float>> input_points;
float rotation_angle = 0.0f;
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

// Function to draw points in OpenGL
void drawPoints(const std::vector<std::tuple<float, float, float>>& points, float r, float g, float b) {
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    glColor3f(r, g, b);
    for (const auto& p : points) {
        glVertex3f(std::get<0>(p), std::get<1>(p), std::get<2>(p));
    }
    glEnd();
}

// Function to display the points in OpenGL
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Apply rotation to the scene
    glRotatef(rotation_angle, 0.0f, 1.0f, 0.0f);

    // Draw the original points in blue
    drawPoints(input_points, 0.0f, 0.0f, 1.0f);

    // Draw the voxelized points in red
    // drawPoints(voxel_points, 1.0f, 0.0f, 0.0f);

    glutSwapBuffers();
}
void specialKeys(int key, int x, int y) {
    switch (key) {
        case GLUT_KEY_LEFT:
            rotation_angle -= 5.0f;
            break;
        case GLUT_KEY_RIGHT:
            rotation_angle += 5.0f;
            break;
    }
    glutPostRedisplay(); // Request a redraw of the scene
}

int main(int argc, const char **argv)
{
    if (argc < 2)
        help();

    const char *value = nullptr;
    bool verbose = false;
    for (int i = 2; i < argc; ++i) {
        if (startswith(argv[i], "--verbose", &value)) {
            verbose = true;
        } else {
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

        // Convert pc_data to vector of 3D points
        float* points_data = static_cast<float*>(pc_data);
        input_points.clear();
        for (size_t i = 0; i < points_num; ++i) {
            float x = points_data[i * params.feature_num];
            float y = points_data[i * params.feature_num + 1];
            float z = points_data[i * params.feature_num + 2];
            input_points.emplace_back(x, y, z);
        }

        // Free the allocated memory
        delete[] points_data;
         // Initialize OpenGL window
        glutInit(&argc, const_cast<char **>(argv));
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(800, 600);
        glutCreateWindow("Input Points Visualization");

        // Set up the OpenGL callbacks
        glutDisplayFunc(display);
        glutSpecialFunc(specialKeys);
        glEnable(GL_DEPTH_TEST);

        // Enter the main loop
        glutMainLoop();

        checkCudaErrors(cudaMemcpy(d_points, pc_data, length, cudaMemcpyHostToDevice));

        pre_->generateVoxels((float *)d_points, points_num, stream);

        unsigned int valid_num = pre_->getOutput(&d_voxel_features, &d_voxel_indices, sparse_shape);
        half* h_voxel_features = new half[valid_num * params.feature_num];
        unsigned int* h_voxel_indices = new unsigned int[valid_num * 4];
        checkCudaErrors(cudaMemcpy(h_voxel_features, d_voxel_features, valid_num * params.feature_num * sizeof(half), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_voxel_indices, d_voxel_indices, valid_num * 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        std::vector<std::tuple<float, float, float>> voxel_points;
        for (int i = 0; i < valid_num; ++i) {
            int offset = i * params.feature_num; // Assuming each voxel has 3D coordinates (x, y, z) as features
            float x = static_cast<float>(h_voxel_features[offset]);
            float y = static_cast<float>(h_voxel_features[offset + 1]);
            float z = static_cast<float>(h_voxel_features[offset + 2]);
            voxel_points.emplace_back(x, y, z);
        }

        std::cout << "voxel count : " << valid_num << std::endl;
        std::cout << ">>>>>>>>>>>" <<std::endl;
    }

    // centerpoint.perf_report();
    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaStreamDestroy(stream));
    return 0;
}
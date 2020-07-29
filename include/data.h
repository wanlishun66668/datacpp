#ifndef DATA_H
#define DATA_H
#include <pthread.h>
#include<opencv.hpp>
#include<opencv/cv.hpp>
#include <vector>
#include<tinyxml2.h>
//#ifdef _DEBUG
//#pragma comment(lib,"opencv_world3410d.lib")
//#else
//#pragma commmet(lib,"opencv_world3410.lib")
//#endif
//#pragma comment(lib,"pthreadVC2.lib")


typedef struct {
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    int label;
} box;

typedef struct {
    int w, h;
    std::vector<cv::Mat> X;
    std::vector<cv::Mat> Y;
    int shallow;
    int num_boxes;
    std::vector<std::vector<box> > boxes;
} data;
typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA
} data_type;

typedef struct load_args {
    std::string image_type;
    int threads;
    int out_flag;
    int thread_index;
    int batch_size;
    int batch_count;
    std::vector<std::string> image_name_list;
    std::string image_dir;
    int n;
    int m;
    std::string label_type;
    std::string label_postfix;
    std::string label_dir;
    int h;
    int w;
    int out_w;
    int out_h;
    int classes;
	data *d;
    float crop_label_threshold;
    bool resize_label;
    bool resized;
    bool horizonta_flip;
    bool vertical_flip;
    bool rand_flip;
    bool rand_rotate_90;
    bool rand_brightness;
    bool rand_blur;
    bool gaussian_blur;
    bool rand_rotate_angle;
    bool rand_crop;
    bool center_crop;
    bool rand_scale;
} load_args;
data concat_datas(data* d, int n);
void *load_threads(void *ptr);
pthread_t load_data(load_args args);
void* load_thread(void* ptr);
data load_data_old(int index, int n, load_args arg, char** labels);
void get_next_batch(data d, int n, int offset, std::vector<cv::Mat> &input, float* y);
pthread_t load_data_in_thread(load_args args);
void read_file_name(std::vector<std::string>& paths, const std::string& file);
void get_random_paths(std::vector<std::string>& image_names, int n);
#endif
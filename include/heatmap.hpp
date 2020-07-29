#ifndef HEATMAP_HPP
#define HEATMAP_HPP
#include"data.h"
std::vector<cv::Mat> create_heatmap(const cv::Mat& image, std::vector<box> boxes, int downscale, float sigma);

#endif
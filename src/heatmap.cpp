#include"heatmap.hpp"


std::vector<cv::Mat> create_heatmap(const cv::Mat &image, std::vector<box> boxes, int downscale,float sigma) {
	int height = image.rows / downscale;
	int width = image.cols / downscale;
	std::vector<cv::Mat > results(5);
	cv::Mat heatmap = cv::Mat::zeros(height, width, CV_32F), temp_heatmap;
	cv::Mat center_x_grid = cv::Mat::zeros(height, width, CV_32F);
	cv::Mat center_y_grid = cv::Mat::zeros(height, width, CV_32F);
	cv::Mat width_grid = cv::Mat::zeros(height, width, CV_32F);
	cv::Mat height_grid = cv::Mat::zeros(height, width, CV_32F);
	cv::Mat grid_x = cv::Mat::zeros(height, width, CV_32F), grid_y = cv::Mat::zeros(height, width, CV_32F);
	cv::Mat temp_x_grid, temp_y_grid;
	cv::Mat  grid_dist;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			grid_x.at<float>(i, j) = j;
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			grid_y.at<float>(i, j) = i;
		}
	}
	box temp_box;
	float xmin, ymin, xmax, ymax,center_x,center_y;
	float original_center_x, original_center_y;
	for (int i = 0; i < boxes.size(); i++) {
		temp_box = boxes[i];
		xmin = temp_box.xmin / downscale;
		ymin = temp_box.ymin / downscale;
		xmax = temp_box.xmax / downscale;
		ymax = temp_box.ymax / downscale;
		original_center_x = (temp_box.xmax + temp_box.xmin)/2.0;
		original_center_y= (temp_box.ymax + temp_box.ymin)/2.0;
		center_x = (xmin + xmax) / 2;
		center_y = (ymin + ymax) / 2;
		temp_x_grid=grid_x - center_x;
		temp_y_grid =grid_y - center_y;
		cv::multiply(temp_x_grid, temp_x_grid, temp_x_grid);
		cv::multiply(temp_y_grid, temp_y_grid, temp_y_grid);
		grid_dist = temp_x_grid + temp_y_grid;
		
		cv::exp(grid_dist / (-2 * sigma * sigma), temp_heatmap);
		heatmap=cv::max(heatmap, temp_heatmap);
		
		center_x_grid.at<float>(center_y,center_x)= original_center_x / downscale - center_x;
		center_y_grid.at<float>(center_y, center_x)=original_center_y / downscale - center_y;
		width = xmax - xmin;
		width_grid.at<float>(center_y, center_x) = cv::log(width + 1e-4);
		height = ymax - ymin;
		height_grid.at<float>(center_y, center_x) = cv::log(height + 1e-4);
	}
	results[0] = heatmap;
 	results[1] = center_x_grid;
	results[2] = center_y_grid;
	results[3] = width_grid;
	results[4] = height_grid;
	return results;
}

//cv::Mat tt = cv::Mat::ones(image.rows / downscale, image.rows / downscale, CV_32F) * 255;
//tt = tt.mul(heatmap);
//cv::Mat mm;
//tt.convertTo(mm, CV_8UC1);
//cv::applyColorMap(mm, mm, 2);
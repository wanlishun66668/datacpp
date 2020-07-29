#include<iostream>
#include"loader.h"
#include"heatmap.hpp"
#include<pybind11/pybind11.h>
#include<time.h>
int main(int argc,char**argv) {
	char file_name[200]= "D:/datasets/VOC/VOCdevkit/VOC2012/ImageSets/Main/train.txt";
	const char* image_dir = "D:/datasets/VOC/VOCdevkit/VOC2012/JPEGImages/";
	const char* label_dir = "D:/datasets/VOC/VOCdevkit/VOC2012/Annotations/";
	const char* label_type = "detection";
	std::string label_postfix = "xml";
	std::string image_type = ".jpg";
	load_args args;
	clock_t start, end;
	start = clock();
	args=parameter_set(image_dir,file_name, 512, 512, 4, label_dir, label_type,label_postfix,image_type,4);
	end = clock();
	std::cout << "para time:" << (double)(end - start) / CLOCKS_PER_SEC<< std::endl;
	for (int i = 0; i < 20; i++) {
		start = clock();
		data temp_data=get_batch(args);
		end = clock();
		std::cout << "get_batch time:" << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
		cv::Mat image = temp_data.X[0];
		std::vector<box> boxes=temp_data.boxes[0];
		for (int i = 0; i < boxes.size(); i++) {
			int x = boxes[i].xmin;
			int y = boxes[i].ymin;
			int x2 = boxes[i].xmax;
			int y2 = boxes[i].ymax;
			cv::rectangle(image, cv::Rect(x, y, x2 - x, y2 - y), cv::Scalar(255, 0, 0));
		}
		if (temp_data.Y.size() != 0) {
			//cv::imshow("src", temp_data.Y[0]);
		}
		
		//cv::imshow("dst", image);
		//cv::waitKey(0);
		start = clock();
		std::vector<cv::Mat> results = create_heatmap(image, boxes, 4,2.65);
		end = clock();
		std::cout << "create_heatmap time:" << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
		std::cout << "i:" << i << std::endl;
	}
	
	cv::Mat image(5, 5, CV_32FC1);

	getchar();
	return 0;
}

#include<pybind11/numpy.h>
#include<pybind11/stl.h>

namespace py=pybind11;

cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char>& input) {
	if (input.ndim() != 3) {
		std::cout << "dim is not 3!" << std::endl;
	}
	py::buffer_info buf = input.request();
	cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
	return mat;

}
py::array_t<float> cv_mat_uint8_1c_to_numpy(cv::Mat& input) {
	//py::array_t<unsigned char> dst = py::array_t<unsigned char>({ input.rows,input.cols,input.channels()}, input.data);
	py::array_t<float> dst = py::array_t<float>({ input.rows,input.cols,input.channels() },(float*) input.data);
	return dst;

}
py::array_t<float> cv_mat_float_1c_to_numpy(cv::Mat& input) {
	py::array_t<float> dst = py::array_t<float>({ input.rows,input.cols},(float*)input.data);
	return dst;

}
void get_image_batch(std::vector<py::array_t<float> > &results,std::vector<cv::Mat> images) {
	py::array_t<float> temp_array;
	for (int i = 0; i < images.size(); i++) {
		temp_array = cv_mat_uint8_1c_to_numpy(images[i]);
		results.push_back(temp_array);
	}
}
void get_box_label_batch(std::vector<std::vector<std::vector<int> > > &results,std::vector<std::vector<box> > &labels) {
	std::vector<int> temp_label;
	std::vector<std::vector<int> >image_label;
	for (int i = 0; i < labels.size(); i++) {
		for (int j = 0; j < labels[i].size(); j++) {
			temp_label.push_back(labels[i][j].xmin);
			temp_label.push_back(labels[i][j].ymin);
			temp_label.push_back(labels[i][j].xmax);
			temp_label.push_back(labels[i][j].ymax);
			image_label.push_back(temp_label);
			temp_label.clear();
		}
		results.push_back(image_label);
		image_label.clear();
	}
}
void get_heatmap_label_batch(std::vector<std::vector<py::array_t<float> > >& results, cv::Mat& image, std::vector<std::vector<box> >& labels) {
	std::vector<int> temp_label;
	std::vector<cv::Mat >image_label;
	std::vector<py::array_t<float> > image_array;
	int downscale = 4;
	float sigma = 2.16;
	py::array_t<float> dst;
	for (int i = 0; i < labels.size(); i++) {
		image_label = create_heatmap(image, labels[i],4,sigma);
		for (int j = 0; j < image_label.size(); j++) {
			dst=cv_mat_float_1c_to_numpy(image_label[j]);
			image_array.push_back(dst);
		}
		results.push_back(image_array);
		image_array.clear();
	}
}
struct batch {
	std::vector<py::array_t<float> > images;
	//std::vector<std::vector<std::vector<int> > > labels;
	std::vector<std::vector<py::array_t<float> > > labels;
};
batch next_batch(load_args& args) {
	data  buffer;
	args.d = &buffer;
	batch temp_batch;
	if ((args.batch_count + 1) * args.batch_size > args.m) {
		args.out_flag = 1;
		return temp_batch;
	}
	pthread_t load_thread = load_data(args);
	pthread_join(load_thread, 0);
	get_image_batch(temp_batch.images,buffer.X);
	//get_label_batch(temp_batch.labels,buffer.boxes);
	get_heatmap_label_batch(temp_batch.labels, buffer.X[0], buffer.boxes);
	args.batch_count++;
	return temp_batch;
}
PYBIND11_MODULE(data, m) {
	m.def("next_batch", &next_batch, "This is function add");
	m.def("paramset", &parameter_set, "This is a function paraset");
	py::class_<load_args>(m, "load_args")
		.def_readwrite("out_flag", &load_args::out_flag)
		.def_readwrite("batch_count", &load_args::batch_count);
	py::class_<batch>(m, "batch")
		.def_readwrite("images", &batch::images)
		.def_readwrite("labels", &batch::labels);
	py::class_<transform_image>(m, "transform_image")
		.def(py::init())
		.def("horizonta_flip", &transform_image::horizonta_flip)
		.def("vertical_flip", &transform_image::vertical_flip)
		.def("rand_flip", &transform_image::rand_flip)
		.def("rand_rotate_90", &transform_image::rand_rotate_90)
		.def("rand_brightness", &transform_image::rand_brightness)
		.def("rand_blur", &transform_image::rand_blur)
		.def("gaussian_blur", &transform_image::gaussian_blur)
		.def("rand_rotate_angle", &transform_image::rand_rotate_angle)
		.def("center_crop", &transform_image::center_crop)
		.def("rand_crop", &transform_image::rand_crop)
		.def("rand_scale", &transform_image::rand_scale)
		.def("resized", &transform_image::resized);
}

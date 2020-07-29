#include<stdlib.h>
#include<time.h>
#include"data.h"
#include<fstream>
#include<algorithm>
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_t load_data(load_args args)
{
    pthread_t thread;
	load_args* ptr = new load_args(args);
    pthread_create(&thread, 0, load_threads, (void*)ptr);
    return thread;
}

void read_file_name(std::vector<std::string>& paths, const std::string& file) {
	std::ifstream in(file);
	std::string line;
	if (in) {
		while (getline(in, line)) {
			paths.push_back(line);
		}
	}
	else {
		std::cout << "no such file" << std::endl;
	}
}


void *load_threads(void *ptr)
{
    int i;
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    data *out = args.d;
    int total = args.batch_size;

    data *buffers = (data*)calloc(args.threads, sizeof(data));

    pthread_t *threads = (pthread_t*)calloc(args.threads, sizeof(pthread_t));
	int index=0;
    for(i = 0; i < args.threads; ++i){
        args.d = buffers + i;
        args.n = (i+1) * total/args.threads - i * total/args.threads;
		args.thread_index = index;
        threads[i] = load_data_in_thread(args);
		index += args.n;
    }
    for(i = 0; i < args.threads; ++i){
        pthread_join(threads[i], 0);
    }
    *out = concat_datas(buffers, args.threads);
    for(i = 0; i < args.threads; ++i){
        buffers[i].shallow = 1;
        //free_data(buffers[i]);
    }
    free(buffers);
	free(threads);
    return 0;
}
void get_next_batch(data d, int n, int offset, std::vector<cv::Mat> &input, float* y)
{
    int j;
    for (j = 0; j < n; ++j) {
        int index = offset + j;
        /*memcpy(X + j * d.X.cols, d.X.vals[index], d.X.cols * sizeof(float));
        if (y) memcpy(y + j * d.y.cols, d.y.vals[index], d.y.cols * sizeof(float));*/
    }
}
pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    //struct load_args *ptr = (struct load_args *)calloc(1, sizeof(struct load_args));
	struct load_args* ptr = new load_args();
    *ptr = args;
    pthread_create(&thread, 0, load_thread, ptr);
    return thread;
}

void* load_thread(void* ptr)
{
    load_args a = *(struct load_args*)ptr;
	int index=a.batch_size* a.batch_count + a.thread_index + a.n;
	if (index > a.m) {
		printf("index:%d Խ�磡", index);
		exit(0);
	}
	clock_t start, end;
	start = clock();
    *a.d = load_data_old(index-a.n, a.n,a,NULL);
	end = clock();
	//std::cout << "thread time:" << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
	delete ptr;
    return 0;
}

void get_random_paths(std::vector<std::string> &image_names, int n)
{
	int i;
	std::string temp;
	srand((unsigned)time(NULL));
	for (i = n-1; i >0; --i) {
		
		int index = rand() % i;
		temp = image_names[index];
		image_names[index] = image_names[i];
		image_names[i] = temp;
	}
}

void cv_copy_mark_boarder(cv::Mat &image,load_args &arg, std::vector<box>& labels) {
	int left = 0, right = 0, top = 0, bottom = 0;
	if (arg.w < arg.out_w) {
		left = (arg.out_w - arg.w) / 2;
		if ((arg.out_w - arg.w) % 2) {
			right = left + 1;
		}
		else {
			right = left;
		}
		arg.w = arg.out_w;
		for (int i = 0; i < labels.size(); i++) {
			labels[i].xmin += left;
			labels[i].xmax += left;
		}
	}
	if (arg.h < arg.out_h) {
		top = (arg.out_h - arg.h) / 2;
		if ((arg.out_h - arg.h) % 2) {
			bottom = top + 1;
		}
		else {
			bottom = top;
		}
		arg.h = arg.out_h;
		for (int i = 0; i < labels.size(); i++) {
			labels[i].ymin += top;
			labels[i].ymax += top;
		}
	}
	cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

void crop_labels(std::vector<box>& labels,int x1,int y1,load_args &arg) {
	box temp_box, crop_box;
	std::vector<box> results;
	int xmin, ymin, xmax, ymax,label;
	float iou_value, src_area;
	for (int i = 0; i < labels.size(); i++) {
		temp_box = labels[i];
		xmin = std::max(temp_box.xmin - x1, 0);
		ymin = std::max(temp_box.ymin - y1, 0);
		xmax = std::min(temp_box.xmax - x1, arg.w);
		ymax = std::min(temp_box.ymax - y1, arg.h);
		label = temp_box.label;
		src_area = (temp_box.xmax - temp_box.xmin) * (temp_box.ymax - temp_box.ymin);
		iou_value = (xmax - xmin) * (ymax - ymin) * 1.0 / src_area;
		if (iou_value >= arg.crop_label_threshold) {
			crop_box.xmin = xmin;
			crop_box.ymin = ymin;
			crop_box.xmax = xmax;
			crop_box.ymax = ymax;
			crop_box.label = label;
			results.push_back(crop_box);
		}
	}
	labels = results;
}

void center_crop(cv::Mat& image, load_args& arg, std::vector<box> &labels) {
	cv_copy_mark_boarder(image, arg, labels);
	int x1 = (arg.w - arg.out_w) / 2;
	int y1 = (arg.h - arg.out_h) / 2;
	image = image(cv::Rect(x1, y1, arg.out_w, arg.out_h));
	arg.w = arg.out_w;
	arg.h = arg.out_h;
	if (arg.label_type=="detection") {
		crop_labels(labels, x1, y1, arg);
	}
}

void rand_crop(cv::Mat& image, load_args& arg, std::vector<box>& labels) {
	//cv_copy_mark_boarder(image, arg);
	srand((unsigned int)(time(NULL)));
	double crop_code = rand() % 6 / 10.0 + 0.5;
	int width = image.cols;
	int height = image.rows;
	int dst_width = int(width * crop_code);
	int dst_height = int(height * crop_code);
	if (width != dst_width && height != dst_height) {
		int x1 = rand() % (width - dst_width);
		int y1 = rand() % (height - dst_height);
		image = image(cv::Rect(x1, y1, dst_width, dst_height));
		arg.w = dst_width;
		arg.h = dst_height;
		if (arg.label_type == "detection") {
			crop_labels(labels, x1, y1, arg);
		}
	}
}

void rand_scale(cv::Mat& image, load_args& arg, std::vector<box>& labels) {
	srand((unsigned int)(time(NULL)));
	double scale_code = rand() % 10 / 10.0 + 0.5;
	double width = image.cols;
	double height = image.rows;
	double resize_width = image.cols * scale_code;
	double resize_height = image.rows * scale_code;
	cv::resize(image, image, cv::Size(0, 0), scale_code, scale_code);
	if (arg.label_type == "detection") {
		for (int i = 0; i < labels.size(); i++) {
			labels[i].xmin = int(labels[i].xmin * 1.0 / width * resize_width);
			labels[i].ymin = int(labels[i].ymin * 1.0 / height * resize_height);
			labels[i].xmax = int(labels[i].xmax * 1.0 / width * resize_width);
			labels[i].ymax = int(labels[i].ymax * 1.0 / height * resize_height);
		}
	}
	
}

void resize_to_out(cv::Mat& image, load_args& arg, std::vector<box>& labels) {
	double width = image.cols;
	double height = image.rows;
	cv::resize(image, image, cv::Size(arg.out_w,arg.out_h));
	double resize_width = arg.out_w;
	double resize_height = arg.out_h;
	arg.w = arg.out_w;
	arg.h = arg.out_h;
	if (arg.label_type == "detection") {
		for (int i = 0; i < labels.size(); i++) {
			labels[i].xmin = int(labels[i].xmin * 1.0 / width * resize_width);
			labels[i].ymin = int(labels[i].ymin * 1.0 / height * resize_height);
			labels[i].xmax = int(labels[i].xmax * 1.0 / width * resize_width);
			labels[i].ymax = int(labels[i].ymax * 1.0 / height * resize_height);
		}
	}

}

void horizonta_flip(cv::Mat& image,load_args &arg,std::vector<box>& labels ) {
	cv::flip(image, image, 1);
	if (arg.label_type == "detection") {
		for (int i = 0; i < labels.size(); i++) {
			int temp_xmin = labels[i].xmin;
			labels[i].xmin = arg.w - labels[i].xmax;
			labels[i].xmax = arg.w - temp_xmin;
		}
	}
}

void vertical_flip(cv::Mat& image, load_args& arg, std::vector<box>& labels) {
	cv::flip(image, image, 0);
	if (arg.label_type == "detection") {
		for (int i = 0; i < labels.size(); i++) {
			int temp_ymin = labels[i].ymin;
			labels[i].ymin = arg.h - labels[i].ymax;
			labels[i].ymax = arg.h - temp_ymin;
		}
	}
}

void rand_flip(cv::Mat& image, load_args& arg, std::vector<box>& labels) {
	srand((unsigned int)(time(NULL)));
	int flip_code = rand() % 4;
	if (flip_code == 0) {
		vertical_flip(image, arg, labels);
	}else if (flip_code == 1) {
		horizonta_flip(image, arg, labels);
	}
	else if (flip_code == 2) {
		vertical_flip(image, arg, labels);
		horizonta_flip(image, arg, labels);
	}
}

void rand_rotate_90(cv::Mat& image, load_args& arg, std::vector<box>& labels) {
	srand((unsigned int)(time(NULL)));
	int rotate_code = rand() % 4;
	cv::rotate(image, image, rotate_code);
	if (arg.label_type == "detection") {
		if (rotate_code == 0) {
			for (int i = 0; i < labels.size(); i++) {
				labels[i].xmin;
				int temp_xmin = labels[i].xmin;
				int temp_xmax= labels[i].xmax;
				int temp_ymin=labels[i].ymin;
				labels[i].xmin = arg.h - labels[i].ymax;
				labels[i].ymin = temp_xmin;
				labels[i].xmax= arg.h - temp_ymin;
				labels[i].ymax = temp_xmax;
			}
		}
		else if (rotate_code == 1) {
			for (int i = 0; i < labels.size(); i++) {
				labels[i].xmin;
				int temp_xmin = labels[i].xmin;
				int temp_ymin = labels[i].ymin;
				labels[i].xmin = arg.w - labels[i].xmax;
				labels[i].ymin = arg.h - labels[i].ymax;
				labels[i].xmax = arg.w - temp_xmin;
				labels[i].ymax = arg.h - temp_ymin;
			}
		}
		else if (rotate_code == 2) {
			for (int i = 0; i < labels.size(); i++) {
				int temp_xmin = labels[i].xmin;
				labels[i].xmin = labels[i].ymin;
				labels[i].ymin = arg.w - labels[i].xmax;
				labels[i].xmax = labels[i].ymax;
				labels[i].ymax = arg.w - temp_xmin;
				
			}
		}
	}
}

void rand_rotate_angle(cv::Mat& image) {
	srand((unsigned int)(time(NULL)));
	if (rand() % 2) {
		int rotate_rand = rand() % 41 + 5;
		if (rand() % 2) {
			rotate_rand = -rotate_rand;
		}
		cv::Mat mat_rotate = cv::getRotationMatrix2D(cv::Point2d((float)(image.cols / 2), (float)(image.rows / 2)), rotate_rand, 1);
		cv::warpAffine(image, image, mat_rotate, cv::Size(image.cols, image.rows));
	}
}

void rand_brightness(cv::Mat& image) {
	srand((unsigned int)(time(NULL)));
	int brightness_value = rand() % 5*20;
	if (rand() % 2) {
		brightness_value = -brightness_value;
	}
	image.convertTo(image, -1, 1,brightness_value);
}

void rand_blur(cv::Mat& image) {
	srand((unsigned int)(time(NULL)));
	int blur_code = rand() % 7 + 1;
	if (blur_code < 6) {
		cv::blur(image, image, cv::Size(blur_code, blur_code));
	}
}

void gaussian_blur(cv::Mat& image) {
	srand((unsigned int)(time(NULL)));
	int gaussian_code = rand() % 2;
	if (gaussian_code) {
		cv::GaussianBlur(image, image, cv::Size(3, 3), 0, 0);
	}
}
void resize_out(cv::Mat& image,load_args &arg) {
	cv::resize(image, image, cv::Size(arg.out_w, arg.out_h));
}
data load_data_old(int index,int n, load_args arg, char **labels)
{
	data d = { 0 };
	d.shallow = 0;
	
	for (int i = 0; i < n; ++i) {
		clock_t start, end;
		std::vector<box> temp_box_vec;
		std::string image_name=arg.image_name_list[index + i];
		std::string temp_path(arg.image_dir+ image_name+arg.image_type);
		//std::cout << "image name:" << temp_path << std::endl;
		start = clock();
		cv::Mat image = cv::imread(temp_path);
		//d.Y.push_back(image.clone());
		end = clock();
		//std::cout << temp_path << "image_size: rows," << image.rows << ".cols:"<<image.cols << std::endl;
		//std::cout << "image read time:" << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
		start = clock();
		int flag = 0;
		if (std::string(arg.label_postfix) == std::string("xml")) {
			//std::string label_name = std::string(arg.label_dir) + image_no_post + ".xml";
			std::string label_name = std::string(arg.label_dir) + image_name + ".xml";
			tinyxml2::XMLDocument doc;
			doc.LoadFile(label_name.c_str());
			tinyxml2::XMLElement* anno = doc.RootElement();
			tinyxml2::XMLElement* object = anno->FirstChildElement("object");
			flag = 0;
			int count = 0;
			box temp_box;
			while (object) {
				tinyxml2::XMLElement* bndbox = object->FirstChildElement("bndbox");
				if (bndbox) {
					tinyxml2::XMLElement* xmin_t = bndbox->FirstChildElement("xmin");
					tinyxml2::XMLElement* ymin_t = bndbox->FirstChildElement("ymin");
					tinyxml2::XMLElement* xmax_t = bndbox->FirstChildElement("xmax");
					tinyxml2::XMLElement* ymax_t = bndbox->FirstChildElement("ymax");
					temp_box.xmin = atoi(xmin_t->GetText());
					temp_box.ymin = atoi(ymin_t->GetText());
					temp_box.xmax = atoi(xmax_t->GetText());
					temp_box.ymax = atoi(ymax_t->GetText());
					temp_box.label = 1;
					flag = 1;
					count++;
				}
				object = object->NextSiblingElement();
				if (flag) {
					temp_box_vec.push_back(temp_box);
				}
			}
			//std::cout << "image name:" << temp_path << ",cout: "<<count<<std::endl;
		}
		/*end = clock();
		std::cout << "image xml time:" << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
		start = clock();*/
		arg.h = image.rows;
		arg.w = image.cols;
		//resize_to_out(image, arg, temp_box_vec);
		if (arg.rand_scale) {
			rand_scale(image, arg, temp_box_vec);
		}
		if (arg.center_crop) {
			center_crop(image, arg, temp_box_vec);
		}
		if (arg.rand_crop) {
			rand_crop(image, arg, temp_box_vec);
		}
		if (arg.horizonta_flip) {
			horizonta_flip(image, arg, temp_box_vec);
		}
		if (arg.vertical_flip) {
			vertical_flip(image, arg, temp_box_vec);
		}
		if (arg.rand_flip) {
			rand_flip(image, arg, temp_box_vec);
		}
		if (arg.rand_rotate_90) {
			rand_rotate_90(image, arg, temp_box_vec);
		}
		if (arg.rand_rotate_angle) {
			
		}
		if (arg.rand_brightness) {
			rand_brightness(image);
		}
		if (arg.rand_blur) {
			rand_blur(image);
		}
		if (arg.gaussian_blur) {
			gaussian_blur(image);
		}
		if (arg.resized) {
			resize_to_out(image, arg, temp_box_vec);
		}
		image.convertTo(image, CV_32F);
		d.X.push_back(image);
		d.boxes.push_back(temp_box_vec);
		/*end = clock();
		std::cout << "image resize time:" << (double)(end - start) / CLOCKS_PER_SEC << std::endl;*/
	}
	return d;
}
data concat_matrix(data image_vec1, data image_vec2) {
	data temp_data;
	std::vector<cv::Mat> merge_vec;
	merge_vec.insert(merge_vec.end(), image_vec1.X.begin(), image_vec1.X.end());
	merge_vec.insert(merge_vec.end(), image_vec2.X.begin(), image_vec2.X.end());

	std::vector<cv::Mat> merge_vec_y;
	merge_vec_y.insert(merge_vec_y.end(), image_vec1.Y.begin(), image_vec1.Y.end());
	merge_vec_y.insert(merge_vec_y.end(), image_vec2.Y.begin(), image_vec2.Y.end());

	std::vector<std::vector<box> > merge_boxes;
	merge_boxes.insert(merge_boxes.end(), image_vec1.boxes.begin(), image_vec1.boxes.end());
	merge_boxes.insert(merge_boxes.end(), image_vec2.boxes.begin(), image_vec2.boxes.end());
	temp_data.X = merge_vec;
	temp_data.boxes = merge_boxes;
	temp_data.Y = merge_vec_y;
	return temp_data;
}
data concat_data(data d1, data d2)
{
    data d = { 0 };
    d.shallow = 1;
    /*d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);*/
	d = concat_matrix(d1, d2);
    return d;
}
data concat_datas(data* d, int n)
{
    int i;
    data out = { 0 };
    for (i = 0; i < n; ++i) {
        data new_t = concat_data(d[i], out);
        //free_data(out);
        out = new_t;
    }
    return out;
}

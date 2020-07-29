#ifndef LOADER_H
#define LOADER_H

#include"data.h"
class transform_image {
public:
	transform_image() {

	}
	void horizonta_flip(load_args &args) {
		args.horizonta_flip = true;
	}
	void vertical_flip(load_args &args) {
		args.vertical_flip = true;
	}
	void rand_flip(load_args &args) {
		args.rand_flip = true;
	}
	void rand_rotate_90(load_args &args) {
		args.rand_rotate_90 = true;
	}
	void rand_brightness(load_args &args) {
		args.rand_brightness = true;
	}
	void rand_blur(load_args &args) {
		args.rand_blur = true;
	}
	void gaussian_blur(load_args &args) {
		args.gaussian_blur = true;
	}
	void rand_rotate_angle(load_args &args) {
		args.rand_rotate_angle = true;
	}
	void center_crop(load_args &args) {
		args.center_crop = true;
	}
	void rand_crop(load_args &args) {
		args.rand_crop = true;
	}
	void rand_scale(load_args &args) {
		args.rand_scale = true;
	}
	void resized(load_args &args) {
		args.resized = true;
	}
};
load_args parameter_set(const std::string &image_dir,char* file_name, int out_h, int out_w, int batch_size,
	const char* label_dir, const char* label_type, const std::string label_postfix, const std::string image_type=".jpg", int threads = 4, bool shuffle = true);
data get_batch(load_args& args);
#endif
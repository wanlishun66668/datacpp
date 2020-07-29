#include"loader.h"
#include<cstdlib>

void transfor_image(load_args& args) {
	args.crop_label_threshold=0.3;
	args.horizonta_flip=false;
	args.vertical_flip = false;
	args.rand_flip = false;
	args.rand_rotate_90 = false;
	args.rand_brightness = false;
	args.rand_blur = false;
	args.gaussian_blur = false;
	args.rand_rotate_angle = false;
	args.center_crop = false;
	args.rand_crop = false;
	args.rand_scale = false;
	args.resized = true;
}


load_args parameter_set(const std::string &image_dir, char* file_name, int out_h, int out_w, int batch_size,
	               const char* label_dir,const char* label_type,const std::string label_postfix, const std::string image_type, int threads,bool shuffle){
	int classes = 3;
	//list* plist = get_paths(file_name);
	////int N = plist->size;
	//char** paths = (char**)list_to_array(plist);
	load_args args;
	read_file_name(args.image_name_list, file_name);
	args.m = args.image_name_list.size();
	get_random_paths(args.image_name_list, args.m);
	args.image_dir = image_dir;
	args.image_type = image_type;
	args.label_type = label_type;
	args.label_dir = label_dir;
	args.label_postfix = label_postfix;
	args.out_h = out_h;
	args.out_w = out_w;
	args.classes = classes;
	args.threads = threads;
	args.batch_size = batch_size;
	args.batch_count = 0;
	args.out_flag = 0;
	//args.resized = true;
	transfor_image(args);
	return args;
}

data get_batch(load_args& args) {
	data  buffer;
	args.d = &buffer;
	pthread_t load_thread = load_data(args);
	pthread_join(load_thread, 0);
	args.batch_count++;
	return buffer;
}


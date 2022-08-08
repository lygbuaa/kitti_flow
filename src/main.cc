#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
#include <fstream>
#include <signal.h>
#include "opencv2/opencv.hpp"

DEFINE_string(input_img_path, "./dataset", "input path.");
DEFINE_string(output_img_path, "./output", "output path.");

void signal_handler(int sig_num){
	std::cout << "\n@q@ --> it's quit signal: " << sig_num << ", see you later.\n";
	exit(sig_num);
}

int print_cuda(){
    int cnt = cv::cuda::getCudaEnabledDeviceCount();
    LOG(INFO) << "getCudaEnabledDeviceCount: " << cnt;
    for(int i=0; i<cnt; ++i){
        cv::cuda::printCudaDeviceInfo(i);
    }
    int dev_num = cv::cuda::getDevice();
    LOG(INFO) << "current cuda device: " << dev_num;
}

int main(int argc, char* argv[]){
    fprintf(stderr, "\n@i@ --> KittiFlow launched.\n");
    for(int i = 0; i < argc; i++){
        fprintf(stderr, "argv[%d] = %s\n", i, argv[i]);
    }
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_minloglevel = 0;
    FLAGS_logtostderr = true;
    FLAGS_colorlogtostderr = true;
    signal(SIGINT, signal_handler); //2
    signal(SIGQUIT, signal_handler);//3
    signal(SIGABRT, signal_handler);//6
    signal(SIGKILL, signal_handler);//9
    signal(SIGSEGV, signal_handler);//11
    signal(SIGTERM, signal_handler);//15

    LOG(INFO) << "KittiFlow init begin. config params:";
    LOG(INFO) << "gflags input_img_path = " << FLAGS_input_img_path;
	LOG(INFO) << "gflags output_img_path = " << FLAGS_output_img_path;

    print_cuda();

    exit(0);
}
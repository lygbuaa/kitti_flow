#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
#include <fstream>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include "SparseLK.h"
#include "DenseFB.h"
#include "DenseDIS.h"

DEFINE_string(kitti_img_path, "./dataset/data_scene_flow/training/image_2", "kitti input path.");
DEFINE_string(kitti_gt_path, "./dataset/data_scene_flow/training/flow_noc", "kitti groundtruth path.");
DEFINE_uint32(kitti_img_width, 1242, "input image width.");
DEFINE_uint32(kitti_img_height, 375, "input image height.");
DEFINE_string(output_img_path, "./output", "output path.");
DEFINE_string(glog_path, "./logs", "output path.");

void signal_handler(int sig_num){
	std::cout << "\n@q@ --> it's quit signal: " << sig_num << ", see you later.\n";
	exit(sig_num);
}

bool is_cuda_avaliable(){
    LOG(INFO) << "opencv version: " << CV_VERSION;
    int cnt = cv::cuda::getCudaEnabledDeviceCount();
    LOG(INFO) << "getCudaEnabledDeviceCount: " << cnt;
    for(int i=0; i<cnt; ++i){
        cv::cuda::printCudaDeviceInfo(i);
    }
    if(cnt > 0){
        LOG(INFO) << "current cuda device: " << cv::cuda::getDevice();
    }
    return (cnt > 0);
}

int main(int argc, char* argv[]){
    fprintf(stderr, "\n@i@ --> KittiFlow launched.\n");
    for(int i = 0; i < argc; i++){
        fprintf(stderr, "argv[%d] = %s\n", i, argv[i]);
    }
    google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_minloglevel = 0;
    FLAGS_log_dir = FLAGS_glog_path;
    // FLAGS_logtostderr = true; //stderr or file, only one destination!
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(argv[0]);

    signal(SIGINT, signal_handler); //2
    signal(SIGQUIT, signal_handler);//3
    signal(SIGABRT, signal_handler);//6
    signal(SIGKILL, signal_handler);//9
    // signal(SIGSEGV, signal_handler);//11
    signal(SIGTERM, signal_handler);//15

    is_cuda_avaliable();
    kittflow::DenseDIS algo = kittflow::DenseDIS(FLAGS_kitti_img_path, FLAGS_kitti_gt_path);
    // algo.test_dataset();
    algo.run_all(false);

   return 0;
}
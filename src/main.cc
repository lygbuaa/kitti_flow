#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
#include <fstream>
#include <signal.h>
#include <SignalBase.h>
#include <opencv2/opencv.hpp>
#include "SparseLK.h"
#include "DenseFB.h"
#include "DenseDIS.h"
#include "VpiLK.h"
#include "VpiDense.h"
#include "ThroughPutFactory.h"
#include "VpiDenseVideo.h"
#include "CvStereoSgbm.h"
#include "VpiStereo.h"

DEFINE_string(kitti_img2_path, "./dataset/data_scene_flow/training/image_2", "kitti left image path.");
DEFINE_string(kitti_img3_path, "./dataset/data_scene_flow/training/image_3", "kitti right image path.");
DEFINE_string(kitti_flow_gt_path, "./dataset/data_scene_flow/training/flow_noc", "kitti optical flow groundtruth path.");
DEFINE_string(kitti_stereo_gt_path, "./dataset/data_scene_flow/training/disp_noc_0", "kitti stereo disparity groundtruth path.");
DEFINE_uint32(kitti_img_width, 1242, "input image width. 000000 ~ 000154: 1242");
DEFINE_uint32(kitti_img_height, 375, "input image height. 000000 ~ 000154: 375");
DEFINE_string(output_img_path, "./output", "output path.");
DEFINE_string(glog_path, "./logs", "output path.");
DEFINE_uint32(stream_number, 1, "number of threads");
DEFINE_string(video_file_path, "./dataset/traffic.mp4", "video file path");
DEFINE_bool(enable_visual, false, "set true to enable visual");

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

    // signal(SIGINT, signal_handler); //2
    // signal(SIGQUIT, signal_handler);//3
    // signal(SIGABRT, signal_handler);//6
    // signal(SIGKILL, signal_handler);//9
    // // signal(SIGSEGV, signal_handler);//11
    // signal(SIGTERM, signal_handler);//15
    SignalBase::CatchSignal();

    is_cuda_avaliable();
    kittflow::VpiDense algo = kittflow::VpiDense(FLAGS_kitti_img2_path, FLAGS_kitti_flow_gt_path);
    algo.run_all(false);

    // kittflow::ThroughtPutFactory<kittflow::VpiDenseVideo> algo = kittflow::ThroughtPutFactory<kittflow::VpiDenseVideo>(FLAGS_video_file_path, FLAGS_stream_number, FLAGS_enable_visual);
    // algo.init_streams();
    // algo.run_streams();

    // kittflow::VpiStereo algo = kittflow::VpiStereo(FLAGS_kitti_img2_path, FLAGS_kitti_img3_path, FLAGS_kitti_stereo_gt_path);
    // algo.run_all(true);

    return 0;
}

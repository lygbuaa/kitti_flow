#pragma once

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <dirent.h>
#include <libgen.h>
#include <chrono>
#include "devkit_scene_flow/devkit/cpp/io_flow.h"

DECLARE_string(kitti_img_path);
DECLARE_string(kitti_gt_path);
DECLARE_uint32(kitti_img_width);
DECLARE_uint32(kitti_img_height);
DECLARE_string(output_img_path);

namespace kittflow
{

class KittiFlowBase
{
private:
    std::deque<std::string> img_file_list_;
    std::deque<std::string> gt_file_list_;

public:
    KittiFlowBase(const std::string& img_path, const std::string& gt_path){
        img_file_list_ = std::move(list_dir(img_path));
        gt_file_list_ = std::move(list_dir(gt_path));
        LOG(INFO) << "img_file_list_: " << img_file_list_.size() << ", gt_file_list_: " << gt_file_list_.size();
    }

    ~KittiFlowBase(){}

    uint64_t current_micros() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::time_point_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now()).time_since_epoch()).count();
    }

    std::deque<std::string> list_dir(const std::string dirpath){
        DIR* dp;
        std::deque<std::string> v_file_list;
        dp = opendir(dirpath.c_str());
        if (nullptr == dp){
            LOG(ERROR) << "read dirpath failed: " << dirpath;
            return v_file_list;
        }

        struct dirent* entry;
        while((entry = readdir(dp))){
            if(DT_DIR == entry->d_type){
                LOG(WARNING) << "subdirectory ignored: " << entry->d_name;
                continue;
            }else if(DT_REG == entry->d_type){
                std::string filepath = dirpath + "/" + entry->d_name;
                v_file_list.emplace_back(filepath);
            }
        }
        //sort into ascending order
        std::sort(v_file_list.begin(), v_file_list.end());
        // for(auto& fp : v_file_list){
        //     LOG(INFO) << "filepath: " << fp;
        // }

        return v_file_list;
    }

    std::map<std::string, std::string> get_img_pair(){
        std::map<std::string, std::string> img_pair;
        // check if empty
        if(img_file_list_.empty() || gt_file_list_.empty()){
            return img_pair;
        }

        img_pair["prev_img"] = img_file_list_.front();
        img_file_list_.pop_front();
        img_pair["this_img"] = img_file_list_.front();
        img_file_list_.pop_front();
        img_pair["gt_img"] = gt_file_list_.front();
        gt_file_list_.pop_front();

        // LOG(INFO) << "prev_img: " << img_pair["prev_img"];
        // LOG(INFO) << "this_img: " << img_pair["this_img"];
        // LOG(INFO) << "gt_img: " << img_pair["gt_img"];
        return img_pair;
    }

    std::shared_ptr<FlowImage> load_flow_gt(const std::string gt_img_path){
        std::shared_ptr<FlowImage> gt_ptr(new FlowImage(gt_img_path));
        LOG(INFO) << "load gt: " << gt_img_path << ", max value: " << gt_ptr->maxFlow();
        return gt_ptr;
    }

    void test_dataset(){
        std::map<std::string, std::string> img_pair = get_img_pair();

        cv::Mat prev_img = cv::imread(img_pair["prev_img"]);
        cv::Mat this_img = cv::imread(img_pair["this_img"]);
        cv::Mat gt_img = cv::imread(img_pair["gt_img"]);

        cv::namedWindow("kitti", cv::WINDOW_NORMAL);
        cv::imshow("kitti", prev_img);
        cv::waitKey(2000);
        cv::imshow("kitti", this_img);
        cv::waitKey(2000);
        cv::imshow("kitti", gt_img);
        cv::waitKey(2000);
    }

};

}
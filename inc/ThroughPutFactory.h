#pragma once

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include "VideoFlowBase.h"

namespace kittflow
{
template <typename T>
class ThroughtPutFactory
{
private:
    std::vector<std::shared_ptr<T>> objects_;
    uint32_t stream_number_ = 0;

public:
    ThroughtPutFactory(const std::string& video_path, uint32_t stream_number = 1, bool enable_visual=false){
        if(enable_visual){
            CHECK_EQ(stream_number, 1);
        }
        stream_number_ = stream_number;
        for(int i=0; i<stream_number; i++){
            objects_.emplace_back(std::shared_ptr<T>(new T(video_path, enable_visual)));
        }
        LOG(INFO) << "total objects: " << objects_.size();
    }

    ~ThroughtPutFactory(){}

    uint64_t current_micros() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::time_point_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now()).time_since_epoch()).count();
    }

    void run_streams(){
        const uint64_t start_us = current_micros();
        LOG(INFO) << "streams start @" << start_us;

        for(int i=0; i<stream_number_; i++){
            auto obj = objects_[i];
            obj -> start_work();
            LOG(INFO) << "start stream-" << i;
        }

        for(int i=0; i<stream_number_; i++){
            auto obj = objects_[i];
            obj -> join_work();
            LOG(INFO) << "exit stream-" << i;
        }

        const uint64_t end_us = current_micros();
        double dsec = (end_us-start_us)*1e-6;

        uint32_t total_frame_num = 0;
        for(int i=0; i<stream_number_; i++){
            auto obj = objects_[i];
            total_frame_num += obj->get_frame_num();
        }
        double throughput = total_frame_num/dsec;
        LOG(INFO) << "all streams done, total frame: " << total_frame_num << ", elapsed sec: " << dsec << ", throughput: " << throughput;
    }
};

}
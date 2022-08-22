#pragma once

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <thread>

namespace kittflow
{
class VideoFlowBase
{
protected:
    std::string input_video_path_;
    cv::VideoCapture video_cap_;
    int32_t video_width_;
    int32_t video_height_;
    int32_t fps_;
    bool enable_visual_;
    uint32_t frame_id_;
    std::shared_ptr<std::thread> pthread_;

public:
    VideoFlowBase(const std::string& video_path, bool enable_visual=false){
        input_video_path_ = video_path;
        enable_visual_ = enable_visual;
        if(!video_cap_.open(video_path)){
            LOG(FATAL) << "open video failed: " << video_path;
        }
        video_width_ = int32_t(video_cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        video_height_ = int32_t(video_cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps_ = int32_t(video_cap_.get(cv::CAP_PROP_FPS));
        LOG(INFO) << "test throughput on: " << video_path << ", w: " << video_width_ << ", h: " << video_height_ << ", fps: " << fps_;
    }

    ~VideoFlowBase(){}

    uint64_t current_micros() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::time_point_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now()).time_since_epoch()).count();
    }

    /* return false if video reach eof */
    virtual bool next_image(cv::Mat& img){
        return video_cap_.read(img);
    }

    virtual bool init_stream(cv::Mat& cvPrevFrame){
        LOG(INFO) << "init_stream from base class!";
        return true;
    }

    virtual bool run_once(cv::Mat& img){
        LOG(INFO) << "run_once from base class!";
        std::this_thread::sleep_for (std::chrono::milliseconds(1));
        return true;
    }

    virtual bool warm_up(){
        frame_id_ = 0;
        cv::Mat img;
        /* we need one frame to init stream */
        if(next_image(img) && init_stream(img)){
            LOG(INFO) << "stream init done!";
        }else{
            LOG(FATAL) << "stream init failed!";
        }

        /* warm-up stage, this is required by https://docs.nvidia.com/vpi/algo_performance.html */
        const uint64_t start_us = current_micros();
        LOG(INFO) << "warm-up start @" << start_us;
        while(next_image(img)){
            run_once(img);
            frame_id_++;
            LOG_EVERY_N(INFO, 100) << "warm-up frame-" << frame_id_;
        }
        const uint64_t end_us = current_micros();
        double dt = (end_us-start_us)*1e-6;
        double latency = dt/frame_id_;
        LOG(INFO) << "warm-up done, total frame: " << frame_id_ << ", elapsed sec: " << dt << ", latency: " << latency;
        return true;
    }

    virtual void run_loop(){
        frame_id_ = 0;
        cv::Mat img;
        video_cap_.set(cv::CAP_PROP_POS_FRAMES, 0);

        const uint64_t start_us = current_micros();
        LOG(INFO) << "loop start @" << start_us;
        while(next_image(img)){
            run_once(img);
            frame_id_++;
            LOG_EVERY_N(INFO, 100) << "process frame-" << frame_id_;
        }
        const uint64_t end_us = current_micros();
        double dt = (end_us-start_us)*1e-6;
        double latency = dt/frame_id_;
        LOG(INFO) << "loop end, total frame: " << frame_id_ << ", elapsed sec: " << dt << ", latency: " << latency;
    }

    uint32_t get_frame_num() const {
        return frame_id_;
    }

    void start_work(){
        pthread_ = std::shared_ptr<std::thread>(new std::thread(&VideoFlowBase::run_loop, this));
    }

    void join_work(){
        pthread_ -> join();
    }
};

}
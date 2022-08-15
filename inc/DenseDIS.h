#pragma once

#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "KittiFlowBase.h"

DECLARE_string(output_img_path);

namespace kittflow
{

class DenseDIS : public KittiFlowBase
{
private:
    uint64_t average_us_ = 0;
    cv::Ptr<cv::DenseOpticalFlow> algo_;

public:
    DenseDIS(const std::string& img_path, const std::string& gt_path)
    : KittiFlowBase(img_path, gt_path)
    {
        /*
        enum  	{
            PRESET_ULTRAFAST = 0,
            PRESET_FAST = 1,
            PRESET_MEDIUM = 2
        }
        */
        algo_ = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
        LOG(INFO) << "DenseDIS init: " << algo_->getDefaultName();
    }

    ~DenseDIS(){}

    std::string gen_output_path(const std::string input_img_path){
        const char* file_name = basename(const_cast<char*>(input_img_path.c_str()));
        std::string output_path = FLAGS_output_img_path + "/" + file_name + ".dense_dis.png";
        return output_path;
    }

    /* flow = cv::Mat(CV_32FC2); */
    FLOW_WRAPPER_t wrap_flow(const int img_h, const int img_w, const cv::Mat& flow, std::shared_ptr<FlowImage> gt_ptr){
        assert(img_h == gt_ptr->height());
        assert(img_w == gt_ptr->width());
        std::shared_ptr<FlowImage> flow_ptr(new FlowImage(img_w, img_h));
        std::shared_ptr<IntegerImage> mask_ptr(new IntegerImage(img_w, img_h));

        static uint32_t total_counter = 0;
        static uint32_t good_counter = 0;
        for(int y=0; y<img_h; ++y){
            for(int x=0; x<img_w; ++x){
                total_counter ++;
                cv::Vec2f val = flow.at<cv::Vec2f>(y, x);
                float fx = val[0];
                float fy = val[1];
                mask_ptr -> setValue(x, y, 1);
                flow_ptr -> setFlowU(x, y, fx);
                flow_ptr -> setFlowV(x, y, fy);
                flow_ptr -> setValid(x, y, true);

                float gt_x = gt_ptr->getFlowU(x, y);
                float gt_y = gt_ptr->getFlowV(x, y);
                float gt_mag = gt_ptr->getFlowMagnitude(x,y);
                float ferr_abs = std::sqrt((gt_x-fx)*(gt_x-fx) + (gt_y-fy)*(gt_y-fy));
                float ferr_per = ferr_abs/gt_mag;
                if(ferr_abs < ABS_THRESH || ferr_per < REL_THRESH){
                    good_counter ++;
                }
            }
        }

        LOG(INFO) << "total_counter: " << total_counter << ", good_counter: " << good_counter;
        return FLOW_WRAPPER_t(flow_ptr, mask_ptr);
    }

    void run_all(bool enable_visual=false){
        // kitti image number from 000000
        unsigned int counter = 0;
        std::vector<float> errors_acc(12, 0.0f);
        while(true){
            std::map<std::string, std::string> img_pair = get_img_pair();
            if(img_pair.empty()){
                break;
            }     
            ++ counter;
            // if(counter > 150) break;

            std::vector<float> errors = run_once(counter, img_pair, enable_visual);
            // print_error_report(errors);
            assert(errors.size() == 12);
            for(int i=0; i<errors.size(); ++i){
                errors_acc[i] += errors[i];
            }
        }
        average_us_ /= counter;
        LOG(INFO) << "all " << counter << " tests done, average_us: " << average_us_;
        print_error_report(errors_acc);
    }

    std::vector<float> run_once(unsigned int counter, std::map<std::string, std::string>& img_pair, bool enable_visual=false){
        cv::Mat prev_img = cv::imread(img_pair["prev_img"]);
        cv::Mat this_img = cv::imread(img_pair["this_img"]);
        // cv::Mat gt_img = cv::imread(img_pair["gt_img"]);

        cv::Mat prev_gray, this_gray;
        cv::cvtColor(prev_img, prev_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(this_img, this_gray, cv::COLOR_BGR2GRAY);
        assert(!prev_gray.empty());
        assert(!this_gray.empty());
        const cv::Size original_size = this_gray.size();

        /* image shape change can cause Segmentation fault, but not the full reason. */
        /* Segmentation fault backtrace: cv::parallel_for_(cv::Range const&, cv::ParallelLoopBody const&, double) */
        if(original_size.height!=FLAGS_kitti_img_height || original_size.width!=FLAGS_kitti_img_width){
            cv::resize(prev_gray, prev_gray, cv::Size(FLAGS_kitti_img_width, FLAGS_kitti_img_height));
            cv::resize(this_gray, this_gray, cv::Size(FLAGS_kitti_img_width, FLAGS_kitti_img_height));
        }

        cv::Mat flow(this_gray.size(), CV_32FC2);

        /*
            prev	first 8-bit single-channel input image.
            next	second input image of the same size and the same type as prev.
            flow	computed flow image that has the same size as prev and type CV_32FC2.
            pyr_scale	parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
            levels	number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
            winsize	averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
            iterations	number of iterations the algorithm does at each pyramid level.
            poly_n	size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
            poly_sigma	standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
        */
        const uint64_t start_us = current_micros();
        algo_ -> calc(prev_gray, this_gray, flow);
        average_us_ += (current_micros() - start_us);

        if(original_size.height!=FLAGS_kitti_img_height || original_size.width!=FLAGS_kitti_img_width){
            cv::resize(flow, flow, original_size);
        }

        std::shared_ptr<FlowImage> gt_ptr = load_flow_gt(img_pair["gt_img"]);
        FLOW_WRAPPER_t flow_wrapper = wrap_flow(original_size.height, original_size.width, flow, gt_ptr);
        std::vector<float> errors = calc_flow_error(gt_ptr, flow_wrapper);

        if(enable_visual){
            cv::Mat flow_parts[2];
            cv::split(flow, flow_parts);
            cv::Mat magnitude, angle, magn_norm;
            cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
            cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
            angle *= ((1.f / 360.f) * (180.f / 255.f));
            //build hsv image
            cv::Mat _hsv[3], hsv, hsv8, bgr;
            _hsv[0] = angle;
            _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
            _hsv[2] = magn_norm;
            cv::merge(_hsv, 3, hsv);
            hsv.convertTo(hsv8, CV_8U, 255.0);
            cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

            cv::namedWindow("kitti", cv::WINDOW_NORMAL);
            cv::imshow("kitti", bgr);
            cv::waitKey(300);
            cv::imwrite(gen_output_path(img_pair["this_img"]), bgr);
        }

        return errors;
    }

};

}
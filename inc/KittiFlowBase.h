#pragma once

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <dirent.h>
#include <libgen.h>
#include <chrono>
#include "devkit_scene_flow/devkit/cpp/io_flow.h"
#include "devkit_scene_flow/devkit/cpp/io_integer.h"

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
    using FLOW_WRAPPER_t = std::tuple<std::shared_ptr<FlowImage>, std::shared_ptr<IntegerImage>>;
    /* copied from dataset/devkit_scene_flow/devkit/cpp/evaluate_scene_flow.cpp */
    static constexpr float ABS_THRESH = 3.0f;
    static constexpr float REL_THRESH = 0.05f;

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
        // interpolate all missing (=invalid) optical flow vectors, no need for groudtruth
        // gt_ptr -> interpolateBackground();
        LOG(INFO) << "load gt: " << gt_img_path << ", valid density: " << gt_ptr->getDensity();
        return gt_ptr;
    }

    /* 
        copied from dataset/devkit_scene_flow/devkit/cpp/evaluate_scene_flow.cpp:
        F_gt: can be F_gt_noc or F_gt_occ, without interpolateBackground();
        F_ipol = F_orig.interpolateBackground();
        O_map: object map (0:background, >0:foreground)
    */
    std::vector<float> flowErrorsOutlier(const std::shared_ptr<FlowImage>& F_gt, const std::shared_ptr<FlowImage>& F_orig, const std::shared_ptr<FlowImage>& F_ipol, const std::shared_ptr<IntegerImage>& O_map) {
            // check file size
        if (F_gt->width()!=F_orig->width() || F_gt->height()!=F_orig->height()) {
            LOG(ERROR) << "ERROR: Wrong file size!";
            throw 1;
        }

        // extract width and height
        int32_t width  = F_gt->width();
        int32_t height = F_gt->height();

        // init errors
        std::vector<float> errors;
        int32_t num_errors_bg = 0;
        int32_t num_pixels_bg = 0;
        int32_t num_errors_bg_result = 0;
        int32_t num_pixels_bg_result = 0;
        int32_t num_errors_fg = 0;
        int32_t num_pixels_fg = 0;
        int32_t num_errors_fg_result = 0;
        int32_t num_pixels_fg_result = 0;
        int32_t num_errors_all = 0;
        int32_t num_pixels_all = 0;
        int32_t num_errors_all_result = 0;
        int32_t num_pixels_all_result = 0;

        // for all pixels do
        for (int32_t u=0; u<width; u++) {
            for (int32_t v=0; v<height; v++) {
                float fu = F_gt->getFlowU(u,v)-F_ipol->getFlowU(u,v);
                float fv = F_gt->getFlowV(u,v)-F_ipol->getFlowV(u,v);
                float f_dist = std::sqrt(fu*fu+fv*fv);
                float f_mag  = F_gt->getFlowMagnitude(u,v);
                bool  f_err  = f_dist>ABS_THRESH && f_dist/f_mag>REL_THRESH;
                // 0 means measure invalid, means background
                if (O_map->getValue(u,v) == 0) {
                    if (F_gt->isValid(u,v)) {
                        num_pixels_bg++;
                        if (f_err) num_errors_bg++;
                        if (F_orig->isValid(u,v)) {
                            if (f_err) num_errors_bg_result++;
                            num_pixels_bg_result++;
                        }
                    }
                // 1 means measure valid, means foreground
                } else {
                    if (F_gt->isValid(u,v)) {
                        num_pixels_fg++;
                        if (f_err) num_errors_fg++;
                        if (F_orig->isValid(u,v)) {
                            if (f_err) num_errors_fg_result++;
                            num_pixels_fg_result++;
                        }
                    }
                }

                if (F_gt->isValid(u,v)) {
                    num_pixels_all++;
                    if (f_err) num_errors_all++;
                    if (F_orig->isValid(u,v)) {
                        if (f_err) num_errors_all_result++;
                        num_pixels_all_result++;
                    }
                }
            }
        }

        /* push back errors and pixel count, postprocess is very straight-forward:
        for (int32_t i=0; i<12; i+=2)
            fprintf(stats_file,"%f ",errors_flow_occ[i]/max(errors_flow_occ[i+1],1.0f)); 
        fprintf(stats_file,"%f ",errors_flow_occ[11]/max(errors_flow_occ[9],1.0f));
        */
        //FL-bg, "all pixels" and "estimated pixels"
        errors.push_back(num_errors_bg);        //0
        errors.push_back(num_pixels_bg);        //1
        errors.push_back(num_errors_bg_result); //2
        errors.push_back(num_pixels_bg_result); //3
        //FL-fg, "all pixels" and "estimated pixels"
        errors.push_back(num_errors_fg);        //4
        errors.push_back(num_pixels_fg);        //5
        errors.push_back(num_errors_fg_result); //6
        errors.push_back(num_pixels_fg_result); //7
        //FL-all, "all pixels" and "estimated pixels"
        errors.push_back(num_errors_all);       //8
        errors.push_back(num_pixels_all);       //9
        errors.push_back(num_errors_all_result);//10
        errors.push_back(num_pixels_all_result);//11

        // push back density, duplicated
        // errors.push_back((float)num_pixels_all_result/std::max((float)num_pixels_all,1.0f));

        // return errors
        return errors;
    }

    std::vector<float> calc_flow_error(const std::shared_ptr<FlowImage>& gt_ptr, const FLOW_WRAPPER_t& flow_wrapper){
        std::shared_ptr<FlowImage> F_orig = std::get<0>(flow_wrapper);
        /* F_ipol should copy data from F_orig, should not share data ptr */
        std::shared_ptr<FlowImage> F_ipol(new FlowImage(*F_orig));
        F_ipol -> interpolateBackground();

        std::shared_ptr<IntegerImage> O_map = std::get<1>(flow_wrapper);
        std::vector<float> errors = flowErrorsOutlier(gt_ptr, F_orig, F_ipol, O_map);
        return errors;
    }

    std::string print_error_report(std::vector<float>& errors){
        assert(errors.size() == 12);
        std::stringstream ss;

        float bg_outlier_percent_all = errors[0] / std::max(errors[1], 1.0f);
        float bg_outlier_percent_estimated = errors[2] / std::max(errors[3], 1.0f);        
        // the foreground count should be important.
        float fg_pixels_cnt = errors[7];
        float fg_outlier_percent_all = errors[4] / std::max(errors[5], 1.0f);
        float fg_outlier_percent_estimated = errors[6] / std::max(errors[7], 1.0f);

        float all_outlier_percent_all = errors[8] / std::max(errors[9], 1.0f);
        float all_outlier_percent_estimated = errors[10] / std::max(errors[11], 1.0f);  

        float density = errors[11] / std::max(errors[9], 1.0f);

        ss << "[non-occluded error report] [all pixels] fl-bg: " << bg_outlier_percent_all << ", fl-fg: " << fg_outlier_percent_all << ", fl-all: " << all_outlier_percent_all \
        << ", [estimated pixels] fl-bg: " << bg_outlier_percent_estimated << ", fl-fg: " << fg_outlier_percent_estimated << ", fl-all: " << all_outlier_percent_estimated \
        << ", foreground-pixels: " << fg_pixels_cnt << ", density: " << density;
        ss << "\nerrors: \n";
        for(int i=0; i<errors.size(); ++i){
            ss << errors[i] << std::endl;
        } 

        std::string msg = ss.str();
        LOG(INFO) << msg;
        return msg;
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
#pragma once

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <dirent.h>
#include <libgen.h>
#include <chrono>
#include "devkit_scene_flow/devkit/cpp/io_disp.h"
#include "devkit_scene_flow/devkit/cpp/io_integer.h"

DECLARE_string(kitti_img2_path);
DECLARE_string(kitti_img3_path);
DECLARE_string(kitti_stereo_gt_path);
DECLARE_uint32(kitti_img_width);
DECLARE_uint32(kitti_img_height);
DECLARE_string(output_img_path);

namespace kittflow
{
class KittiStereoBase
{
private:
    std::deque<std::string> left_img_file_list_;
    std::deque<std::string> right_img_file_list_;
    std::deque<std::string> gt_file_list_;
public:
    using STEREO_WRAPPER_t = std::tuple<std::shared_ptr<DisparityImage>, std::shared_ptr<IntegerImage>>;
    /* copied from dataset/devkit_scene_flow/devkit/cpp/evaluate_scene_flow.cpp */
    static constexpr float ABS_THRESH = 3.0f;
    static constexpr float REL_THRESH = 0.05f;

public:
    KittiStereoBase(const std::string& left_img_path, const std::string& right_img_path, const std::string& gt_path){
        left_img_file_list_ = std::move(list_dir(left_img_path));
        right_img_file_list_ = std::move(list_dir(right_img_path));
        gt_file_list_ = std::move(list_dir(gt_path));
        LOG(INFO) << "left_img_file_list_: " << left_img_file_list_.size() << ", right_img_file_list_: " << right_img_file_list_.size() << ", gt_file_list_: " << gt_file_list_.size();
    }

    ~KittiStereoBase(){}

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
        if(left_img_file_list_.empty() || right_img_file_list_.empty() || gt_file_list_.empty()){
            return img_pair;
        }

        /* only retrieve *_10.png, and deprecate *_11.png */
        img_pair["left_img"] = left_img_file_list_.front();
        left_img_file_list_.pop_front();
        left_img_file_list_.pop_front();
        img_pair["right_img"] = right_img_file_list_.front();
        right_img_file_list_.pop_front();
        right_img_file_list_.pop_front();
        img_pair["gt_img"] = gt_file_list_.front();
        gt_file_list_.pop_front();

        // LOG(INFO) << "prev_img: " << img_pair["prev_img"];
        // LOG(INFO) << "this_img: " << img_pair["this_img"];
        // LOG(INFO) << "gt_img: " << img_pair["gt_img"];
        return img_pair;
    }

    std::shared_ptr<DisparityImage> load_stereo_gt(const std::string gt_img_path){
        std::shared_ptr<DisparityImage> gt_ptr(new DisparityImage(gt_img_path));
        // interpolate all missing (=invalid) optical flow vectors, no need for groudtruth
        // gt_ptr -> interpolateBackground();
        LOG(INFO) << "load gt: " << gt_img_path << ", max disparity: " << gt_ptr->maxDisp();
        return gt_ptr;
    }

    /*
        copied from dataset/devkit_scene_flow/devkit/cpp/evaluate_scene_flow.cpp:
        F_gt: can be F_gt_noc or F_gt_occ, without interpolateBackground();
        F_ipol = F_orig.interpolateBackground();
        O_map: object map (0:background, >0:foreground)
    */
    std::vector<float> disparityErrorsOutlier(const std::shared_ptr<DisparityImage>& D_gt, const std::shared_ptr<DisparityImage>& D_orig, const std::shared_ptr<DisparityImage>& D_ipol, const std::shared_ptr<IntegerImage>& O_map) {
        // check file size
        if (D_gt->width()!=D_orig->width() || D_gt->height()!=D_orig->height()) {
          LOG(FATAL) << "Wrong file size!";
        }

        // extract width and height
        int32_t width  = D_gt->width();
        int32_t height = D_gt->height();

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
            if (D_gt->isValid(u,v)) {
              float d_gt  = D_gt->getDisp(u,v);
              float d_est = D_ipol->getDisp(u,v);
              bool  d_err = fabs(d_gt-d_est)>ABS_THRESH && fabs(d_gt-d_est)/fabs(d_gt)>REL_THRESH;
              if (O_map->getValue(u,v)==0) {
                if (d_err)
                  num_errors_bg++;
                num_pixels_bg++;
                if (D_orig->isValid(u,v)) {
                  if (d_err)
                    num_errors_bg_result++;
                  num_pixels_bg_result++;
                }
              } else {
                if (d_err)
                  num_errors_fg++;
                num_pixels_fg++;
                if (D_orig->isValid(u,v)) {
                  if (d_err)
                    num_errors_fg_result++;
                  num_pixels_fg_result++;
                }
              }
              if (d_err)
                num_errors_all++;
              num_pixels_all++;
              if (D_orig->isValid(u,v)) {
                if (d_err)
                  num_errors_all_result++;
                num_pixels_all_result++;
              }
            }
          }
        }

        // push back errors and pixel count
        errors.push_back(num_errors_bg);
        errors.push_back(num_pixels_bg);
        errors.push_back(num_errors_bg_result);
        errors.push_back(num_pixels_bg_result);
        errors.push_back(num_errors_fg);
        errors.push_back(num_pixels_fg);
        errors.push_back(num_errors_fg_result);
        errors.push_back(num_pixels_fg_result);
        errors.push_back(num_errors_all);
        errors.push_back(num_pixels_all);
        errors.push_back(num_errors_all_result);
        errors.push_back(num_pixels_all_result);

        // push back density
        errors.push_back((float)num_pixels_all_result/std::max((float)num_pixels_all, 1.0f));

        // return errors
        return errors;
    }

    std::vector<float> calc_stereo_error(const std::shared_ptr<DisparityImage>& gt_ptr, const STEREO_WRAPPER_t& stereo_wrapper){
        std::shared_ptr<DisparityImage> D_orig = std::get<0>(stereo_wrapper);
        /* F_ipol should copy data from F_orig, should not share data ptr */
        std::shared_ptr<DisparityImage> D_ipol(new DisparityImage(*D_orig));
        D_ipol -> interpolateBackground();

        std::shared_ptr<IntegerImage> O_map = std::get<1>(stereo_wrapper);
        std::vector<float> errors = disparityErrorsOutlier(gt_ptr, D_orig, D_ipol, O_map);
        return errors;
    }

    /*
      D1: Percentage of stereo disparity outliers in first frame
      D2: Percentage of stereo disparity outliers in second frame
    */
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

        ss << "[non-occluded error report] [all pixels] D1-bg: " << bg_outlier_percent_all << ", D1-fg: " << fg_outlier_percent_all << ", D1-all: " << all_outlier_percent_all \
        << ", [estimated pixels] D1-bg: " << bg_outlier_percent_estimated << ", D1-fg: " << fg_outlier_percent_estimated << ", D1-all: " << all_outlier_percent_estimated \
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
        cv::namedWindow("kitti", cv::WINDOW_NORMAL);
        for(int i=0; i<200; i++){
            std::map<std::string, std::string> img_pair = get_img_pair();
            cv::Mat left_img = cv::imread(img_pair["left_img"]);
            cv::Mat right_img = cv::imread(img_pair["right_img"]);
            cv::Mat gt_img = cv::imread(img_pair["gt_img"]);

            cv::imshow("kitti", left_img);
            cv::waitKey(1000);
            cv::imshow("kitti", right_img);
            cv::waitKey(1000);
            cv::imshow("kitti", gt_img);
            cv::waitKey(1000);
        }

    }


};

}

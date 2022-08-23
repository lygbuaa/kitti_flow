#pragma once

#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "KittiStereoBase.h"

DECLARE_string(output_img_path);
DECLARE_uint32(kitti_img_width);

namespace kittflow
{

/* opencv/samples/cpp/stereo_match.cpp */
class CvStereoSgbm : public KittiStereoBase
{
private:
    uint64_t average_us_ = 0;
    cv::Ptr<cv::StereoSGBM> sgbm_;
    int numberOfDisparities_;

public:
    CvStereoSgbm(const std::string& left_img_path, const std::string& right_img_path, const std::string& gt_path)
    : KittiStereoBase(left_img_path, right_img_path, gt_path)
    {
        init_sgbm();
        LOG(INFO) << "CvStereoSgbm init.";
    }

    ~CvStereoSgbm(){}

    void init_sgbm(){
        /*
        minDisparity	Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
        numDisparities	Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
        blockSize	Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        P1	The first parameter controlling the disparity smoothness. See below.
        P2	The second parameter controlling the disparity smoothness. The larger the values are, the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 > P1 . See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8*number_of_image_channels*blockSize*blockSize and 32*number_of_image_channels*blockSize*blockSize , respectively).
        disp12MaxDiff	Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
        preFilterCap	Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function.
        uniquenessRatio	Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
        speckleWindowSize	Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleRange	Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
        mode	Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming algorithm. It will consume O(W*H*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures. By default, it is set to false .
        */
        sgbm_ = cv::StereoSGBM::create(0, 16, 3);
        numberOfDisparities_ = ((FLAGS_kitti_img_width/8) + 15) & -16;
        LOG(INFO) << "numberOfDisparities_: " << numberOfDisparities_;
        int sgbmWinSize = 3;
        int cn = 3; //image channel
        sgbm_->setPreFilterCap(63);
        sgbm_->setBlockSize(sgbmWinSize);
        sgbm_->setP1(8*cn*sgbmWinSize*sgbmWinSize);
        sgbm_->setP2(32*cn*sgbmWinSize*sgbmWinSize);
        sgbm_->setMinDisparity(0);
        sgbm_->setNumDisparities(numberOfDisparities_);
        sgbm_->setUniquenessRatio(10);
        sgbm_->setSpeckleWindowSize(100);
        sgbm_->setSpeckleRange(32);
        sgbm_->setDisp12MaxDiff(1);
        sgbm_->setMode(cv::StereoSGBM::MODE_SGBM);
    }

    std::string gen_output_path(const std::string input_img_path){
        const char* file_name = basename(const_cast<char*>(input_img_path.c_str()));
        std::string output_path = FLAGS_output_img_path + "/" + file_name + ".sgbm.png";
        return output_path;
    }

    /* disp = cv::Mat(CV_16S); */
    STEREO_WRAPPER_t wrap_stereo(const int img_h, const int img_w, const cv::Mat& disp, std::shared_ptr<DisparityImage> gt_ptr){
        std::shared_ptr<DisparityImage> disp_ptr(new DisparityImage(img_w, img_h));
        std::shared_ptr<IntegerImage> mask_ptr(new IntegerImage(img_w, img_h));

        cv::Mat floatDisp;
        float disparity_multiplier = 1.0f;
        /* sgbm output CV_16S, every disparity is multiplied by 16 */
        if (disp.type() == CV_16S){
            disparity_multiplier = 16.0f;
        }
        disp.convertTo(floatDisp, CV_32F, 1.0f/disparity_multiplier);

        static uint32_t total_counter = 0;
        static uint32_t good_counter = 0;
        float val_max = 0.0f;
        float val_min = 0.0f;
        float gt_max = 0.0f;
        float gt_min = 0.0f;
        for(int y=0; y<img_h; ++y){
            for(int x=0; x<img_w; ++x){
                total_counter ++;
                float val = floatDisp.at<float>(y, x);
                disp_ptr -> setDisp(x, y, val);

                if(val<0 || val>numberOfDisparities_){
                  disp_ptr -> setInvalid(x, y);
                  mask_ptr -> setValue(x, y, 0);
                }else{
                  mask_ptr -> setValue(x, y, 1);
                }
                if(val > val_max){
                  val_max = val;
                }else if(val < val_min){
                  val_min = val;
                }

                float gt_val = gt_ptr->getDisp(x, y);
                if(gt_val > gt_max){
                  gt_max = gt_val;
                }else if(gt_val < gt_min){
                  gt_min = gt_val;
                }

                float ferr_abs = fabs(gt_val-val);
                float ferr_per = ferr_abs/fabs(gt_val);
                if(ferr_abs < ABS_THRESH || ferr_per < REL_THRESH){
                    good_counter ++;
                }
            }
        }
        LOG(INFO) << "gt_max: " << gt_max << ", gt_min: " << gt_min << ", val_max: " << val_max << ", val_min: " << val_min;
        LOG(INFO) << "total_counter: " << total_counter << ", good_counter: " << good_counter;
        return STEREO_WRAPPER_t(disp_ptr, mask_ptr);
    }

    void run_all(bool enable_visual=false){
        // kitti image number from 000000
        unsigned int counter = 0;
        std::vector<float> errors_acc(12, 0.0f);
        while(true){
            std::map<std::string, std::string> img_pair = get_img_pair();
            if(img_pair.empty()){
                average_us_ /= counter;
                LOG(INFO) << "all " << counter << " tests done, average_us: " << average_us_;
                break;
            }
            std::vector<float> errors = run_once(counter, img_pair, enable_visual);
            // print_error_report(errors);

            ++ counter;
            for(int i=0; i<errors.size(); ++i){
                errors_acc[i] += errors[i];
            }

            // if(counter > 10) break;
        }
        print_error_report(errors_acc);
    }

    std::vector<float> run_once(unsigned int counter, std::map<std::string, std::string>& img_pair, bool enable_visual=false){
        cv::Mat left_img = cv::imread(img_pair["left_img"]);
        cv::Mat right_img = cv::imread(img_pair["right_img"]);
        cv::Mat disp;

        const uint64_t start_us = current_micros();
        /* sgbm output is CV_16S */
        sgbm_->compute(left_img, right_img, disp);
        average_us_ += (current_micros() - start_us);

        std::shared_ptr<DisparityImage> gt_ptr = load_stereo_gt(img_pair["gt_img"]);
        STEREO_WRAPPER_t stereo_wrapper = wrap_stereo(left_img.rows, left_img.cols, disp, gt_ptr);
        std::vector<float> errors(12, 0.0f);
        errors = calc_stereo_error(gt_ptr, stereo_wrapper);

        if(enable_visual){
            cv::Mat disp8;
            disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities_*16.0f));
            cv::namedWindow("kitti", cv::WINDOW_NORMAL);
            cv::imshow("kitti", disp8);
            cv::waitKey(1000);
            cv::imwrite(gen_output_path(img_pair["left_img"]), disp8);
        }

        return errors;
    }

};

}

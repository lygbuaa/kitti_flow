#pragma once

#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "KittiFlowBase.h"

DECLARE_string(output_img_path);

namespace kittflow
{

class SparseLK : public KittiFlowBase
{
private:
    uint64_t average_us_ = 0;

public:
    SparseLK(const std::string& img_path, const std::string& gt_path)
    : KittiFlowBase(img_path, gt_path)
    {
        LOG(INFO) << "SparseLK init.";
    }

    ~SparseLK(){}

    std::string gen_output_path(const std::string input_img_path){
        const char* file_name = basename(const_cast<char*>(input_img_path.c_str()));
        std::string output_path = FLAGS_output_img_path + "/" + file_name + ".sparse_lk.png";
        return output_path;
    }

    FLOW_WRAPPER_t wrap_flow(const int img_h, const int img_w, const std::vector<cv::Point2f>& p0, const std::vector<cv::Point2f>& p1, const std::vector<unsigned char>& status, std::shared_ptr<FlowImage> gt_ptr){
        std::shared_ptr<FlowImage> flow_ptr(new FlowImage(img_w, img_h));
        std::shared_ptr<IntegerImage> mask_ptr(new IntegerImage(img_w, img_h));
        DLOG(INFO) << "wrap optical flow into FlowImage, h: " << img_h << ", w: " << img_w;

        static uint32_t total_counter = 0;
        static uint32_t good_counter = 0;
        for(uint i = 0; i < p0.size(); i++){
            // Select good points
            if(status[i]==1) {
                int32_t u = p0[i].x;
                int32_t v = p0[i].y;
                cv::Point2f dp = p1[i] - p0[i];
                if(u>img_w || v>img_h){
                    LOG(WARNING) << "overflow[" << u << ", " << v << "]: " << dp.x << ", " << dp.y;
                    continue;
                }
                mask_ptr -> setValue(u, v, 1);
                flow_ptr -> setFlowU(u, v, dp.x);
                flow_ptr -> setFlowV(u, v, dp.y);
                flow_ptr -> setValid(u, v, true);
                total_counter ++;
                float gt_x = gt_ptr->getFlowU(u, v);
                float gt_y = gt_ptr->getFlowV(u, v);
                float gt_mag = gt_ptr->getFlowMagnitude(u,v);

                float ferr_abs = std::sqrt((gt_x-dp.x)*(gt_x-dp.x) + (gt_y-dp.y)*(gt_y-dp.y));
                float ferr_per = ferr_abs/gt_mag;
                if(ferr_abs < ABS_THRESH || ferr_per < REL_THRESH){
                    good_counter ++;
                    // LOG(INFO) << "flow truth: (" << gt_x << ", " << gt_y << "), flow measure: (" << dp.x << ", " << dp.y << "), error_abs: " << ferr_abs << ", error_per: " << ferr_per;
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
        cv::Mat prev_img = cv::imread(img_pair["prev_img"]);
        cv::Mat this_img = cv::imread(img_pair["this_img"]);
        // cv::Mat gt_img = cv::imread(img_pair["gt_img"]);

        cv::Mat prev_gray, this_gray;
        cv::cvtColor(prev_img, prev_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(this_img, this_gray, cv::COLOR_BGR2GRAY);

        /*
            image	    Input 8-bit or floating-point 32-bit, single-channel image.
            corners	    Output vector of detected corners.
            maxCorners	Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned. maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned.
            qualityLevel	Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal ) or the Harris function response (see cornerHarris ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
            minDistance	Minimum possible Euclidean distance between the returned corners.
            mask	    Optional region of interest. If the image is not empty (it needs to have the type CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
            blockSize	Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. See cornerEigenValsAndVecs .
            useHarrisDetector	Parameter indicating whether to use a Harris detector (see cornerHarris) or cornerMinEigenVal.
            k	        Free parameter of the Harris detector.
        */
        std::vector<cv::Point2f> p0, p1;
        cv::goodFeaturesToTrack(prev_gray, p0, 0, 0.3, 3, cv::Mat(), 3, false, 0.04);
        LOG(INFO) << "[" << counter << "] init features: " << p0.size();

        std::vector<unsigned char> status;
        std::vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 30, 0.01);
        cv::Size winSize(21, 21);
        /*
            prevImg	first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
            nextImg	second input image or pyramid of the same size and the same type as prevImg.
            prevPts	vector of 2D points for which the flow needs to be found; point coordinates must be single-precision floating-point numbers.
            nextPts	output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image; when OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
            status	output status vector (of unsigned chars); each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, it is set to 0.
            err	    output vector of errors; each element of the vector is set to an error for the corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't found then the error is not defined (use the status parameter to find such cases).
            winSize	size of the search window at each pyramid level.
            maxLevel	0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on; if pyramids are passed to input then algorithm will use as many levels as pyramids have but no more than maxLevel.
            criteria	parameter, specifying the termination criteria of the iterative search algorithm (after the specified maximum number of iterations criteria.maxCount or when the search window moves by less than criteria.epsilon.
            flags	operation flags:
                OPTFLOW_USE_INITIAL_FLOW uses initial estimations, stored in nextPts; if the flag is not set, then prevPts is copied to nextPts and is considered the initial estimate.
                OPTFLOW_LK_GET_MIN_EIGENVALS use minimum eigen values as an error measure (see minEigThreshold description); if the flag is not set, then L1 distance between patches around the original and a moved point, divided by number of pixels in a window, is used as a error measure.
            minEigThreshold	the algorithm calculates the minimum eigen value of a 2x2 normal matrix of optical flow equations (this matrix is called a spatial gradient matrix in [30]), divided by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding feature is filtered out and its flow is not processed, so it allows to remove bad points and get a performance boost.
        */
        const uint64_t start_us = current_micros();
        cv::calcOpticalFlowPyrLK(prev_gray, this_gray, p0, p1, status, err, winSize, 3, criteria);
        average_us_ += (current_micros() - start_us);

        // load groud truth using FlowImage::readFlowField
        std::shared_ptr<FlowImage> gt_ptr = load_flow_gt(img_pair["gt_img"]);
        FLOW_WRAPPER_t flow_wrapper = wrap_flow(this_gray.rows, this_gray.cols, p0, p1, status, gt_ptr);
        std::vector<float> errors = calc_flow_error(gt_ptr, flow_wrapper);

        if(enable_visual){
            cv::Mat mask = cv::Mat::zeros(prev_img.size(), prev_img.type());
            std::vector<cv::Point2f> good_new;
            std::vector<cv::Scalar> colors;
            cv::RNG rng;
            for(int i = 0; i < p0.size(); i++){
                int r = rng.uniform(0, 256);
                int g = rng.uniform(0, 256);
                int b = rng.uniform(0, 256);
                colors.push_back(cv::Scalar(r,g,b));
            }

            for(uint i = 0; i < p0.size(); i++){
                // Select good points
                if(status[i] == 1) {
                    good_new.push_back(p1[i]);
                    // draw the tracks
                    cv::line(mask, p1[i], p0[i], colors[i], 2);
                    cv:: circle(prev_img, p0[i], 5, colors[i], -1);
                }
            }
            LOG(INFO) << "[" << counter << "] tracked features: " << good_new.size();

            cv::Mat vis_img;
            cv::add(prev_img, mask, vis_img);
            cv::namedWindow("kitti", cv::WINDOW_NORMAL);
            cv::imshow("kitti", vis_img);
            cv::waitKey(1000);
            cv::imwrite(gen_output_path(img_pair["this_img"]), vis_img);
        }

        return errors;
    }

};

}
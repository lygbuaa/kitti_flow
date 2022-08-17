#pragma once

#include <numeric>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "KittiFlowBase.h"
#include <vpi/algo/OpticalFlowDense.h>
#include <vpi/Image.h>
#include <vpi/Array.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/Version.h>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/algo/ConvertImageFormat.h>

DECLARE_string(output_img_path);

namespace kittflow
{

class VpiDense : public KittiFlowBase
{
public:
    /* dense opitcal flow only support nvenc backend */
    static constexpr VPIBackend backend_ = VPI_BACKEND_NVENC;
    static constexpr VPIOpticalFlowQuality quality_ = VPI_OPTICAL_FLOW_QUALITY_HIGH;

    #define CHECK_STATUS(STMT)                                      \
        do                                                          \
        {                                                           \
            VPIStatus status__ = (STMT);                            \
            if (status__ != VPI_SUCCESS)                            \
            {                                                       \
                char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];         \
                vpiGetLastStatusMessage(buffer, sizeof(buffer));    \
                std::ostringstream ss;                              \
                ss << vpiStatusGetName(status__) << ": " << buffer; \
                throw std::runtime_error(ss.str());                 \
            }                                                       \
        } while (0);

private:
    uint64_t average_us_ = 0;
    VPIStream stream_         = NULL;
    VPIPayload payload_       = NULL;
    VPIImage imgPrevFramePL_  = NULL;
    VPIImage imgPrevFrameTmp_ = NULL;
    VPIImage imgPrevFrameBL_  = NULL;

    VPIImage imgCurFramePL_   = NULL;
    VPIImage imgCurFrameTmp_  = NULL;
    VPIImage imgCurFrameBL_   = NULL;

    VPIImage imgMotionVecBL_  = NULL;
    int mvWidth_  = 0;
    int mvHeight_ = 0;

public:
    VpiDense(const std::string& img_path, const std::string& gt_path)
    : KittiFlowBase(img_path, gt_path)
    {
        init_vpi();
        LOG(INFO) << "vpi version: " << vpiGetVersion();
    }

    ~VpiDense(){
        release_vpi();
    }

    void init_vpi(){
        CHECK_STATUS(vpiStreamCreate(backend_ | VPI_BACKEND_CUDA | VPI_BACKEND_VIC, &stream_));
        // The Dense Optical Flow on NVENC backend expects input to be in block-linear format.
        // Since Convert Image Format algorithm doesn't currently support direct BGR
        // pitch-linear (from OpenCV) to NV12 block-linear conversion, it must be done in two
        // passes, first from BGR/PL to NV12/PL using CUDA, then from NV12/PL to NV12/BL using VIC.
        // The temporary image buffer below will store the intermediate NV12/PL representation.
        // Define the image formats we'll use throughout this sample.

        CHECK_STATUS(vpiImageCreate(FLAGS_kitti_img_width, FLAGS_kitti_img_height, VPI_IMAGE_FORMAT_NV12_ER, 0, &imgPrevFrameTmp_));
        CHECK_STATUS(vpiImageCreate(FLAGS_kitti_img_width, FLAGS_kitti_img_height, VPI_IMAGE_FORMAT_NV12_ER, 0, &imgCurFrameTmp_));

        // Now create the final block-linear buffer that'll be used as input to the
        // algorithm.
        CHECK_STATUS(vpiImageCreate(FLAGS_kitti_img_width, FLAGS_kitti_img_height, VPI_IMAGE_FORMAT_NV12_ER_BL, 0, &imgPrevFrameBL_));
        CHECK_STATUS(vpiImageCreate(FLAGS_kitti_img_width, FLAGS_kitti_img_height, VPI_IMAGE_FORMAT_NV12_ER_BL, 0, &imgCurFrameBL_));

        // Motion vector image width and height, On NVENC, dimensions must be 1/4 of curImg, align to be multiple of 4
        mvWidth_  = (FLAGS_kitti_img_width + 3) / 4;
        mvHeight_ = (FLAGS_kitti_img_height + 3) / 4;

        // Create the output motion vector buffer
        CHECK_STATUS(vpiImageCreate(mvWidth_, mvHeight_, VPI_IMAGE_FORMAT_2S16_BL, 0, &imgMotionVecBL_));
        LOG(INFO) << "OpticalFlowDense output size: " << mvWidth_ << ", " << mvHeight_;

        // Create Dense Optical Flow payload to be executed on the given backend
        CHECK_STATUS(vpiCreateOpticalFlowDense(backend_, FLAGS_kitti_img_width, FLAGS_kitti_img_height, VPI_IMAGE_FORMAT_NV12_ER_BL, quality_, &payload_));
        LOG(INFO) << "vpiCreate OpticalFlowDense on backend: " << (int)backend_;
    }

    void release_vpi(){
        vpiStreamDestroy(stream_);
        vpiPayloadDestroy(payload_);
    }

    std::string gen_output_path(const std::string input_img_path){
        const char* file_name = basename(const_cast<char*>(input_img_path.c_str()));
        std::string output_path = FLAGS_output_img_path + "/" + file_name + ".vpi_dense.png";
        return output_path;
    }

    /* flow = cv::Mat(CV_32FC2); */
    FLOW_WRAPPER_t wrap_flow(const int img_h, const int img_w, const cv::Mat& flow, std::shared_ptr<FlowImage> gt_ptr){
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
            /* only test images with 1242*375 resolution */
            if(counter > 10) break;

            std::vector<float> errors = run_once(counter, img_pair, enable_visual);
            // print_error_report(errors);

            if(errors.size() < 12){
                continue;
            }
            for(int i=0; i<errors.size(); ++i){
                errors_acc[i] += errors[i];
            }

            // if(counter > 10) break;
        }
        average_us_ /= counter;
        LOG(INFO) << "all " << counter << " tests done, average_us: " << average_us_;
        print_error_report(errors_acc);
    }

    std::vector<float> run_once(unsigned int counter, std::map<std::string, std::string>& img_pair, bool enable_visual=false, bool enable_calc=true){
        cv::Mat prev_img = cv::imread(img_pair["prev_img"]);
        cv::Mat this_img = cv::imread(img_pair["this_img"]);
        // cv::Mat gt_img = cv::imread(img_pair["gt_img"]);

        // cv::Mat prev_gray, this_gray;
        // cv::cvtColor(prev_img, prev_gray, cv::COLOR_BGR2GRAY);
        // cv::cvtColor(this_img, this_gray, cv::COLOR_BGR2GRAY);

        /* dense optical flow support bgr image */
        vpiImageCreateWrapperOpenCVMat(prev_img, 0, &imgPrevFramePL_);
        vpiImageCreateWrapperOpenCVMat(this_img, 0, &imgCurFramePL_);

        const uint64_t start_us = current_micros();
        // Convert frame to NV12_BL format
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, imgPrevFramePL_, imgPrevFrameTmp_, nullptr));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC, imgPrevFrameTmp_, imgPrevFrameBL_, nullptr));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, imgCurFramePL_, imgCurFrameTmp_, nullptr));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC, imgCurFrameTmp_, imgCurFrameBL_, nullptr));
        CHECK_STATUS(vpiSubmitOpticalFlowDense(stream_, backend_, payload_, imgPrevFrameBL_, imgCurFrameBL_, imgMotionVecBL_));

        // Wait for processing to finish.
        CHECK_STATUS(vpiStreamSync(stream_));
        average_us_ += (current_micros() - start_us);

        cv::Mat flow(this_img.size(), CV_32FC2);
        std::vector<float> errors(12, 0.0f);

        if(enable_calc){
            VPIImageData mvData;
            CHECK_STATUS(vpiImageLockData(imgMotionVecBL_, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &mvData));

            // Create a cv::Mat that points to the input image data
            cv::Mat mvImage;
            CHECK_STATUS(vpiImageDataExportOpenCVMat(mvData, &mvImage));

            // Convert S10.5 format to float
            cv::Mat flow_small(mvImage.size(), CV_32FC2);
            mvImage.convertTo(flow_small, CV_32F, 1.0f / (1 << 5));

            // Image not needed anymore, we can unlock it.
            CHECK_STATUS(vpiImageUnlock(imgMotionVecBL_));
            cv::resize(flow_small, flow, flow.size());

            std::shared_ptr<FlowImage> gt_ptr = load_flow_gt(img_pair["gt_img"]);
            FLOW_WRAPPER_t flow_wrapper = wrap_flow(this_img.rows, this_img.cols, flow, gt_ptr);
            errors = calc_flow_error(gt_ptr, flow_wrapper);
        }

        if(enable_calc && enable_visual){
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
            cv::waitKey(1000);
            cv::imwrite(gen_output_path(img_pair["this_img"]), bgr);
        }

        return errors;
    }

};

}

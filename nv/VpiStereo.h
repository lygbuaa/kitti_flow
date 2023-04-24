/*
* refer to /opt/nvidia/vpi2/samples/02-stereo_disparity
*/

#pragma once

#include <numeric>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "KittiStereoBase.h"
#include <vpi/Image.h>
#include <vpi/Array.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/Version.h>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

DECLARE_string(output_img_path);
// DEFINE_uint32(vpi_img_width, 1248, "input image width. should be 16*N");
// DEFINE_uint32(vpi_img_height, 376, "input image height. should be 4*N");
DECLARE_uint32(kitti_img_width);
DECLARE_uint32(kitti_img_height);

namespace kittflow
{

#define USING_BACKEND 0

class VpiStereo : public KittiStereoBase
{
public:
    /* dense opitcal flow only support nvenc backend
    typedef enum
    {
        VPI_BACKEND_CPU     = (1ULL << 0),
        VPI_BACKEND_CUDA    = (1ULL << 1),
        VPI_BACKEND_PVA     = (1ULL << 2),
        VPI_BACKEND_VIC     = (1ULL << 3),
        VPI_BACKEND_NVENC   = (1ULL << 4),
        VPI_BACKEND_OFA     = (1ULL << 5),
        VPI_BACKEND_INVALID = (1ULL << 15)
    } VPIBackend;
    */
#if USING_BACKEND == 0
    static constexpr uint64_t backend_ = (VPI_BACKEND_OFA); // | VPI_BACKEND_PVA | VPI_BACKEND_VIC);
#elif USING_BACKEND == 1
    static constexpr uint64_t backend_ = VPI_BACKEND_CUDA;
#endif
    static constexpr int conf_threshold_ = 100;

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
    VPIPayload stereo_      = NULL;
    VPIImage inLeft_        = NULL;
    VPIImage inRight_       = NULL;
    VPIImage tmpLeft_        = NULL;
    VPIImage tmpRight_        = NULL;
    VPIImage stereoLeft_    = NULL;
    VPIImage stereoRight_   = NULL;
    VPIImage disparity_     = NULL;
    VPIImage confidenceMap_ = NULL;

    VPIConvertImageFormatParams convParams_;

    /*
      maxDisparity: On CUDA backends, maxDisparity must be >= 0 and <= 256.
      windowSize: On CUDA backend this is ignored. A 9x7 window is used instead.
     */
    VPIStereoDisparityEstimatorCreationParams stereoParams_;

#if USING_BACKEND == 0
    VPIImageFormat stereoFormat_    = VPI_IMAGE_FORMAT_Y16_ER_BL;
    VPIImageFormat disparityFormat_ = VPI_IMAGE_FORMAT_S16_BL;
#elif USING_BACKEND == 1
    VPIImageFormat stereoFormat_    = VPI_IMAGE_FORMAT_Y16_ER;
    VPIImageFormat disparityFormat_ = VPI_IMAGE_FORMAT_S16;
#endif
    VPIImageFormat tmpFormat_    = VPI_IMAGE_FORMAT_Y16_ER;

public:
    VpiStereo(const std::string& left_img_path, const std::string& right_img_path, const std::string& gt_path)
    : KittiStereoBase(left_img_path, right_img_path, gt_path)
    {
        init_vpi();
        LOG(INFO) << "vpi version: " << vpiGetVersion() << ", backend: " << backend_;
    }

    ~VpiStereo(){
        release_vpi();
    }

    void init_vpi(){
        int inputWidth = FLAGS_kitti_img_width;
        int inputHeight = FLAGS_kitti_img_height;
        int stereoWidth  = FLAGS_kitti_img_width;
        int stereoHeight = FLAGS_kitti_img_height;
        int outputWidth  = FLAGS_kitti_img_width;
        int outputHeight = FLAGS_kitti_img_height;

        // CHECK_STATUS(vpiStreamCreate(backend_, &stream_));
        /* 0 open all backends */
        CHECK_STATUS(vpiStreamCreate(0, &stream_));
        // Format conversion parameters needed for input pre-processing
        CHECK_STATUS(vpiInitConvertImageFormatParams(&convParams_));
        // Set algorithm parameters to be used. Only values what differs from defaults will be overwritten.
        CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&stereoParams_));
        //stereoParams_.maxDisparity = ((FLAGS_kitti_img_width/8) + 15) & -16;;
        /* 
        * On OFA or OFA+PVA+VIC backend, maxDisparity must be 128 or 256. 
        * 256 lead to VPI_ERROR_INTERNAL: maximum bound exceeded during bounded integer assignment.
        */
        stereoParams_.maxDisparity = 256; 
        LOG(INFO) << "maxDisparity: " << stereoParams_.maxDisparity;

        // Create the payload for Stereo Disparity algorithm.
        // Payload is created before the image objects so that non-supported backends can be trapped with an error.
        CHECK_STATUS(vpiCreateStereoDisparityEstimator(backend_, stereoWidth, stereoHeight, stereoFormat_, &stereoParams_, &stereo_));
        // Create the image where the disparity map will be stored.
        CHECK_STATUS(vpiImageCreate(outputWidth, outputHeight, disparityFormat_, 0, &disparity_));
        // Create the input stereo images
        CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, tmpFormat_, 0, &tmpLeft_));
        CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, tmpFormat_, 0, &tmpRight_));
        CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, stereoFormat_, 0, &stereoLeft_));
        CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, stereoFormat_, 0, &stereoRight_));

#if USING_BACKEND == 0
        confidenceMap_ = NULL;
#else
        CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, VPI_IMAGE_FORMAT_U16, 0, &confidenceMap_));
#endif

    }

    void release_vpi(){
      // Destroying stream first makes sure that all work submitted to
      // it is finished.
      vpiStreamDestroy(stream_);

      // Only then we can destroy the other objects, as we're sure they
      // aren't being used anymore.

      vpiImageDestroy(inLeft_);
      vpiImageDestroy(inRight_);
      vpiImageDestroy(tmpLeft_);
      vpiImageDestroy(tmpRight_);
      vpiImageDestroy(stereoLeft_);
      vpiImageDestroy(stereoRight_);
      vpiImageDestroy(confidenceMap_);
      vpiImageDestroy(disparity_);
      vpiPayloadDestroy(stereo_);
    }

    std::string gen_output_path(const std::string input_img_path){
        const char* file_name = basename(const_cast<char*>(input_img_path.c_str()));
        std::string output_path = FLAGS_output_img_path + "/" + file_name + ".vpi_stereo.png";
        return output_path;
    }

    STEREO_WRAPPER_t wrap_stereo(const int img_h, const int img_w, const cv::Mat& floatDisp, const cv::Mat& u16Conf, std::shared_ptr<DisparityImage> gt_ptr){
        std::shared_ptr<DisparityImage> disp_ptr(new DisparityImage(img_w, img_h));
        std::shared_ptr<IntegerImage> mask_ptr(new IntegerImage(img_w, img_h));

        static uint32_t total_counter = 0;
        static uint32_t good_counter = 0;
        static uint32_t low_conf_counter = 0;
        float val_max = 0.0f;
        float val_min = stereoParams_.maxDisparity;
        float gt_max = 0.0f;
        float gt_min = stereoParams_.maxDisparity;
        int conf_min = 65535;
        int conf_max = 0;
        for(int y=0; y<img_h; ++y){
            for(int x=0; x<img_w; ++x){
                total_counter ++;
                float val = floatDisp.at<float>(y, x);
                unsigned short conf = u16Conf.at<unsigned short>(y, x);
                disp_ptr -> setDisp(x, y, val);

                if(conf > conf_max){
                  conf_max = conf;
                }else if(conf < conf_min){
                  conf_min = conf;
                }

                if(val<=0 || val>stereoParams_.maxDisparity || conf<conf_threshold_){
                  disp_ptr -> setInvalid(x, y);
                  mask_ptr -> setValue(x, y, 0);
                  ++low_conf_counter;
                }else{
                  mask_ptr -> setValue(x, y, 1);
                  if(val > val_max){
                    val_max = val;
                  }else if(val < val_min){
                    val_min = val;
                  }
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
        LOG(INFO) << "gt_max: " << gt_max << ", gt_min: " << gt_min << ", val_max: " << val_max << ", val_min: " << val_min << ", conf_max: " << conf_max << ", conf_min: " << conf_min;
        LOG(INFO) << "total_counter: " << total_counter << ", good_counter: " << good_counter << ", low_conf_counter: " << low_conf_counter;
        return STEREO_WRAPPER_t(disp_ptr, mask_ptr);
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
            if(counter > 155) break;

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
        cv::Mat left_img = cv::imread(img_pair["left_img"]);
        cv::Mat right_img = cv::imread(img_pair["right_img"]);

        // We now wrap the loaded images into a VPIImage object to be used by VPI.
        // VPI won't make a copy of it, so the original image must be in scope at all times.
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(left_img, 0, &inLeft_));
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(right_img, 0, &inRight_));

        const uint64_t start_us = current_micros();
        // Convert opencv input to grayscale format using CUDA
#if USING_BACKEND == 0
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, inLeft_, tmpLeft_, &convParams_));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, inRight_, tmpRight_, &convParams_));

        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC, tmpLeft_, stereoLeft_, &convParams_));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC, tmpRight_, stereoRight_, &convParams_));
#elif USING_BACKEND == 1
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, inLeft_, stereoLeft_, &convParams_));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, inRight_, stereoRight_, &convParams_));
#endif
        // Submit it with the input and output images
        CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stream_, backend_, stereo_, stereoLeft_, stereoRight_, disparity_, confidenceMap_, NULL));
        // Wait for processing to finish.
        CHECK_STATUS(vpiStreamSync(stream_));

        average_us_ += (current_micros() - start_us);
        std::vector<float> errors(12, 0.0f);
        // Make an OpenCV matrix out of this image
        cv::Mat cvDisparity(left_img.size(), CV_32FC1);
        cv::Mat cvConfidence(left_img.size(), CV_16UC1, 65535);

        if(enable_calc){
            std::shared_ptr<DisparityImage> gt_ptr = load_stereo_gt(img_pair["gt_img"]);
            // Output pre-processing and saving to disk
            // Lock output to retrieve its data on cpu memory
            VPIImageData data;
            CHECK_STATUS(vpiImageLockData(disparity_, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));
            CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cvDisparity));
            // Scale result and write it to disk. Disparities are in Q10.5 format,
            // so to map it to float, it gets divided by 32.
            cvDisparity.convertTo(cvDisparity, CV_32FC1, 1.0/32.0f, 0);
            // Done handling output, don't forget to unlock it.
            CHECK_STATUS(vpiImageUnlock(disparity_));

            if(confidenceMap_){
                CHECK_STATUS(vpiImageLockData(confidenceMap_, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));
                CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cvConfidence));

                // Confidence map varies from 0 to 65535
                CHECK_STATUS(vpiImageUnlock(confidenceMap_));
            }

            STEREO_WRAPPER_t stereo_wrapper = wrap_stereo(left_img.rows, left_img.cols, cvDisparity, cvConfidence, gt_ptr);
            errors = calc_stereo_error(gt_ptr, stereo_wrapper);
        }

        if(enable_calc && enable_visual){
            cv::Mat disp8(left_img.size(), CV_8UC1);
            //the resulting disparity range, from 0 to stereo.maxDisparity gets mapped to 0-255 for proper output.
            cvDisparity.convertTo(disp8, CV_8UC1, 255.0 / (stereoParams_.maxDisparity), 0);
            cv::namedWindow("kitti", cv::WINDOW_NORMAL);
            cv::imshow("kitti", disp8);
            cv::waitKey(1000);
            cv::imwrite(gen_output_path(img_pair["left_img"]), disp8);
        }

        return errors;
    }

};

}

#pragma once

#include <numeric>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "VideoFlowBase.h"
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

class VpiDenseVideo : public VideoFlowBase
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
    static constexpr VPIBackend backend_ = VPI_BACKEND_NVENC;

    /* 
        typedef enum
        {
            VPI_OPTICAL_FLOW_QUALITY_LOW,
            VPI_OPTICAL_FLOW_QUALITY_MEDIUM,
            VPI_OPTICAL_FLOW_QUALITY_HIGH
        } VPIOpticalFlowQuality;
     */
    static constexpr VPIOpticalFlowQuality quality_ = VPI_OPTICAL_FLOW_QUALITY_LOW;

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
    cv::Mat mvOutputImage_;
    int32_t mvWidth_  = 0;
    int32_t mvHeight_ = 0;
    cv::VideoWriter video_writer_;
    std::string output_video_path_;

public:
    VpiDenseVideo(const std::string& video_path, bool enable_visual=false)
    : VideoFlowBase(video_path, enable_visual)
    {
        init_vpi();
        LOG(INFO) << "init vpi version: " << vpiGetVersion();
        if(enable_visual){
            init_cv_writer();
            LOG(INFO) << "output video path: " << output_video_path_;
        }
    }

    ~VpiDenseVideo(){
        release_vpi();
        if(!video_writer_.isOpened()){
            video_writer_.release();
        }
    }

    std::string gen_output_path(const std::string input_video_path){
        const char* file_name = basename(const_cast<char*>(input_video_path.c_str()));
        std::string output_path = FLAGS_output_img_path + "/" + file_name + ".vpi_dense.mp4";
        return output_path;
    }

    void init_cv_writer(){
        output_video_path_ = gen_output_path(input_video_path_);
        video_writer_.open(output_video_path_, cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), fps_, cv::Size(mvWidth_, mvHeight_));
        if(!video_writer_.isOpened()){
            LOG(FATAL) << "create video failed: " << output_video_path_;
        }
    }

    void init_vpi(){
        CHECK_STATUS(vpiStreamCreate(backend_ | VPI_BACKEND_CUDA | VPI_BACKEND_VIC, &stream_));
        // The Dense Optical Flow on NVENC backend expects input to be in block-linear format.
        // Since Convert Image Format algorithm doesn't currently support direct BGR
        // pitch-linear (from OpenCV) to NV12 block-linear conversion, it must be done in two
        // passes, first from BGR/PL to NV12/PL using CUDA, then from NV12/PL to NV12/BL using VIC.
        // The temporary image buffer below will store the intermediate NV12/PL representation.
        // Define the image formats we'll use throughout this sample.

        CHECK_STATUS(vpiImageCreate(video_width_, video_height_, VPI_IMAGE_FORMAT_NV12_ER, 0, &imgPrevFrameTmp_));
        CHECK_STATUS(vpiImageCreate(video_width_, video_height_, VPI_IMAGE_FORMAT_NV12_ER, 0, &imgCurFrameTmp_));

        // Now create the final block-linear buffer that'll be used as input to the
        // algorithm.
        CHECK_STATUS(vpiImageCreate(video_width_, video_height_, VPI_IMAGE_FORMAT_NV12_ER_BL, 0, &imgPrevFrameBL_));
        CHECK_STATUS(vpiImageCreate(video_width_, video_height_, VPI_IMAGE_FORMAT_NV12_ER_BL, 0, &imgCurFrameBL_));

        // Motion vector image width and height, On NVENC, dimensions must be 1/4 of curImg, align to be multiple of 4
        mvWidth_  = (video_width_ + 3) / 4;
        mvHeight_ = (video_height_ + 3) / 4;
        // mvWidth_  = video_width_;
        // mvHeight_ = video_height_;

        // Create the output motion vector buffer
        CHECK_STATUS(vpiImageCreate(mvWidth_, mvHeight_, VPI_IMAGE_FORMAT_2S16_BL, 0, &imgMotionVecBL_));
        LOG(INFO) << "OpticalFlowDense output size: " << mvWidth_ << ", " << mvHeight_;

        // Create Dense Optical Flow payload to be executed on the given backend
        CHECK_STATUS(vpiCreateOpticalFlowDense(backend_, video_width_, video_height_, VPI_IMAGE_FORMAT_NV12_ER_BL, quality_, &payload_));
        LOG(INFO) << "vpiCreate OpticalFlowDense on backend: " << (int)backend_;
    }

    void release_vpi(){
        vpiStreamDestroy(stream_);
        vpiPayloadDestroy(payload_);
        vpiImageDestroy(imgPrevFramePL_);
        vpiImageDestroy(imgPrevFrameTmp_);
        vpiImageDestroy(imgPrevFrameBL_);
        vpiImageDestroy(imgCurFramePL_);
        vpiImageDestroy(imgCurFrameTmp_);
        vpiImageDestroy(imgCurFrameBL_);
        vpiImageDestroy(imgMotionVecBL_);
    }

    virtual bool init_stream(cv::Mat& cvPrevFrame) override {
        // LOG(INFO) << "init_stream from derived class!";
        // return true;
        vpiImageCreateWrapperOpenCVMat(cvPrevFrame, 0, &imgPrevFramePL_);
        vpiImageCreateWrapperOpenCVMat(cvPrevFrame, 0, &imgCurFramePL_);
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, imgPrevFramePL_, imgPrevFrameTmp_, nullptr));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC, imgPrevFrameTmp_, imgPrevFrameBL_, nullptr));
        return true;
    }

    virtual bool run_once(cv::Mat& cvCurFrame) override {
        // LOG(INFO) << "run_once from derived class!";
        // return true;
        CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgCurFramePL_, cvCurFrame));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, imgCurFramePL_, imgCurFrameTmp_, nullptr));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC, imgCurFrameTmp_, imgCurFrameBL_, nullptr));
        CHECK_STATUS(vpiSubmitOpticalFlowDense(stream_, backend_, payload_, imgPrevFrameBL_, imgCurFrameBL_, imgMotionVecBL_));
        CHECK_STATUS(vpiStreamSync(stream_));
        std::swap(imgPrevFramePL_, imgCurFramePL_);
        std::swap(imgPrevFrameBL_, imgCurFrameBL_);

        if(enable_visual_ && video_writer_.isOpened()){
            ProcessMotionVector(imgMotionVecBL_, mvOutputImage_);
            video_writer_ << mvOutputImage_;
            // DLOG(INFO) << "visual image saved to video";
        }
        return true;
    }

    static void ProcessMotionVector(VPIImage mvImg, cv::Mat &outputImage)
    {
        // Lock the input image to access it from CPU
        VPIImageData mvData;
        CHECK_STATUS(vpiImageLockData(mvImg, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &mvData));

        // Create a cv::Mat that points to the input image data
        cv::Mat mvImage;
        CHECK_STATUS(vpiImageDataExportOpenCVMat(mvData, &mvImage));

        // Convert S10.5 format to float
        cv::Mat flow(mvImage.size(), CV_32FC2);
        mvImage.convertTo(flow, CV_32F, 1.0f / (1 << 5));

        // Image not needed anymore, we can unlock it.
        CHECK_STATUS(vpiImageUnlock(mvImg));

        // Create an image where the motion vector angle is
        // mapped to a color hue, and intensity is proportional
        // to vector's magnitude.
        cv::Mat magnitude, angle;
        {
            cv::Mat flowChannels[2];
            split(flow, flowChannels);
            cv::cartToPolar(flowChannels[0], flowChannels[1], magnitude, angle, true);
        }

        float clip = 5;
        cv::threshold(magnitude, magnitude, clip, clip, cv::THRESH_TRUNC);

        // build hsv image
        cv::Mat _hsv[3], hsv, bgr;
        _hsv[0] = angle;
        _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magnitude / clip; // intensity must vary from 0 to 1
        merge(_hsv, 3, hsv);

        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        bgr.convertTo(outputImage, CV_8U, 255.0);
    }

};

}

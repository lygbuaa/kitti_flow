#pragma once

#include <numeric>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "KittiFlowBase.h"
#include <vpi/algo/OpticalFlowPyrLK.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/Image.h>
#include <vpi/Array.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/Version.h>
#include <vpi/OpenCVInterop.hpp>

DECLARE_string(output_img_path);

namespace kittflow
{


class VpiLK : public KittiFlowBase
{
public:
    // Max number of corners detected by harris corner algo
    static constexpr int MAX_HARRIS_CORNERS = 8192;

    // Max number of keypoints to be tracked
    static constexpr int MAX_KEYPOINTS = 8192;

    static constexpr VPIBackend backend_ = VPI_BACKEND_CUDA;

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
    VPIStream stream_ = nullptr;
    VPIImage pre_image_ = nullptr;
    VPIImage this_image_ = nullptr;
    VPIPyramid pyrPrevFrame_ = nullptr;
    VPIPyramid pyrCurFrame_ = nullptr;
    VPIPayload optflow_ = nullptr;
    VPIPayload harris_  = nullptr;
    VPIOpticalFlowPyrLKParams lkParams_;
    VPIHarrisCornerDetectorParams harrisParams_;
    VPIArray prevFeatures_ = nullptr;
    VPIArray curFeatures_ = nullptr;
    VPIArray status_ = nullptr;
    VPIArray scores_    = nullptr;

public:
    VpiLK(const std::string& img_path, const std::string& gt_path)
    : KittiFlowBase(img_path, gt_path)
    {
        init_vpi();
        LOG(INFO) << "vpi version: " << vpiGetVersion();
    }

    ~VpiLK(){
        release_vpi();
    }

    void init_vpi(){
        CHECK_STATUS(vpiStreamCreate(0, &stream_));
        // Create the image pyramids used by the algorithm
        int pyrLevel = 3; /* #define 	VPI_MAX_PYRAMID_LEVEL_COUNT   (10) */
        float pyrScale = 0.5f;
        CHECK_STATUS(vpiPyramidCreate(FLAGS_kitti_img_width, FLAGS_kitti_img_height, VPI_IMAGE_FORMAT_U8, pyrLevel, pyrScale, 0, &pyrPrevFrame_));
        CHECK_STATUS(vpiPyramidCreate(FLAGS_kitti_img_width, FLAGS_kitti_img_height, VPI_IMAGE_FORMAT_U8, pyrLevel, pyrScale, 0, &pyrCurFrame_));

        // Create Optical Flow payload
        CHECK_STATUS(vpiCreateOpticalFlowPyrLK(backend_, FLAGS_kitti_img_width, FLAGS_kitti_img_height, VPI_IMAGE_FORMAT_U8, pyrLevel, pyrScale, &optflow_));

        /*
        Defaults:
            useInitialFlow = 0 //previous frame keypoints are copied to current frame keypoints array and is considered the initial estimate.
            termination = VPI_TERMINATION_CRITERIA_ITERATIONS | VPI_TERMINATION_CRITERIA_EPSILON
            epsilonType = VPI_LK_ERROR_L1
            epsilon = 0
            windowDimension = 15 //Must be >= 6 and <= 32.
            numIterations = 6 //Must be >= 1 and <= 32.
        here we set it same with cv::calcOpticalFlowPyrLK
        */
        lkParams_.numIterations = 30;
        lkParams_.windowDimension = 21;
        CHECK_STATUS(vpiInitOpticalFlowPyrLKParams(&lkParams_));
        CHECK_STATUS(vpiInitHarrisCornerDetectorParams(&harrisParams_));
        harrisParams_.strengthThresh = 0;
        harrisParams_.sensitivity    = 0.01;

        CHECK_STATUS(vpiCreateHarrisCornerDetector(backend_, FLAGS_kitti_img_width, FLAGS_kitti_img_height, &harris_));
        CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &prevFeatures_));
        CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &curFeatures_));
        // CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U8, 0, &status_));
        CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U32, 0, &scores_));
    }

    void release_vpi(){
        vpiStreamDestroy(stream_);
        vpiPayloadDestroy(harris_);
        vpiPayloadDestroy(optflow_);
        vpiImageDestroy(pre_image_);
        vpiImageDestroy(this_image_);
        vpiPyramidDestroy(pyrPrevFrame_);
        vpiPyramidDestroy(pyrCurFrame_);
        vpiArrayDestroy(prevFeatures_);
        vpiArrayDestroy(curFeatures_);
        // vpiArrayDestroy(status_);
        vpiArrayDestroy(scores_);
    }

    std::string gen_output_path(const std::string input_img_path){
        const char* file_name = basename(const_cast<char*>(input_img_path.c_str()));
        std::string output_path = FLAGS_output_img_path + "/" + file_name + ".vpi_lk.png";
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
                break;
            }
            ++ counter;
            /* only test images with 1242*375 resolution */
            if(counter > 154) break;

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

    std::vector<float> run_once(unsigned int counter, std::map<std::string, std::string>& img_pair, bool enable_visual=false){
        cv::Mat prev_img = cv::imread(img_pair["prev_img"]);
        cv::Mat this_img = cv::imread(img_pair["this_img"]);
        cv::Mat prev_gray, this_gray;
        cv::cvtColor(prev_img, prev_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(this_img, this_gray, cv::COLOR_BGR2GRAY);
        vpiImageCreateWrapperOpenCVMat(prev_gray, 0, &pre_image_);
        vpiImageCreateWrapperOpenCVMat(this_gray, 0, &this_image_);
        /* status should be cleared every cycle, since kitti images are discontinuous,
           if input is continuous video, global status will be ok.
        */
        VPIArray status = nullptr;
        CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U8, 0, &status));

        //Gather feature points from first frame using Harris Corners on CPU.
        {
            // Convert input to grayscale to conform with harris corner detector restrictions
            CHECK_STATUS(vpiSubmitHarrisCornerDetector(stream_, backend_, harris_, pre_image_, prevFeatures_, scores_, &harrisParams_));
            CHECK_STATUS(vpiStreamSync(stream_));
            SortKeypoints(prevFeatures_, scores_, MAX_KEYPOINTS);
        }

        const uint64_t start_us = current_micros();
        // Generate a pyramid out of it
        CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(stream_, backend_, pre_image_, pyrPrevFrame_, VPI_BORDER_CLAMP));
        CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(stream_, backend_, this_image_, pyrCurFrame_, VPI_BORDER_CLAMP));
        // Estimate the features' position in current frame given their position in previous frame
        CHECK_STATUS(vpiSubmitOpticalFlowPyrLK(stream_, 0, optflow_, pyrPrevFrame_, pyrCurFrame_, prevFeatures_, curFeatures_, status, &lkParams_));
        // Wait for processing to finish.
        CHECK_STATUS(vpiStreamSync(stream_));
        average_us_ += (current_micros() - start_us);

        // load groud truth using FlowImage::readFlowField
        std::shared_ptr<FlowImage> gt_ptr = load_flow_gt(img_pair["gt_img"]);
        std::vector<cv::Point2f> p0, p1;
        std::vector<unsigned char> vstatus;
        retrieve_results(prevFeatures_, curFeatures_, status, p0, p1, vstatus);
        vpiArrayDestroy(status);

        // return std::vector<float>(12, 0.0f);
        FLOW_WRAPPER_t flow_wrapper = wrap_flow(this_gray.rows, this_gray.cols, p0, p1, vstatus, gt_ptr);
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
                if(vstatus[i] == 1) {
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

    static int retrieve_results(VPIArray prevFeatures, VPIArray curFeatures, VPIArray status, std::vector<cv::Point2f>& vp0, std::vector<cv::Point2f>& vp1, std::vector<unsigned char>& vstatus)
    {
        // Now that optical flow is completed, there are usually two approaches to take:
        // 1. Add new feature points from current frame using a feature detector such as
        //    \ref algo_harris_corners "Harris Corner Detector"
        // 2. Keep using the points that are being tracked.
        //
        // The sample app uses the valid feature point and continue to do the tracking.

        // Lock the input and output arrays to draw the tracks to the output mask.
        VPIArrayData curFeaturesData, statusData;
        CHECK_STATUS(vpiArrayLockData(curFeatures, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &curFeaturesData));
        CHECK_STATUS(vpiArrayLockData(status, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &statusData));

        const VPIArrayBufferAOS &aosCurFeatures = curFeaturesData.buffer.aos;
        const VPIArrayBufferAOS &aosStatus      = statusData.buffer.aos;

        const VPIKeypointF32 *pCurFeatures = (VPIKeypointF32 *)aosCurFeatures.data;
        const uint8_t *pStatus          = (uint8_t *)aosStatus.data;

        const VPIKeypointF32 *pPrevFeatures;
        if (prevFeatures)
        {
            VPIArrayData prevFeaturesData;
            CHECK_STATUS(vpiArrayLockData(prevFeatures, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &prevFeaturesData));
            pPrevFeatures = (VPIKeypointF32 *)prevFeaturesData.buffer.aos.data;
        }
        else
        {
            pPrevFeatures = nullptr;
        }

        int numTrackedKeypoints = 0;
        int totKeypoints        = *curFeaturesData.buffer.aos.sizePointer;
        LOG(INFO) << "totKeypoints: " << totKeypoints;

        for (int i = 0; i < totKeypoints; i++)
        {
            if (pPrevFeatures != nullptr)
            {
                cv::Point2f prevPoint{pPrevFeatures[i].x, pPrevFeatures[i].y};
                vp0.emplace_back(prevPoint);
            }
            cv::Point2f curPoint{pCurFeatures[i].x, pCurFeatures[i].y};
            vp1.emplace_back(curPoint);
            // keypoint is being tracked?
            if (pStatus[i] == 0)
            {
                numTrackedKeypoints++;
                vstatus.emplace_back(1);
            }else{
                vstatus.emplace_back(0);
            }
        }
        LOG(INFO) << "numTrackedKeypoints: " << numTrackedKeypoints;

        // We're finished working with the arrays.
        if (prevFeatures)
        {
            CHECK_STATUS(vpiArrayUnlock(prevFeatures));
        }
        CHECK_STATUS(vpiArrayUnlock(curFeatures));
        CHECK_STATUS(vpiArrayUnlock(status));

        return numTrackedKeypoints;
    }

    // Sort keypoints by decreasing score, and retain only the first 'max'
    static void SortKeypoints(VPIArray keypoints, VPIArray scores, int max)
    {
        VPIArrayData ptsData, scoresData;
        CHECK_STATUS(vpiArrayLockData(keypoints, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &ptsData));
        CHECK_STATUS(vpiArrayLockData(scores, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &scoresData));

        VPIArrayBufferAOS &aosKeypoints = ptsData.buffer.aos;
        VPIArrayBufferAOS &aosScores    = scoresData.buffer.aos;

        std::vector<int> indices(*aosKeypoints.sizePointer);
        std::iota(indices.begin(), indices.end(), 0);

        stable_sort(indices.begin(), indices.end(), [&aosScores](int a, int b) {
            uint32_t *score = reinterpret_cast<uint32_t *>(aosScores.data);
            return score[a] >= score[b]; // decreasing score order
        });

        // keep the only 'max' indexes.
        indices.resize(std::min<size_t>(indices.size(), max));

        VPIKeypointF32 *kptData = reinterpret_cast<VPIKeypointF32 *>(aosKeypoints.data);

        // reorder the keypoints to keep the first 'max' with highest scores.
        std::vector<VPIKeypointF32> kpt;
        std::transform(indices.begin(), indices.end(), std::back_inserter(kpt),
                       [kptData](int idx) { return kptData[idx]; });
        std::copy(kpt.begin(), kpt.end(), kptData);

        // update keypoint array size.
        *aosKeypoints.sizePointer = kpt.size();

        vpiArrayUnlock(scores);
        vpiArrayUnlock(keypoints);
    }

};

}

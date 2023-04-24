/*
* Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION >= 3
#    include <opencv2/imgcodecs.hpp>
#else
#    include <opencv2/contrib/contrib.hpp> // for colormap
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include <cstring> // for memset
#include <iostream>
#include <sstream>

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

int main(int argc, char *argv[])
{
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    cv::Mat cvImageLeft, cvImageRight;

    // VPI objects that will be used
    VPIImage inLeft        = NULL;
    VPIImage inRight       = NULL;
    VPIImage tmpLeft       = NULL;
    VPIImage tmpRight      = NULL;
    VPIImage stereoLeft    = NULL;
    VPIImage stereoRight   = NULL;
    VPIImage disparity     = NULL;
    VPIImage confidenceMap = NULL;
    VPIStream stream       = NULL;
    VPIPayload stereo      = NULL;

    int retval = 0;

    try
    {
        // =============================
        // Parse command line parameters

        if (argc != 4)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] +
                                     " <cpu|pva|cuda|pva-nvenc-vic|ofa|ofa-pva-vic> <left image> <right image>");
        }

        std::string strBackend       = argv[1];
        std::string strLeftFileName  = argv[2];
        std::string strRightFileName = argv[3];

        uint64_t backends;

        if (strBackend == "cpu")
        {
            backends = VPI_BACKEND_CPU;
        }
        else if (strBackend == "cuda")
        {
            backends = VPI_BACKEND_CUDA;
        }
        else if (strBackend == "pva")
        {
            backends = VPI_BACKEND_PVA;
        }
        else if (strBackend == "pva-nvenc-vic")
        {
            backends = VPI_BACKEND_PVA | VPI_BACKEND_NVENC | VPI_BACKEND_VIC;
        }
        else if (strBackend == "ofa")
        {
            backends = VPI_BACKEND_OFA;
        }
        else if (strBackend == "ofa-pva-vic")
        {
            backends = VPI_BACKEND_OFA | VPI_BACKEND_PVA | VPI_BACKEND_VIC;
        }
        else
        {
            throw std::runtime_error(
                "Backend '" + strBackend +
                "' not recognized, it must be either cpu, cuda, pva, ofa, ofa-pva-vic or pva-nvenc-vic.");
        }

        // =====================
        // Load the input images
        cvImageLeft = cv::imread(strLeftFileName);
        if (cvImageLeft.empty())
        {
            throw std::runtime_error("Can't open '" + strLeftFileName + "'");
        }

        cvImageRight = cv::imread(strRightFileName);
        if (cvImageRight.empty())
        {
            throw std::runtime_error("Can't open '" + strRightFileName + "'");
        }

        // =================================
        // Allocate all VPI resources needed

        int32_t inputWidth  = cvImageLeft.cols;
        int32_t inputHeight = cvImageLeft.rows;

        // Create the stream that will be used for processing.
        CHECK_STATUS(vpiStreamCreate(0, &stream));

        // We now wrap the loaded images into a VPIImage object to be used by VPI.
        // VPI won't make a copy of it, so the original image must be in scope at all times.
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImageLeft, 0, &inLeft));
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImageRight, 0, &inRight));

        // Format conversion parameters needed for input pre-processing
        VPIConvertImageFormatParams convParams;
        CHECK_STATUS(vpiInitConvertImageFormatParams(&convParams));

        // Set algorithm parameters to be used. Only values what differs from defaults will be overwritten.
        VPIStereoDisparityEstimatorCreationParams stereoParams;
        CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&stereoParams));

        // Default format and size for inputs and outputs
        VPIImageFormat stereoFormat    = VPI_IMAGE_FORMAT_Y16_ER;
        VPIImageFormat disparityFormat = VPI_IMAGE_FORMAT_S16;

        int stereoWidth  = inputWidth;
        int stereoHeight = inputHeight;
        int outputWidth  = inputWidth;
        int outputHeight = inputHeight;

        // Override some backend-dependent parameters
        if (strBackend == "pva-nvenc-vic")
        {
            // Input and output width and height has to be 1920x1080 in block-linear format for pva-nvenc-vic pipeline
            stereoFormat = VPI_IMAGE_FORMAT_Y16_ER_BL;
            stereoWidth  = 1920;
            stereoHeight = 1080;

            // For PVA+NVENC+VIC mode, 16bpp input must be MSB-aligned, which
            // is equivalent to say that it is Q8.8 (fixed-point, 8 decimals).
            convParams.scale = 256;

            // Maximum disparity is fixed to 256.
            stereoParams.maxDisparity = 256;

            // pva-nvenc-vic pipeline only supports downscaleFactor = 4
            stereoParams.downscaleFactor = 4;
            outputWidth                  = stereoWidth / stereoParams.downscaleFactor;
            outputHeight                 = stereoHeight / stereoParams.downscaleFactor;
        }
        else if (strBackend.find("ofa") != std::string::npos)
        {
            // Implementations using OFA require BL input
            stereoFormat = VPI_IMAGE_FORMAT_Y16_ER_BL;

            if (strBackend == "ofa")
            {
                disparityFormat = VPI_IMAGE_FORMAT_S16_BL;
            }

            // Output width including downscaleFactor must be at least max(64, maxDisparity/downscaleFactor) when OFA+PVA+VIC are used
            if (strBackend.find("pva") != std::string::npos)
            {
                int downscaledWidth = (inputWidth + stereoParams.downscaleFactor - 1) / stereoParams.downscaleFactor;
                int minWidth = std::max(stereoParams.maxDisparity / stereoParams.downscaleFactor, downscaledWidth);
                outputWidth  = std::max(64, minWidth);
                outputHeight = (inputHeight * stereoWidth) / inputWidth;
                stereoWidth  = outputWidth * stereoParams.downscaleFactor;
                stereoHeight = outputHeight * stereoParams.downscaleFactor;
            }

            // Maximum disparity can be either 128 or 256
            stereoParams.maxDisparity = 128;
        }
        else if (strBackend == "pva")
        {
            // PVA requires that input and output resolution is 480x270
            stereoWidth = outputWidth = 480;
            stereoHeight = outputHeight = 270;

            // maxDisparity must be 64
            stereoParams.maxDisparity = 64;
        }

        // Create the payload for Stereo Disparity algorithm.
        // Payload is created before the image objects so that non-supported backends can be trapped with an error.
        CHECK_STATUS(vpiCreateStereoDisparityEstimator(backends, stereoWidth, stereoHeight, stereoFormat, &stereoParams,
                                                       &stereo));

        // Create the image where the disparity map will be stored.
        CHECK_STATUS(vpiImageCreate(outputWidth, outputHeight, disparityFormat, 0, &disparity));

        // Create the input stereo images
        CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, stereoFormat, 0, &stereoLeft));
        CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, stereoFormat, 0, &stereoRight));

        // Create some temporary images, and the confidence image if the backend can support it
        if (strBackend == "pva-nvenc-vic")
        {
            // Need an temporary image to convert BGR8 input from OpenCV into pixel-linear 16bpp grayscale.
            // We can't convert it directly to block-linear since CUDA backend doesn't support it, and
            // VIC backend doesn't support BGR8 inputs.
            CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, VPI_IMAGE_FORMAT_Y16_ER, 0, &tmpLeft));
            CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, VPI_IMAGE_FORMAT_Y16_ER, 0, &tmpRight));

            // confidence map is needed for pva-nvenc-vic pipeline
            CHECK_STATUS(vpiImageCreate(outputWidth, outputHeight, VPI_IMAGE_FORMAT_U16, 0, &confidenceMap));
        }
        else if (strBackend.find("ofa") != std::string::npos)
        {
            // OFA also needs a temporary buffer for format conversion
            CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, VPI_IMAGE_FORMAT_Y16_ER, 0, &tmpLeft));
            CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, VPI_IMAGE_FORMAT_Y16_ER, 0, &tmpRight));

            if (strBackend.find("pva") != std::string::npos)
            {
                // confidence map is supported by OFA+PVA
                CHECK_STATUS(vpiImageCreate(outputWidth, outputHeight, VPI_IMAGE_FORMAT_U16, 0, &confidenceMap));
            }
        }
        else if (strBackend == "pva")
        {
            // PVA also needs a temporary buffer for format conversion and rescaling
            CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, stereoFormat, 0, &tmpLeft));
            CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, stereoFormat, 0, &tmpRight));
        }
        else if (strBackend == "cuda")
        {
            CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, VPI_IMAGE_FORMAT_U16, 0, &confidenceMap));
        }

        // ================
        // Processing stage

        // -----------------
        // Pre-process input
        if (strBackend == "pva-nvenc-vic" || strBackend == "pva" || strBackend == "ofa" || strBackend == "ofa-pva-vic")
        {
            // Convert opencv input to temporary grayscale format using CUDA
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inLeft, tmpLeft, &convParams));
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inRight, tmpRight, &convParams));

            // Do both scale and final image format conversion on VIC.
            CHECK_STATUS(
                vpiSubmitRescale(stream, VPI_BACKEND_VIC, tmpLeft, stereoLeft, VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
            CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_VIC, tmpRight, stereoRight, VPI_INTERP_LINEAR,
                                          VPI_BORDER_CLAMP, 0));
        }
        else
        {
            // Convert opencv input to grayscale format using CUDA
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inLeft, stereoLeft, &convParams));
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inRight, stereoRight, &convParams));
        }

        // ------------------------------
        // Do stereo disparity estimation

        // Submit it with the input and output images
        CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stream, backends, stereo, stereoLeft, stereoRight, disparity,
                                                       confidenceMap, NULL));

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(stream));

        // ========================================
        // Output pre-processing and saving to disk
        // Lock output to retrieve its data on cpu memory
        VPIImageData data;
        CHECK_STATUS(vpiImageLockData(disparity, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

        // Make an OpenCV matrix out of this image
        cv::Mat cvDisparity;
        CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cvDisparity));

        // Scale result and write it to disk. Disparities are in Q10.5 format,
        // so to map it to float, it gets divided by 32. Then the resulting disparity range,
        // from 0 to stereo.maxDisparity gets mapped to 0-255 for proper output.
        cvDisparity.convertTo(cvDisparity, CV_8UC1, 255.0 / (32 * stereoParams.maxDisparity), 0);

        // Apply JET colormap to turn the disparities into color, reddish hues
        // represent objects closer to the camera, blueish are farther away.
        cv::Mat cvDisparityColor;
        applyColorMap(cvDisparity, cvDisparityColor, cv::COLORMAP_JET);

        // Done handling output, don't forget to unlock it.
        CHECK_STATUS(vpiImageUnlock(disparity));

        // If we have a confidence map,
        if (confidenceMap)
        {
            // Write it to disk too.
            //
            VPIImageData data;
            CHECK_STATUS(vpiImageLockData(confidenceMap, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

            cv::Mat cvConfidence;
            CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cvConfidence));

            // Confidence map varies from 0 to 65535, we scale it to
            // [0-255].
            cvConfidence.convertTo(cvConfidence, CV_8UC1, 255.0 / 65535, 0);
            imwrite("confidence_" + strBackend + ".png", cvConfidence);

            CHECK_STATUS(vpiImageUnlock(confidenceMap));

            // When pixel confidence is 0, its color in the disparity
            // output is black.
            cv::Mat cvMask;
            threshold(cvConfidence, cvMask, 1, 255, cv::THRESH_BINARY);
            cvtColor(cvMask, cvMask, cv::COLOR_GRAY2BGR);
            bitwise_and(cvDisparityColor, cvMask, cvDisparityColor);
        }

        imwrite("disparity_" + strBackend + ".png", cvDisparityColor);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    // ========
    // Clean up

    // Destroying stream first makes sure that all work submitted to
    // it is finished.
    vpiStreamDestroy(stream);

    // Only then we can destroy the other objects, as we're sure they
    // aren't being used anymore.

    vpiImageDestroy(inLeft);
    vpiImageDestroy(inRight);
    vpiImageDestroy(tmpLeft);
    vpiImageDestroy(tmpRight);
    vpiImageDestroy(stereoLeft);
    vpiImageDestroy(stereoRight);
    vpiImageDestroy(confidenceMap);
    vpiImageDestroy(disparity);
    vpiPayloadDestroy(stereo);

    return retval;
}

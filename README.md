# kitti_flow

1) 'noc' refers to non-occluded regions, ie, regions for which the matching correspondence is inside the image domain. 'occ' refers to all image regions for which ground truth could be measured (including regions which map to points outside the image domain in the other view). for ranking the methods and for the main table all image regions are considered (corresponding to the 'occ' folders).

2) The folders testing and training contain the color video images in the sub-folders image_2 (left image) and image_3 (right image).

3) Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
contains the u-component, the second channel the v-component and the third
channel denotes if the pixel is valid or not (1 if true, 0 otherwise). To convert
the u-/v-flow into floating point values, convert the value to float, subtract
2^15 and divide the result by 64.0:

flow_u(u,v) = ((float)I(u,v,1)-2^15)/64.0;
flow_v(u,v) = ((float)I(u,v,2)-2^15)/64.0;
valid(u,v)  = (bool)I(u,v,3);

4) dataset/devkit_scene_flow/devkit/cpp/io_flow.h :: readFlowField

5) Test flow pair lies in image_2 folder

# nvidia_vpi
1) sudo apt install libnvvpi2 vpi2-dev vpi2-samples
2) if apt unavaliable, download vpi-dev-2.0.14-aarch64-l4t.deb vpi-lib-2.0.14-aarch64-l4t.deb in sdk_manager.
3) maybe NVENC hardware with dense optical flow support not present on orin,  refer to https://forums.developer.nvidia.com/t/nvenc-hardware-with-dense-optical-flow-support-not-present/224194

# Jetson Clock Frequency and Power Settings
https://docs.nvidia.com/vpi/algo_performance.html

sudo ./clocks.sh --max      # maximize the clock frequencies
sudo ./clocks.sh --restore  # restore the clock frequencies
sudo jetson_clocks --show   # show config
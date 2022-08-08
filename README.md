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
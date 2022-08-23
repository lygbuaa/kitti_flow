#!/bin/sh
# set -ex
TARGET=$1

kitti_img2_path="--kitti_img2_path=./dataset/data_scene_flow/training/image_2"
kitti_flow_gt_path="--kitti_flow_gt_path=./dataset/data_scene_flow/training/flow_noc"
OUTPUT_PATH="--output_img_path=./output"
GLOG_PATH="--glog_path=./logs"
VIDEO_PATH="--video_file_path=./dataset/traffic.mp4"
STREAM_NUM="--stream_number=8"
ENABLE_VISUAL="--enable_visual=false"
BIN_PATH=./build/bin/

if [ -e $BIN_PATH/KittiFlow ]; then
    $BIN_PATH/KittiFlow $kitti_img2_path $kitti_flow_gt_path $OUTPUT_PATH $GLOG_PATH $STREAM_NUM $VIDEO_PATH $ENABLE_VISUAL
else
    echo -e "\n@e@ --> KittiFlow not found \n"
    exit 1
fi
exit 0
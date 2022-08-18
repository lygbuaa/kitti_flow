#!/bin/sh
# set -ex
TARGET=$1

KITTI_IMG_PATH="--kitti_img_path=./dataset/data_scene_flow/training/image_2"
KITTI_GT_PATH="--kitti_gt_path=./dataset/data_scene_flow/training/flow_noc"
OUTPUT_PATH="--output_img_path=./output"
GLOG_PATH="--glog_path=./logs"
VIDEO_PATH="--video_file_path=./dataset/traffic.mp4"
STREAM_NUM="--stream_number=1"
ENABLE_VISUAL="--enable_visual=false"
BIN_PATH=./build/bin/

if [ -e $BIN_PATH/KittiFlow ]; then
    $BIN_PATH/KittiFlow $KITTI_IMG_PATH $KITTI_GT_PATH $OUTPUT_PATH $GLOG_PATH $STREAM_NUM $VIDEO_PATH $ENABLE_VISUAL
else
    echo -e "\n@e@ --> KittiFlow not found \n"
    exit 1
fi
exit 0
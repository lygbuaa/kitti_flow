#!/bin/sh
# set -ex
TARGET=$1

KITTI_IMG_PATH="--kitti_img_path=./dataset/data_scene_flow/training/image_2"
KITTI_GT_PATH="--kitti_gt_path=./dataset/data_scene_flow/training/flow_noc"
OUTPUT_PATH="--output_img_path=./output"
GLOG_PATH="--glog_path=./logs"
BIN_PATH=./build/bin/

if [ -e $BIN_PATH/KittiFlow ]; then
    $BIN_PATH/KittiFlow $KITTI_IMG_PATH $KITTI_GT_PATH $OUTPUT_PATH $GLOG_PATH
else
    echo -e "\n@e@ --> KittiFlow not found \n"
    exit 1
fi
exit 0
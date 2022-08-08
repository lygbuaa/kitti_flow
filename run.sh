#!/bin/sh
# set -ex
TARGET=$1

INPUT_PATH="--input_img_path=./dataset"
OUTPUT_PATH="--output_img_path=./output"
BIN_PATH=./build/bin/

if [ -e $BIN_PATH/KittiFlow ]; then
    $BIN_PATH/KittiFlow $INPUT_PATH $OUTPUT_PATH
else
    echo -e "\n@e@ --> KittiFlow not found \n"
    exit 1
fi
exit 0
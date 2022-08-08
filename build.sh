#!/bin/bash
set -ex
reset
mkdir -p build
cd build
make clean
cmake ..
make
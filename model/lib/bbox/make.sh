#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda

echo "Compiling nms kernels by nvcc..."
cd torch_nms/src/cuda
$CUDA_PATH/bin/nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61

cd ../../
python3 build.py

echo "Building nms and overlap by cython..."
cd ../cpu
python3 setup.py build_ext --inplace
rm -rf build

# Introduction

This example shows how to use the uni-Freiburg dataset available under 
`/cephyr/NOBACKUP/Datasets/uni-freiburg-oVision-Scene-Flow-dataset/FlyingThings3D_subset/train/disparity_occlusions` to train a CNN  with TensorFlow. We will use 
Keras' preprocessing utilities specifically the `flow_from_directory` function from the `ImageDataGenerator` class just to 
illustrate the workflow. Note that the directory structure of the above path is recognized by this function. It thus implies a binary classification problem with the two classes being left and right. However, we do not aim for a converged model in this example, as it takes several days for it to get trained on more than 43000 images. The aim of this example is just to illustrate the use of the utilities and tools. 

## Environment setup

The following modules are required for this example:

`ml GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 TensorFlow/2.1.0-Python-3.7.4 Pillow/6.2.1`

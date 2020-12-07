# Introduction

This example shows how to use the YouTube dataset available under 
`/cephyr/NOBACKUP/Datasets/YouTube-dataset/YouTubeVos_InstanceSegmentation_2019/` to train a CNN  with TensorFlow. We will use 
Keras' preprocessing utilities specifically the `flow_from_directory` function form the `ImageDataGenerator` class just to 
illustrate the workflow. Note however that we are using an incorrect model as the directory structure does not match the 
classification problem. 

## Environment setup

The following modules are required for this example:

`ml GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 TensorFlow/2.1.0-Python-3.7.4 Pillow/6.2.1`

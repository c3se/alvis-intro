# Introduction

This example shows how to use the tiny-imagenet dataset available under 
`/cephyr/NOBACKUP/Datasets/tiny-imagenet-200/train` to train a CNN  with TensorFlow. We will use 
Keras' preprocessing utilities specifically the `flow_from_directory` function from the `ImageDataGenerator` class just to 
illustrate the workflow. Note that the directory structure of the above path is recognized by this function. It thus implies a classification problem with the 200 classes 

## Environment setup

The following modules are required for this example:

`ml GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5 TensorFlow/2.5.0 Pillow/8.0.1`

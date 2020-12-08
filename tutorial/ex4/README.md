# Introduction

In this tutorial, we will show how to use multiple GPUs to speed up an intensive PyTorch training application using a custom datasets. The code is based on a 
slightly modified version of the PyTorch GPU benchmark which uses fake image data: <https://github.com/ryujaehun/pytorch-gpu-benchmark>.

We make use of the `torchvision` built-in models specifically the `resnext101_32x8d` model. This is only to show the application. You can experiment with other 
models as well. The benchmark can be run with quite many different datatypes. This example is restricted to float and half-precision datatypes, but you can run it 
with double precision as well. It will just take longer to finish. 

## Environment setup
The following modules need to be loaded to run the code:

`ml GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 PyTorch/1.4.0-Python-3.7.4 torchvision/0.7.0-Python-3.7.4-PyTorch-1.6.0 IPython`

Alternatively, you can use the relevant job script to run it using the PyTorch singularity container image.

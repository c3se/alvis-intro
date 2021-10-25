# Introduction

This is an introductory example on using PyTorch using the publicly available
MNIST dataset. You can use PyTorch `datasets` module to access the datasets
that are not provided by us under `/cephyr/NOBACKUP/Datasets`. It can also be
used for your custom datasets. In this example, the performance of the T4 and
V100 GPUs for training a CNN using the `torchvision` models will be compared.
We will use a slightly modified example provided on PyTorch GitHub page:
<https://github.com/pytorch/examples/blob/master/mnist/main.py>.

Note that in this case the data _is_ available at `/cephyr/NOBACKUP/Datasets`
and should instead have been loaded with:
```python
train_set = datasets.MNIST(
    '/cephyr/NOBACKUP/Datasets',
    download=False,
    train=True,
    transform=transform,
)
```


## Environment setup

To run the python code, the following modules need to be loaded:

`ml GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 PyTorch/1.4.0-Python-3.7.4 torchvision/0.7.0-Python-3.7.4-PyTorch-1.6.0 IPython`

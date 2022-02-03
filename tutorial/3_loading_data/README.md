# Introduction
In this part we will investigate different ways to get and use data. However,
it is worth noting that there are numerous different ways of doing all of these
and what you will learn here are simply three different examples when using:
 1. Your own data on the cluster
 2. Data available at `/cephyr/NOBACKUP/Datasets/`
 3. Data through some API

In the future we will introduce a data transfer node that is meant to
specifically handle downloading datasets and a utility to make this easier. If
you are planning on downloading datasets to Alvis in the near future don't
hesitate to reach out through support and we can help you and make sure that we
will cover similar use cases in the future.

## PyTorch
For the following excercises you will need to load the following modules:
```bash
flat_modules
ml PyTorch/1.9.0-fosscuda-2020b matplotlib/3.3.3-fosscuda-2020b JupyterLab/2.2.8-GCCcore-10.2.0
```

Now you should open up `data-pytorch.ipynb` and follow the instructions there.

## TensorFlow
For the following excercises you will need to load the following modules:
```bash
flat_modules
ml TensorFlow/2.5.0-fosscuda-2020b matplotlib/3.3.3-fosscuda-2020b JupyterLab/2.2.8-GCCcore-10.2.0
```

Now you should open up `data-tensorflow.ipynb` and follow the instructions there.

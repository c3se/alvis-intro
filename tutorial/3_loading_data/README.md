# Introduction
In this part we will investigate different ways to get and use data. However,
it is worth noting that there are numerous different ways of doing all of these
and what you will learn here are simply three different examples when using:
 1. Your own data on the cluster
 2. Data available at `/mimer/NOBACKUP/Datasets/`
 3. Data through some API

Log-in node alvis2 is also the dedicated data-transfer node and is the node to
use if you are transfering any significant amount of data. There is also a
data-transfer utility in the works with a backend that is currently in alpha
testing. If you are planning on downloading datasets to Alvis in the near future
don't hesitate to reach out through support and we can help you and make sure
that we will cover similar use cases in the future.

Note, that with the new Mimer storage resource, data access should be quite
fast and the need to store data locally is significantly reduced. 

## PyTorch
For the following excercises you will need to load the following modules:
```bash
ml purge
ml PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
ml matplotlib/3.7.2-gfbf-2023a
ml JupyterLab/4.0.5-GCCcore-12.3.0
```

Now you should open up `data-pytorch.ipynb` and follow the instructions there.

## TensorFlow
For the following excercises you will need to load the following modules:
```bash
ml purge
ml TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
ml matplotlib/3.7.2-gfbf-2023a
ml JupyterLab/4.0.5-GCCcore-12.3.0
```

Now you should open up `data-tensorflow.ipynb` and follow the instructions there.


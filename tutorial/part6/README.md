# Introduction

In this tutorial, we will show how to use multiple GPUs when training with
PyTorch. If you are going to parallelize your work-load there are primarily two
different approaches:
 - Data Parallelism
 - Model Parallelism

With data parallelism you will have your model broadcast to all GPUs and then
have separete batches on the different GPUs calculate the weight updates in
parallel and then summarise into an update as if you had had a single large
batch. This is useful if you have a large dataset and want to have larger
batches than fit on the GPUs memory.

Model parallelism is about storing parts of the model on different GPUs. This is
used if your model is too large to fit on a single GPU, for the GPUs available
on Alvis this should rarely be a problem but in some rare cases you might reach
this limit. Remember that you can see your resource usage for a job with the
command `job_stats.py`. 

## Environment setup
***TODO***

## Data Parallelism with DP
The simplest option for Data Parallelism on a single node is
[Data Parallel](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html)
(DP). However, this is not the recomended way (see PyTorch Documentation for
details on why), but we will go through how it can be used anyway. 

To use DP you wrap your model
```python
model = ...
model = torch.nn.DataParallel(model)
model = model.to("cuda")
```
and then this wrapper will chunk the batch to each forward and distribute them
to model replicas on each device.


## Data Parallelism with DDP
In this part we will take a look at Distributed Data Parallel (DDP).
According to the
[PyTorch documentation](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)
DDP is currently the go to method for data parallelism even on a single node.

DDP works differently from DP by running on several different processes. Therefore, we have some options to choose between:
1. Launching processes with `torch.multiprocessing` **TODO**
2. Launching processes with `torch.distributed.launch` **TODO**
3. Launching processes with `srun` or `mpirun` **TODO**


# Introduction
Many ML tasks scale far across the resources contained in a single compute node.
Unfortunately, simply asking Slurm for more cores or GPUs will most likely not
make your job work across multiple nodes, instead you might end up running the
same code several times (wih undefined outcomes, if you specify the same output
destination). What you need is to tell your application to work (train)
distributed. In HPC, using mostly CPUs, this has traditionally been done using
MPI or a framework built on top of MPI. For NVIDIA GPUs 
[NCCL](https://developer.nvidia.com/nccl) is another option.

## PyTorch
Note that we can no longer use `torch.nn.DataParallel` when we want to
parallelize across nodes. However, we will instead consider the framework
Horovod which we skipped in the previous part.

### Environment setup
Load PyTorch:
```bash
module purge
ml PyTorch/2.12.1-foss-2022a-CUDA-11.7.0
```

### Data Parallelism with DDP
In this part we will take a look at Distributed Data Parallel (DDP). The
recommended option is to launch one `torchrun` process per node. This would
look something like
```bash
srun torchrun ... my_distributed_ml.py
```

For details how this is done check out the scripts `PyTorch/ddp.py` and `PyTorch/jobscript.sh`.

## PyTorch with Horovod
In this tutorial we will run a distributed deep learning training job across
several compute nodes using the [Horovod](https://github.com/horovod/horovod)
framework. Horovod is a distributed traning framework first developed by Uber,
but with significant contributions from Facebook and from an active open-source
community. You can read more about Horovod
[here](https://arxiv.org/pdf/1802.05799.pdf).

### Environment set-up
Load the relevant modules
```
module purge
ml Horovod/0.28.1-foss-2022a-CUDA-11.7.0-PyTorch-1.13.1
```


## TensorFlow
TensorFlow has their own [guide to distributed training](https://www.tensorflow.org/guide/distributed_training)
which is a good reference to know of. Here we will cover some of that material.

*N.B.* the TensorFlow example currently only works with Horovod. If you are
planning to run multi-node TensorFlow jobs, contact support at
<https://supr.naiss.se/support/> and we'll help you directly.

### Environment setup
To run these examples load pytorch:
```bash
module purge
ml TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
```

### Data Parallelism with MultiWorkerMirroredStrategy
MultiWorkerMirroredStrategy works similar to MirroredStrategy that we saw
before, except now it is possible to use several workers. To be able to handle
the communications between the workers we'll now need to use a cluster resolver
that finds the necessary parameters for communcation to go right. In our case we
can simply use `tf.distribute.cluster_resolver.SlurmClusterResolver`.

```python
cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
strategy = tf.distributed.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver)

with strategy.scope():
    my_model = MyModel()

# ... use model as usual
```

**Note:** MultiWorkerMirroredStrategy has to be created at the beginning of the
program. Else you might encounter
```
RuntimeError: Collective ops must be configured at program startup
```

Another option that we have now is what collectives to use for cross-host
communication. On Alvis you should typically specify `NCCL`.

## TensorFlow with Horovod
In this tutorial we will run a distributed deep learning training job across
several compute nodes using the [Horovod](https://github.com/horovod/horovod)
framework. Horovod is a distributed traning framework first developed by Uber,
but with significant contributions from Facebook and from an active open-source
community. You can read more about Horovod
[here](https://arxiv.org/pdf/1802.05799.pdf).

For this part take a look at `TensorFlow/hvd.py` and note that there are atleast
3 important parts with using Horovod with TensorFlow. With
`import horovod.tensorflow as hvd`:

1. Initializing Horovod `hvd.init()`
2. Wrapping the optimizer in `hvd.DistributedOptimizer`
3. Using the `hvd.keras.callbacks.BroadcastGlobalVariablesCallback` callback

### Environment set-up
Load the relevant modules
```
module purge
ml Horovod-0.28.1-foss-2022a-CUDA-11.7.0-TensorFlow-2.11.0.eb
```

Also see `jobscript.sh` for how to submit this to a job.


## Exercises
1. Checkout the different scripts and try to get an idea of what they are doing and what their differences are.
2. Submit jobscripts and see the different outputs, try with verbose=True for the model to see the size of the inputs.
3. (PyTorch only) From 2 you might note that for DDP all the processes process all the data. Checkout `torch.utils.data.DistributedSampler` and modify one of the scripts such that the data actually becomes distributed over the different devices. You might have to change batch size for this to work. In practice keep in mind that changing batch size then you should usually change learning rate as well.
4. Play around with different sizes of the dataset, different models and sizes of models, and/or different number of nodes. Check out the Grafana generated by `job_stats.py YOUR_JOBID_HERE`, are you using all GPUs? To what extent? You might have to increase dataset size or other things such that the job runs long enough to show up on the grafana plots.

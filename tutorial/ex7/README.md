# Introduction
In this tutorial we will run a distributed deep learning training job across
several compute nodes using the [Horovod](https://github.com/horovod/horovod)
framework. Horovod is a distributed traning framework first developed by Uber,
but with significant contributions from Facebook and from an active open-source
community. You can read more about Horovod
[here](https://arxiv.org/pdf/1802.05799.pdf).

## Motivation
Many ML tasks scale far across the resources contained in a single compute
node. Unfortunately, simply asking Slurm for more cores or GPUs will most
likely not make your job work across multiple nodes, instead you might end up
running the same code several times (wih undefined outcomes, if you specify the
same output destination). What you need is to tell your application to work
(train) distributed. In HPC, using mostly CPUs, this has traditionally been
done using [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) or a
framework built on top of MPI. Horovod relies on MPI (or Gloo, if MPI is not
installed) to communicate between the tasks, or ranks, that makes up a
distributed job.

## Environment setup
As this example is meant to run across several compute nodes you need to submit
the code using Slurm. We provide the Slurm job file `jobsubmit` as a working example.

### The following dataset needs to be available
This example is based on the previous examples of training a model using the
MNIST dataset. We provide the dataset in `/cephyr/NOBACKUP/Datasets/MNIST` but
to simplify the code in this example we download the dataset in pickled format.
For production use you should always check first if the dataset is already
available in `/cephyr/NOBACKUP/Datasets` before you download it. The size for
MNIST should only be around 11M compressed.
```
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

## Running the code
You will need to edit the job script `jobsubmit` at the places where edits are
highlighted, for instance, you will need to enter your SNIC-project. You can then submit
the job to Slurm using `sbatch` from a login node.
```
$ sbatch jobsubmit
```
The job may not immediately run, but you will get a job ID you can track. You
can check the status of your scheduled jobs bry running `squeue`.
```
$ squeue -u $USER
```
Once the job completed you will find that several additional files have been
created in your working directory. This is expected as the example builds upon
the profiling example, and that Alvis is configured to provide a performance
report after each job completes (and as your job hopefully ran on multiple
nodes you will get one report showing compute utilization for each compute
node).

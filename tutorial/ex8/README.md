# Introduction
In this tutorial we will run a distributed deep learning training job across
several compute nodes using the [Horovod](https://github.com/horovod/horovod)
framework. Horovod is a distributed traning framework first developed by Uber,
but with significant contributions from Facebook and from an active open-source
community. You can read more about Horovod
[here](https://arxiv.org/pdf/1802.05799.pdf).

## Motivation
Many ML tasks scale far across the resources contained in a single compute node.
Unfortunately, simply asking Slurm for more cores or GPUs will most likely not
make your job work across multiple nodes, instead you might end up running the
same code several times (wih undefined outcomes, if you specify the same output
destination). What you need is to tell your application to work (train)
distributed. In HPC, using mostly CPUs, this has traditionally been done using
[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) or a framework
built on top of MPI. Horovod relies on MPI (or Gloo, if MPI is not installed) to
communicate between the tasks, or ranks, that makes up a distributed job.

## Environment setup
As this example is meant to run across several compute nodes you need to submit
the code using Slurm. We provide the Slurm job file `ex8.sh` as a working
example.

## Running the code
You will need to edit the job script `ex8.sh` at the places where edits are
highlighted, for instance, you will need to enter your SNIC-project. You can
then submit the job to Slurm using `sbatch` from a login node.
```
$ sbatch ex8.sh
```
The job may not immediately run, but you will get a job ID you can track. You
can check the status of your scheduled jobs bry running `squeue`.
```
$ squeue -u $USER
```
Once the job completed you will find that several additional files have been
created in your working directory. This is expected as Alvis is configured to
provide a performance report after each job completes (and as your job hopefully
ran on multiple nodes you will get one report showing compute utilization for
each compute node).

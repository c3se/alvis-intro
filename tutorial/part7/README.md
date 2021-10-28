# Introduction
A natural continuation of parallizing over GPUs on a single node is to
parallelize the workload over several nodes. However, this will have to be
handled slightly differently, luckily for us most of the work is already done by
other parties.

## PyTorch
Note that we can no longer use `torch.nn.DataParallel` when we want to
parallelize across nodes. However, we will instead consider the framework
Horovod which we skipped in the previous part.

### Environment setup
Load PyTorch:
```bash
flat_modules
ml PyTorch/1.9.0-fosscuda-2020b
```

### Data Parallelism with DDP
In this part we will take a look at Distributed Data Parallel (DDP). The simplest option is to let MPI handle launching the different processes, to do this you specify how many tasks you want in your submit script and then launch the program with `srun`. This could look something like
```bash
...
#SBATCH --ntasks-per-node=8
...
srun python my_distributed_ml.py
```

### Data Parallelism with Horovod

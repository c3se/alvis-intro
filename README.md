# Introduction to Alvis


# Aims of this seminar
* Aims
* This presentation is available on C3SE's web page:
    * <https://www.c3se.chalmers.se/documentation/intro-alvis/presentation.html>
    * <https://www.c3se.chalmers.se/documentation/intro-alvis/slides>


# Our systems: Alvis
* SNIC resource dedicated to AI/ML research
* consists of SMP nodes accelerated with multiple GPUs
* Alvis goes in production in three phases:
    * Phase 1A: equipped with 44 Nvidia Tesla V100 GPUs (in production)
    * Phase 1B: equipped with 164 Nvidia Tesla T4 GPUs, 4 Nvidia Tesla A100 GPUs (in production)
    * Phase 2: will be in production in 2021
* Node details: <https://www.c3se.chalmers.se/about/Alvis/>
* Login server: `ssh CID@alvis1.c3se.chalmers.se`
* The same principles regarding how to login, where to run heavy jobs, how to submit job scripts, etc. apply


# Modules and containers
* Alvis software stack was designed to make heavy use of optimized container images mostly from, but not restricted to, Nvidia NGC catalogue
   * Available both from the central storage under `apps/hpc-ai-containers` as well as on Singularity Hub: <https://singularity-hub.org/collections/4791>
   * Allows you to prototype your work on your local machine and deploy it in a reproducible way on Alvis for large-scale training using the exact same environment  
   * Instructions on how to use the images and some tutorial can be found on <https://github.com/c3se/containers>

* In addition, some pieces of software are also provided in the form of modules, and you can use them in a similar way as you would do on our other systems 


# Software installation
* We provide `pip`, `singularity`, `conda`, and `virtualenv` so you can install your own Python packages locally.
* We build a lot of software and containers for general use
* To build your own singularity containers, see: <https://github.com/c3se/containers/tree/master/Tutorial#further-remarks>
   * and, for running your own Singularity containers, see: <https://www.c3se.chalmers.se/documentation/software/#singularity>


# Datasets
* Depending on the license type and permissions, a number of popular datasets have been made semi-publicly available through the central storage under `/cephyr/NOBACKUP/Datasets/`
* In all cases, the use of the datasets are only allowed for non-commercial, research applications
* note that in certain cases, the provider of the dataset requires you to cite some literature if you use the dataset in your research
* it is the responsibility of the users to make sure their use of the datasets complies with the above-mentioned permissions and requirements
* In some cases, further information about the dataset can be found in a README file under the pertinent directory
* A list of the currently available datasets and supplementary information can found on <https://www.c3se.chalmers.se/documentation/applications/datasets/>
* If you need a particular dataset that is publicly available but is missing in the above list, feel free to contact support. We may be able to provide it centrally to all users. 


# Storage
* See <https://www.c3se.chalmers.se/documentation/filesystem/>
* Your home directory is quite limited; additional storage areas must be applied for via SUPR.
* The `C3SE_quota` shows you all your centre storage areas, usage and quotas.

```text
Path: /cephyr/users/my_user
  Space used:    17.5GiB       Quota:      30GiB
  Files used:      20559       Quota:      60000

Path: /cephyr/NOBACKUP/groups/my_storage_project
  Space used:  2646.5GiB       Quota:    5000GiB
  Files used:     795546       Quota:    1000000
```


# Running jobs on Alvis
* Alvis is dedicated to GPU-hungry computations, therefore your job must allocate at least one GPU
* You only allocate GPUs, cores and RAM is assigned automatically
* Hyperthreading is **disabled** on Alvis
* Alvis comes in three phases (I, II, and III), and there is a variety in terms of:
     * number of cores
     * CPU architecture
     * number and type of GPUs
     * memory per GPU
     * memory per node
* Pay close attention to the above-mentioned items in your job submission script to pick the right hardware  
     * for instance, phase Ia comes with NVIDIA V100 GPUs, while phase Ib is equipped with T4 GPUs

# Allocating GPUs on Alvis
* Specify the type, of GPUs you want and the number of them per node, e.g:
    * `#SBATCH --gpus-per-node=V100:2`
    * `#SBATCH --gpus-per-node=T4:3`
    * `#SBATCH --gpus-per-node=A100:1`
* Limit to nodes with more RAM:
    * `#SBATCH --gpus-per-node=V100:2 -C 2xV100`
    * `#SBATCH --gpus-per-node=T4:1 -C MEM1536`
* Many more expert options:
    * `#SBATCH --gpus-per-node=T4:8 -N 2 --cpus-per-task=32`
    * `#SBATCH -N 2 --gres=ptmpdir:1`
    * `#SBATCH --gres=gpuexlc:1,mps:1`
* Mixing GPUs of different types is not possible

# GPU cost on Alvis
| Type | Memory per GPU | Cores per GPU | Cost |
|------|----------------|---------------|------|
| V100 | 96 or 192 GB   | 8             | 8    |
| T4   | 72 or 192 GB   | 4             | 2    |
| A100 | 192 GB         | 8             | 16   |

* E.g. using 2xT4 gpus for 10 hours costs 40 "core hours"
* Cost reflects the price of the hardware

# Querying visible devices
* Many codes will pick up `$CUDA_VISIBLE_DEVICES` automatically and do the right thing
```
   srun -A YOUR_ACCOUNT -t 00:02:00 --gpus-per-node=V100:2 --pty bash
   srun: job 22441 queued and waiting for resources
   srun: job 22441 has been allocated resources
   $ echo ${CUDA_VISIBLE_DEVICES}
   0,1
```

# Alvis batch script example
```bash
#!/bin/bash
#SBATCH -A SNIC2020-Y-X -p alvis
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:V100:1

unzip many_tiny_files_dataset.zip -d $TMPDIR/
singularity exec --nv ~/tensorflow-2.1.0.sif trainer.py --training_input=$TMPDIR/
# or use available containers e.g.
# /apps/hpc-ai-containers/TensorFlow/TensorFlow_v2.3.1-tf2-py3-GPU-Jupyter.sif
```

# Alvis batch script example

```bash
#!/bin/bash
#SBATCH -A SNIC2020-Y-X -p alvis
#SBATCH --gpus-per-node=T4:1
#SBATCH -t 5:00:00
#SBATCH --array=0-99
#SBATCH --mail-user=zapp.brannigan@chalmers.se --mail-type=end

module load MATLAB
cp cat_pictures_$SLURM_ARRAY_TASK_ID.mat $TMPDIR/training_set.mat
cp analysis.m $TMPDIR
cd $TMPDIR
RunMatlab.sh -f analysis.m
cp out.mat $SLURM_SUBMIT_DIR/out_$SLURM_ARRAY_TASK_ID.mat
```

* Environment variables like `$SLURM_ARRAY_TASK_ID` can also be accessed from within all programming languages, e.g:

```matlab
array_id = getenv('SLURM_ARRAY_TASK_ID'); % matlab
```

```python
array_id = os.getenv('SLURM_ARRAY_TASK_ID') # python
```

# Alvis batch script example

```bash
#!/bin/bash
#SBATCH -A SNIC2020-Y-X -p alvis
#SBATCH -N 2
#SBATCH --gpus-per-node=T4:8
## Parallel If you want to use parallel TMPDIR as well:
#SBATCH --gres=ptmpdir:1

module load fosscuda/2020a TensorFlow/2.3.1
mpirun python training.py
```

# Interactive use

* Login node allows for light interactive use.
* SSH or Thinlinc
    * <https://www.c3se.chalmers.se/documentation/connecting/>
* Run interactively on compute nodes with `srun`, e.g.
    * `srun -A SNIC2020-X-Y -p alvis --gpus-per-node=T4:1 bash`
* Jupyter Notebooks can run on login node or on compute nodes
    * Follow steps on <https://www.c3se.chalmers.se/documentation/applications/jupyter/>
    * `srun -A SNIC2020-X-Y -p alvis -t 8:00:00 --gpus-per-node=T4:1 --pty jupyter lab`
    * `srun -A SNIC2020-X-Y -p alvis -t 4:00:00 --gpus-per-node=T4:2 --pty `
      `singularity exec --nv my_container.sif jupyter notebook`


# Job monitoring
* `jobinfo` shows you the queue and available GPUs
* dcgmi reports after jobs finish
* `job_stats.py JOBID` (work in progress)
* `sinfo -Rl` command shows how many nodes are down for repair.
* The health status page gives an overview of what the node(s) in your job are doing
* Check e.g. memory usage, user, system, and wait CPU utilization, disk usage, etc
* See summary of CPU and memory utilization (only available after job completes): `seff JOBID`


# Things to keep in mind
* Never run (big or long) jobs on the login node! If you do, we will kill the processes.
  If you keep doing it, we'll throw you out and block you from logging in for a while!
  Prepare your job, do tests and check that everything's OK before submitting the job, but don't run the job there!
* Keep an eye on what's going on - use normal Linux tools on the login node and on the allocated nodes to check CPU, memory and network usage, etc. Especially for new jobscripts/codes!
* Think about what you do - if you by mistake copy very large files back and forth you can slow the storage servers or network to a crawl

# Getting support
* We provide support to our users, but not for any and all problems
* We can help you with software installation issues, and recommend compiler flags etc. for optimal performance
* We can install software system-wide if there are many users who need it - but probably not for one user (unless the installation is simple)
* We don't support your application software or help debugging your code/model or prepare your input files

# Getting support
* C3SE staff are available in our offices, to help with those things that are hard to put into a support request email (book a time in advance please)
* Rooms O5105B, O5110 and O5111 Origo building - Fysikgården 1, one floor up, ring the bell to the right
* We also offer advanced support for things like performance optimization, advanced help with software development tools or debuggers, workflow automation through scripting, etc.

# Getting support - error reports
* In order to help you, we need as much and as good information as possible:
    * **What's the job-ID of the failing job?**
    * **What working directory and what job-script?**
    * What software are you using?
    * What's happening - especially error messages?
    * Did this work before, or has it never worked?
    * Do you have a minimal example?
    * No need to attach files; just point us to a directory on the system.
    * Where are the files you've used - scripts, logs etc?
    * Look at our Getting support page

* Support cases through <https://supr.snic.se/support>

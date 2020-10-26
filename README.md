---
title: "Introduction to Alvis"
fontsize: 10pt
---

# Aims of this seminar
* Aims
* This presentation is available on C3SE's web page:
    * <https://www.c3se.chalmers.se/documentation/intro-alvis/presentation.html>
    * <https://www.c3se.chalmers.se/documentation/intro-alvis/slides>


# Our systems: Alvis
* SNIC resource dedicated to AI/ML research
* consists of SMP nodes accelerated with multiple GPUs
* Alvis goes in production in three phases:
    * Phase 1A: equipped with Nvidia Tesla V100 GPUs (in production)
    * Phase 1B: equipped with Nvidia Tesla T4 GPUs   (in production)
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


# Running jobs on Alvis
* Alvis is dedicated to GPU-hungry computations, therefore your job must allocate at least one GPU
* On Alvis you can allocate individual **cores** (tasks)
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
* You can specify the number of GPUs and let the scheduler decide the type
    * `#SBATCH --gpus-per-node=3`
* You can also specify the type (recommended):
    * `#SBATCH --gpus-per-node=V100:3`
* Currently, mixing GPUs of different types is not allowed 



# Vera script example

Note: You can (currently) only allocate a minimum of 1 core = 2 threads on Vera

```bash
#!/bin/bash
#SBATCH -A C3SE2018-1-2
## Note! Vera has hyperthreading enabled:
## n * c = 128 threads total = 2 nodes
## This should launch 32 MPI-processes on each node.
#SBATCH -n 64
#SBATCH -c 2
#SBATCH -t 2-00:00:00
#SBATCH --gres=ptmpdir:1

module load ABAQUS intel
cp train_break.inp $TMPDIR
cd $TMPDIR

abaqus cpus=$SLURM_NTASKS mp_mode=mpi job=train_break

cp train_break.odb $SLURM_SUBMIT_DIR
```

# Vera script example

```bash
#!/bin/bash
#SBATCH -A C3SE2018-1-2 -p hebbe
#SBATCH --gres=gpu:V100:1

unzip many_tiny_files_dataset.zip -d $TMPDIR/
singularity exec --nv ~/tensorflow-2.1.0.simg trainer.py --training_input=$TMPDIR/
```


# Hebbe script example
* Submitted with `sbatch --array=0-99 wind_turbine.sh`

```bash
#!/bin/bash
#SBATCH -A SNIC2017-1-2
#SBATCH -n 1
#SBATCH -t 15:00:00
#SBATCH --mail-user=zapp.brannigan@chalmers.se --mail-type=end

module load MATLAB
cp wind_load_$SLURM_ARRAY_TASK_ID.mat $TMPDIR/wind_load.mat
cp wind_turbine.m $TMPDIR
cd $TMPDIR
RunMatlab.sh -f wind_turbine.m
cp out.mat $SLURM_SUBMIT_DIR/out_$SLURM_ARRAY_TASK_ID.mat
```

* Environment variables like `$SLURM_ARRAY_TASK_ID` can also be accessed from within all programming languages, e.g:

```matlab
array_id = getenv('SLURM_ARRAY_TASK_ID'); % matlab
```

```python
array_id = os.getenv('SLURM_ARRAY_TASK_ID') # python
```

# Hebbe script example
* Submitted with `sbatch --array=0-50:5 diffusion.sh`

```bash
#!/bin/bash
#SBATCH -A C3SE2017-1-2
#SBATCH -n 40 -t 2-00:00:00

module load intel/2017a
# Set up new folder, copy the input file there
temperature=$SLURM_ARRAY_TASK_ID
dir=temp_$temperature
mkdir $dir; cd $dir
cp $HOME/base_input.in input.in
# Set the temperature in the input file:
sed -i 's/TEMPERATURE_PLACEHOLDER/$temperature' input.in

mpirun $HOME/software/my_md_tool -f input.in
```

Here, the array index is used directly as input.
If it turns out that 50 degrees was insufficient, then we could do another run:
```bash
sbatch --array=55-80:5 diffusion.sh
```

# Hebbe script example
Submitted with: `sbatch -N 3 -J residual_stress run_oofem.sh`

```bash
#!/bin/bash
#SBATCH -A C3SE507-15-6 -p mob
#SBATCH --ntasks-per-node=20
#SBATCH -t 6-00:00:00
#SBATCH --gres=ptmpdir:1

module load intel/2017a PETSc
cp $SLURM_JOB_NAME.in $TMPDIR
cd $TMPDIR
mkdir $SLURM_SUBMIT_DIR/$SLURM_JOB_NAME
while sleep 1h; do
  rsync -a *.vtu $SLURM_SUBMIT_DIR/$SLURM_JOB_NAME
done &
LOOPPID=$!

mpirun $HOME/bin/oofem -p -f "$SLURM_JOB_NAME.in"
kill $LOOPPID
rsync -a *.vtu $SLURM_SUBMIT_DIR/oofem/$SLURM_JOBNAME/
```

# Alvis script example

```bash
#!/bin/bash
#SBATCH -A C3SE2020-2-3
#SBATCH -n 4
#SBATCH -t 2-00:00:00
#SBATCH --gpu-per-node=T4:2

#If you want to use parallel TMPDIR as well:
#SBATCH --gres=ptmpdir:1

module load foo

mpirun -n 4 ./bar
```


# Interactive use

* Jupyter Notebooks
* OOD
* Thinlinc

# GPU flags
* exclusive
* mps

# Job monitoring
* dcgmi
* ganglia integrated in OOD?
* job_stats.py JOBID
* ganglia_url.py JOBID
* `sinfo -Rl` command shows how many nodes are down for repair.
* The health status page gives an overview of what the node(s) in your job are doing
* Check e.g. memory usage, user, system, and wait CPU utilization, disk usage, etc
* See summary of CPU and memory utilization (only available after job completes): `seff JOBID`
* System status information for each resource is available through the C3SE homepage:
* Current health status:
  * <http://url.c3se.chalmers.se/ganglia-web/?c=Hebbe>


# Things to keep in mind
* Never run (big or long) jobs on the login node! If you do, we will kill the processes.
  If you keep doing it, we'll throw you out and block you from logging in for a while!
  Prepare your job, do tests and check that everything's OK before submitting the job, but don't run the job there!
* Keep an eye on what's going on - use normal Linux tools on the login node and on the allocated nodes to check CPU, memory and network usage, etc. Especially for new jobscripts/codes!
* Think about what you do - if you by mistake copy very large files back and forth you can slow the storage servers or network to a crawl

# Getting support
* We provide support to our users, but not for any and all problems
* We can help you with software installation issues, and recommend compiler flags etc. for optimal performance
* We can install software system-wide if there are many users who need it - but not for one user (unless the installation is simple)
* We don't support your application software or help debugging your code/model or prepare your input files

# Getting support
* C3SE staff are available in our offices, to help with those things that are hard to put into a support request email (book a time in advance please)
* Rooms O5105B, O5110 and O5111 Origo building - Fysikgården 1, one floor up, ring the bell to the right
* We also offer advanced support for things like performance optimization, advanced help with software development tools or debuggers, workflow automation through scripting, etc.

# Getting support - support requests
* If you run into trouble, first figure out what seems to go wrong. Use the following as a checklist:
  * something wrong with your job script or input file?
  * does your simulation diverge?
  * is there a bug in the program? 
  * any error messages? Look in your manuals, and use Google!
  * check the node health: Did you over-allocate memory until linux killed the program?
  * Try to isolate the problem - does it go away if you run a smaller job? does it go away if you use your home directory instead of the local disk on the node?
  * Try to create a test case - the smallest and simplest possible case that reproduces the problem

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

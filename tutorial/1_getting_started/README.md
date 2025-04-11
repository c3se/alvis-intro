# Getting started
This part contains information about how to get up and running on the Alvis
system, if you have used other HPC systems much of it will be familiar.

## Accessing Alvis
Now that you've [gotten
access](https://www.c3se.chalmers.se/documentation/first_time_users/), are
getting started
and have attended the [introduction presentation for
Alvis](https://www.c3se.chalmers.se/documentation/first_time_users/intro-alvis/slides/) you are
certainly itching to access Alvis and start doing stuff.

To access Alvis there are a few different alternatives and they can all be found
at [c3se.chalmers.se](https://www.c3se.chalmers.se):
 - [Connecting through terminal](https://www.c3se.chalmers.se/documentation/connecting/)
 - [Alvis OnDemand](https://www.c3se.chalmers.se/documentation/connecting/ondemand/)
 - [Remote graphics](https://www.c3se.chalmers.se/documentation/connecting/thinlinc/)
 - [Remote development with Visual Studio Code](https://www.c3se.chalmers.se/documentation/software/development/vscode/)

### Exercises
1. Access Alvis (and open up a terminal)

## Using Alvis
In this part we will explore how to get started with using Alvis as a first
time user. As a first step you should connect to Alvis (see previous section)
and now you should have access to Alvis through a terminal. If you're using the
portal you can get a terminal on the log-in node through "File" > "Alvis Shell
Access". Note that you could use either alvis1 log-in node or alvis2 log-in
node.

A note on the two log-in nodes is that alvis1 has 4 T4 GPUs that can be used for
light testing while alvis2 is the dedicated data transfer node for when you want
to transfer datasets or similar to and from the system.

If there are any commands that you are unsure of what they do you can use the
command `man`, e.g to find out about `ls` do
```bash
man ls
```
or
```bash
ls --help
ls -h
```

If you want to abort a command pressing <kbd>Ctrl</kbd>+<kbd>C</kbd> is usually
the way to go (to copy use <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>C</kbd>).
Though, to quit from `man` or `less` you use <kbd>Q</kbd>.

Note: If you want a more thorough introduction to the command line and computer
clusters we can recommend the self-paced course
[Practical Intro to Computer CLusters](https://chalmers.instructure.com/courses/21205/).
You access the course with the same username and password as you have on Alvis.

### Get tutorial files
Now in this terminal we want to get this tutorial, either clone the repository
```bash
[USER@alvis2 ~]$ git clone https://github.com/c3se/alvis-intro.git
```
or download it as an archive
```bash
[USER@alvis2 ~]$ wget https://github.com/c3se/alvis-intro/archive/refs/heads/main.zip
[USER@alvis2 ~]$ unzip main.zip
[USER@alvis2 ~]$ mv alvis-intro-main alvis-intro
```

Now lets move to this file (README.md)
```bash
[USER@alvis2 ~]$ cd alvis-intro/tutorial/1_getting_started
[USER@alvis2 1_getting_started]$ ls
README.md  jobscript.sh		jobscript_container.sh
hello.sh   jobscript_module.sh
```

To read this file you can use your favourite command line text editor (`nano`,
`vim`, ...) or favourite file reader (`cat`, `less`, ...)
```bash
[USER@alvis2 1_getting_started]$ less README.md
```
to exit `less` press <kbd>q</kbd>.

#### Exercises
1. On Alvis, clone or download the alvis-intro repository
2. Change directory to the one containing this text

### Submitting a job
You want to make sure that there are no obvious errors before you submit it, to
do this it is absolutely OK to run small tests directly on the log-in nodes.

Take a look at the contents of `hello.sh` and if you think that it looks ok you
can try running it directly on the log-in node.
```bash
bash hello.sh
```

Usually you wouldn't run everything that you are going to submit on the log-in
node, what you could usually do is reduce it as much as possible such as:

 - number of epochs,
 - size of the dataset,
 - number of GPUs and CPU cores used, etc.

That way you have an idea of if it will run as expected before submitting the
entire job to the queue.

Now there are three remaining things to determine before we submit our script:
- The name of your project
- What GPUs to allocate
- How long the script is expected to run

#### Project name
To determine the name of your project use `projinfo`, e.g.
```bash
[USER@alvis2 1_getting_started]$ projinfo
 Project                Used[h]         Allocated[h]      Queue
    User
---------------------------------------------------------------
NAISS2024-X-YY             12.18                 3500      alvis
    USER                   6.25
```
in this case the project is `NAISSYYYY-RR-XXXX`. If you are part of the
Introduction to Alvis workshop project then the project is `NAISS2024-22-219`.

#### Deciding GPU type
When deciding what GPUs to allocate the main consideration is what demands the
application has, the secondary is what GPUs are available right now and the
price for GPUs should usually only be considered last if at all.

When it comes to performence these are a few considerations:

- GPU memory is hard limit when it comes to the largest models and would
either need GPUs with a lot of VRAM or use model parallelism,
- When using large datasets A100 nodes are recommended as they have fast
connections to Mimer,
- When using multiple nodes A40 nodes should be avoided as they are not
connected with Infiniband,
- T4s are technically built for inference and does not perform as well for
training, but given their low cost it may still be cost-effective to use them
for light tasks,
- Depending on which floating point precision used different GPUs can have
very different performance, see
[GPU Hardware Details](https://www.c3se.chalmers.se/about/Alvis/#gpu-hardware-details),
- When there are multiple feasible options, go for the currently most abundant
GPU (`jobinfo -s`).

The script `hello.sh` does not have any constraints on what GPU to use (in fact
it does not use a GPU and does not technically belong on Alvis, but you can still
learn the principles SLURM from it). Therefore, we should try to see what GPU type is
most available right now to reduce how long we have to wait. This we can do with
the command `jobinfo` e.g.
```
[USER@alvis2 1_getting_started]$ jobinfo -s
CLUSTER: alvis

Summary: 93 running jobs using 26 nodes, 1 waiting normal jobs wanting <= 1 nodes

Total node usage:
PARTITION        ALLOCATED       IDLE    OFFLINE      TOTAL
alvis                   26        152         31        209
chair                    0          8          0          8

Total GPU usage:
TYPE   ALLOCATED IDLE OFFLINE TOTAL
T4            96   56       0   160
A40            0  260      60   348
V100          14   28       2    44
A100           4  260      44   308
A100fat         1   31       0    32

Free nodes per number of GPU:s:
PARTITION  # NODES  GPU:s
alvis            1  A100:1 
alvis            1  A100:3 
alvis           73  A100:4 
alvis            1  A100fat:3
alvis            7  A100fat:4
alvis           85  A40:4  
alvis            2  T4:1   
alvis            4  T4:2   
alvis            2  T4:3   
alvis            6  T4:8   
alvis            4  V100:1 
alvis            5  V100:2 
alvis            4  V100:4 
chair            2  A100:4 
chair            2  A40:4
```
from this we can see that we have a lot of idle A40s and A100s, thus we could
probably choose either one.

#### The time it takes
When choosing how long timespan to allocate, one should estimate an upper bound
for how long the job will take. If the job does not finish within the allocated
time everything will be lost (unless you are using checkpointing). On the other
hand you might have to wait longer in the queue if you are allocating an
unneccessarily long time.

In this case the script is pretty much instantaneous so one minute upper bound
should be enough. The maximum time you can allocate is seven days.

#### Interactive session
Now to submit our job interactively we will use `srun`.
```bash
[USER@alvis2 1_getting_started]$ srun -A NAISS2024-X-YY --gpus-per-node=A40:1 -t 00:01:00 --pty bash
srun: job 102893 queued and waiting for resources
srun: job 102893 has been allocated resources
[USER@alvisX-Y 1_getting_started]$ bash hello.sh
Hello Alvis!
[USER@alvisX-Y 1_getting_started]$ exit
```
Here an interactive (pseudo)-terminal was started by adding the flag `--pty` and
note that `exit` or <kbd>Ctrl</kbd>+<kbd>D</kbd> is used to end the session.


#### Monitoring session
It is important to take alook at the job run statistics. Before checking the status of a job, 
you first need to find your job ID. To do this, run:
```bash
[USER@alvis2 1_getting_started]$ squeue --me
```
This command shows your current jobs and their job IDs. 
if nothing shows up, the most likely reason is that your job has already
finished. To also see accounting information from finished submissions use
```bash
[USER@alvis2 1_getting_started]$ sacct
```
Once you have the job ID, you can check the run statistics to see how the job has performed. 
This can provide hints if something has gone wrong or is running inefficiently. 
To see these statistics, run (replacing with your job ID):
```bash
[USER@alvis2 1_getting_started]$ job_stats.py 102893
```
Then, visit the link that appears after running the above command:
```bash
https://scruffy.c3se.chalmers.se/d/alvis-job/alvis-job?var-jobid=102893&from=1632492049000&to=1632492074000
```
In this example the program was too fast to get any data points, but what you would
have would be that there was little to no load on GPU, CPU and memory. Which for
most applications is a sign that something is wrong. Though, in our case it is a
sign that we probably did not need a supercomputer to run it.

#### Submitting a jobscript
Submitting jobscripts is done with `sbatch` and an example jobscript can be
found in `jobscript.sh`

There are three parts to a successful jobscripts
1. A shebang at the very start of the script, usually `#!/bin/env bash`.
2. Specifying flags to sbatch. Either directly when calling sbatch or in the
jobscript as `#SBATCH --flag-name value`.
3. The body of the script, this is where stuff happens.
    1. Setting up the environment e.g. loading modules.
    2. Calling what you want to run.

Now take a look at `jobscript.sh` and see that you understand what is going on.
Then, when you feel comfortable you can submit the jobscript with
```bash
[USER@alvis2 1_getting_started]$ sbatch jobscript.sh
```

Next make sure to look at how it has gone for the script using what you learnt
in the previous section.

#### Exercises
1. Find out the name of your project
2. Find out what GPU type has highest availability right now
3. Take a look at `hello.sh` and estimate how long it will take
4. Use `srun` and what you figured out in the previous steps to run `hello.sh`
5. Use the link from `job_stats.py` to take a look at the statistics of the
previous submission
6. Update `jobscript.sh` with the details you found out in 1--3
7. Submit `jobscript.sh` and look at the statistics of this job
8. If the graphs from 5 and 7 were empty. Pick a job from the queue (`squeue`)
that has been running for a while and try to tell whether this job is using the
resources okay or not.

### Setting up the environment
There are primarily two ways to set-up your environment on Alvis:
1. Modules
2. Containers

#### Loading modules
In this section we will go through the essentials for using modules to set up
your preferred software for more details, see the
[C3SE documentation](https://www.c3se.chalmers.se/documentation/module_system/modules/).

The first command we will consider is
```bash
module purge
```
this commands will unload all modules you've loaded previously. Thus, we can get
a clean slate before loading what we want.

Then to search for a module we will use `module spider`, e.g.
```bash
module spider pytorch
```
and finally following the instructions loading the wanted modules with `module
load`.

In `jobscript_module.sh` you can find how to use the module tree to load PyTorch.

**Exercises:**
1. Update `jobscript_module.sh` and submit it with `sbatch`.
2. Redo 1 but for TensorFlow instead of PyTorch
3. Install an additional small Python package beyond what is available through
   modules by following our
   [Python instructions](https://www.c3se.chalmers.se/documentation/module_system/python/)

#### Using containers
Containers are a way for you to work with with a portable and reproducible
environment for any HPC system that supports it. For more details about using
containers see the
[C3SE documentation](https://www.c3se.chalmers.se/documentation/miscellaneous/containers/).
It might be worth noting that containers are self contained and are not
influenced by what operating system you have outside the container. As such
containers that work one system many times work just as well on other systems
with minimal changes. That is, as long as the software is built compatible to
the hardware. For example, old versions of PyTorch does not recognize the A40
GPUs.

In `/apps/containers/` we provide some base containers for your use, but will
later go through how you could build your own container on Alvis.

See `jobscript_container.sh` for how to use a singularity container in a
script and to submit use
```bash
[USER@alvis2 1_getting_started]$ sbatch jobscript_container.sh
```

The container variant that is used on Alvis is called Apptainer. It is set-up
so that you can build your containers directly on the cluster.

Building containers is done with the command:
```bash
[USER@alvis2 1_getting_started]$ apptainer build <IMAGE PATH> <BUILD SPEC>
```

 - `<IMAGE PATH>` is path to the container to be created
 - `<BUILD SPEC>` is in this case the recipe for how the container is to be
   built

For inspiration we have a repository of all the containers that we provide
<https://github.com/c3se/containers>. But, it is also possible to get this
information from built containers.

As an example we will look at how to build a container with the PyTorch and
[Transformers](https://huggingface.co/docs/transformers/index) packages. From
our
[instructions](https://www.c3se.chalmers.se/documentation/miscellaneous/containers/),
we make the following guess for a recipe in a file called `my\_recipe.def`.

```Singularity
Bootstrap: localimage
From: /apps/containers/PyTorch/PyTorch-NGC-latest.sif

%post
    pip install transformers

%test
    python -c "import torch, transformers"
```

You can find more extensive instructions at
<https://apptainer.org/docs/user/main/build_a_container.html>. For now, note
that it is possible to bootstrap from localimage to make use of the containers
that we provide at /apps/containers/. The test step is a sanity check that the
build has succeeded.

We are now ready to build a container from our definition file:
```
[USER@alvis2 1_getting_started]$ apptainer build my_container.sif my_recipe.def
```

Then you can use your newly created container with
```bash
[USER@alvis2 1_getting_started]$ apptainer exec my_container.sif python my_script.py
```

**Exercises:**
1. Update and submit `jobscript_container.sh`
2. Redo 1 but for TensorFlow instead of PyTorch
3. Create your own container with a package of your choice
4. Create a new jobscript in which you use your newly installed package

# Getting started
This part contains information about how to get up and running on the Alvis
system, if you have used other HPC systems much of it will be familiar.

## Accessing Alvis
Now that you've [gotten
access](https://www.c3se.chalmers.se/documentation/getting_access/), are
[getting started](https://www.c3se.chalmers.se/documentation/getting_started/)
and have attended the [introduction presentation for
Alvis](https://www.c3se.chalmers.se/documentation/intro-alvis/slides/) you are
certainly itching to access Alvis and start doing stuff.

To access Alvis there are a few different alternatives and they can all be found
at [c3se.chalmers.se](https://www.c3se.chalmers.se):
 - [Connecting through terminal](https://www.c3se.chalmers.se/documentation/connecting/)
 - [Alvis OnDemand](https://www.c3se.chalmers.se/documentation/alvis-ondemand/)
 - [Remote graphics](https://www.c3se.chalmers.se/documentation/remote_graphics/)
 - [Remote development with Visual Studio Code](https://www.c3se.chalmers.se/documentation/remote-vscode/remote_vscode/)

### Exercises
1. Access Alvis (and open up a terminal)

## Using Alvis
In this part we will explore how to get started with using Alvis as a first time
user. As a first step you should connect to Alvis (see previous section) and now
you should have access to Alvis through a terminal or if you are using Thinlinc
or VS Code you can open a terminal. Note that you could use either alvis1 log-in
node or the new alvis2 log-in node. If you've previously used Alvis but want to
get an update about all the changes since the implementation of part 2 see this
[page](https://www.c3se.chalmers.se/news/alvis-phase-2/).

A note on the two log-in nodes is that alvis1 has 4 T4 GPUs that can be used for
light testing while alvis2 is the dedicated data transfer for when you want to
transfer datasets or similar to the system.

If there are any command that you are unsure of what it does you can use the
command `man`, e.g to find out about `ls` do
```bash
man ls
```
or
```bash
ls --help
```

If you want to abort a command pressing <kbd>Ctrl</kbd>+<kbd>C</kbd> is usually
the way to go (to copy use <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>C</kbd>).
Though to quit from `man` or `less` you use <kbd>q</kbd>.

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
README.md  jobscript.sh		jobscript_singularity.sh
hello.sh   jobscript_module.sh	runmatlab.sh
```

To read this file you can use your favourite command line text editor (`nano`,
`vim`, ...) or favourite file reader (`cat`, `less`, ...)
```bash
[USER@alvis2 1_getting_started]$ less README.md
```
to exit `less` press <kbd>q</kbd>.

#### Exercises
1. On Alvis clone or download the alvis-intro code
2. Change directory to the one containing this text

### Submitting a job
You want to make sure that there are no obvious errors before you submit it, to
do this it is absolutely OK to run small tests directly on the log-in nodes.

Take a look at the contents of `hello.sh` and if you think that it looks ok you
can try running it on the log-in node.
```bash
bash hello.sh
```

Usually you wouldn't run everything that you are going to submit on the log-in
node, what you could usually do is reduce the number of epochs and/or the size
of the dataset etc. to see that it appears to run as you'd like before submitting
the entire job.

Now there are three things to determine before we submit our script:
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
SNIC2021-X-YY             12.18                 3500      alvis
    USER                   6.25
```
in this case the project is `SNIC2021-X-YY`. If you're part of the Introduction
to Alvis workshop project then the project is `SNIC2021-7-120`.

#### Deciding GPU type
When deciding what GPUs to allocate the main consideration is what demands the
application has, the secondary is what GPUs are available right now and the
price for GPUs should usually only be considered last if at all. A note is that
the T4 GPUs are technically meant for inference only and doesn't perform as well
in training, but given their low cost it can still be cost-effective to do so.
Depending on the precision you use, different GPUs can be more or less suited for
the task, see [GPU Hardware Details](https://www.c3se.chalmers.se/documentation/intro-alvis/slides/#gpu-hardware-details).

The hard limit from a particular application that can influence what GPUs
you should choose is primarilly based on memory requirements. If your machine
learning model can fit on the GPU at the same time as a batch from your dataset,
then this GPU will probably work for your application.

The script `hello.sh` doesn't have any constraints on what GPU to use (in fact
it doesn't use a GPU and doesn't technically belong on Alvis, but you can still
learn the principles from it). Therefore, we should try to see what GPU type is
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
[USER@alvis2 1_getting_started]$ srun -A SNIC2021-X-YY --gpus-per-node=A40:1 -t 00:01:00 --pty bash
srun: job 102893 queued and waiting for resources
srun: job 102893 has been allocated resources
[USER@alvisX-Y 1_getting_started]$ bash hello.sh
Hello Alvis!
[USER@alvisX-Y 1_getting_started]$ exit
```
Here an interactive (pseudo)-terminal was started by adding the flag `--pty` and
note that `exit` or <kbd>Ctrl</kbd>+<kbd>D</kbd> is used to end the session.


#### Monitoring session
You should also make a habit of taking a look at the run statistics to see how
the job has run, this can give hints for if something has gone wrong or is
running inefficiently. To see these statistics run (but with your job ID)
```bash
[USER@alvis2 1_getting_started]$ job_stats.py 102893
https://scruffy.c3se.chalmers.se/d/alvis-job/alvis-job?var-jobid=102893&from=1632492049000&to=1632492074000
```
in this case the program was too fast to get any data points, but what you would
have would be that there was little to no load on GPU, CPU and memory. Which for
most applications is a sign that something is wrong. Though, in our case it is a
sign that we probably didn't need a supercomputer to run it.

To see the current status of your job and find out your job ID you can also run
```bash
[USER@alvis2 1_getting_started]$ squeue -u $USER
```
if nothing shows up, the most likely reason is that your job has already
finished. To also see accounting information from finished submissions use
```bash
[USER@alvis2 1_getting_started]$ sacct
```

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

### Setting up the environment
There are primarily two ways to set-up your environment on Alvis:
1. Modules
2. Containers

#### Loading modules
In this section we will go through the essentials for using modules to set up
your preferred software for more details see the [C3SE
documentation](https://www.c3se.chalmers.se/documentation/modules/).

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

In `jobsubmit_module.sh` you can find how to use the module tree to load PyTorch.

**Exercises:**
1. Update and submit `jobsubmit_module.sh` with `sbatch`.
2. Redo 1 but for TensorFlow instead of PyTorch

#### Using containers
Containers are a way for you to work with with a portable and reproducible
environment for any HPC system that supports it. For more details about using
containers see the
[C3SE documentation](https://www.c3se.chalmers.se/documentation/applications/containers/).
It might be worth noting that containers are self contained and are not
influenced by what operating system you have outside the container. As such
containers that work on phase 1 should work just as well on the phase 2 nodes.
That is, as long as the software work with the hardware. For example, old
versions of PyTorch does not recognize the A40 GPUs.

In `/apps/containers/` we provide containers for your use, but if you want to
build your own see the
[build instructions](https://www.c3se.chalmers.se/documentation/applications/containers-building/building/).

See `jobscript_singularity.sh` for how to use a singularity container in a
script and to submit use
```bash
[USER@alvis2 1_getting_started]$ sbatch jobscript_singularity.sh
```

If you'd like to do persistent changes to the environment that is available in a
container then there is a possibility to use overlays for persistent storage. We
provide ready to go overlays at `/apps/containers/overlay_<size>.img`. However,
note that you will probably not be able to use overlays while they are stored on
Mimer due to
[limitations](https://docs.sylabs.io/guides/3.8/admin-guide/installation.html?highlight=localstatedir#filesystem-support-limitations)
regarding the filesystem.

One usage for these is to complement an existing container with a few extra
packages. As an example we will look at how to add the python package Seaborn
over a PyTorch container. The steps will be as follow:
```bash
[USER@alvis2 1_getting_started]$ cp /apps/containers/overlay_1G.img seaborn.img
[USER@alvis2 1_getting_started]$ singularity shell --overlay seaborn.img /apps/containers/PyTorch/PyTorch-1.10-NGC-21.08.sif
Singularity> conda install -y seaborn
...
Singularity> exit
```
then whenever you want to use your container with the new changes you can do
something like
```bash
[USER@alvis2 1_getting_started]$ singularity exec --overlay seaborn.img /apps/containers/PyTorch/PyTorch-1.10-NGC-21.08.sif python my_script.py
```

These steps can be seen as:
1. Copy an empty overlay to your own storage
2. Open a singularity session with this overlay
3. Make the changes you want to do
4. You can now use this overlay together with the container that was used in
step 2

**Exercises:**
1. Update and submit `jobsubmit_singularity.sh`
2. Redo 1 but for TensorFlow instead of PyTorch
3. Copy an overlay image from `/apps/containers/` and use it to install a
package of your choice
4. Create a new jobscript in which you use your newly installed package

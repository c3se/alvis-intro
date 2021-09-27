# Getting started
This part contains information to how to get up and running on the Alvis system, if you have used other

## Accessing Alvis
Now that you've [gotten access](https://www.c3se.chalmers.se/documentation/getting_access/), are [getting started](https://www.c3se.chalmers.se/documentation/getting_started/) and have attended the [introduction presentation for Alvis](https://www.c3se.chalmers.se/documentation/intro-alvis/slides/) you are certainly itching to access Alvis and start doing stuff.

To access Alvis there are a few different alternatives and they can all be found at [c3se.chalmers.se](c3se.chalmers.se):
 - [Connecting through terminal](https://www.c3se.chalmers.se/documentation/connecting/)
 - [Remote graphics](https://www.c3se.chalmers.se/documentation/remote_graphics/)
 - [Remote development with Visual Studio Code](https://www.c3se.chalmers.se/documentation/remote-vscode/remote_vscode/)

## Using Alvis
In this part we will explore how to get started with using Alvis as a first time user. As a first step you should connect to Alvis (see previous section) and now you should have access to Alvis through a terminal or if you are using Thinlinc or VS Code you can open a terminal.

If there is any command that you are unsure of what it does you can use the command `man` e.g to find out about ls
```bash
man ls
```
or
```bash
ls --help
```

If you want to abort a command pressing <kbd>Ctrl</kbd>+<kbd>C</kbd> is usually the way to go (to copy use <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>C</kbd>).

### Get tutorial files
Now in this terminal we want to get this tutorial either clone the repository
```bash
[USER@alvis1 ~]$ git clone https://github.com/c3se/alvis-intro.git
```
or download it as an archive
```bash
[USER@alvis1 ~]$ wget https://github.com/c3se/alvis-intro/archive/refs/heads/main.zip
[USER@alvis1 ~]$ mv alvis-intro-main alvis-intro
```

Now lets move to this file
```bash
[USER@alvis1 ~]$ cd alvis-intro/tutorial/part1
[USER@alvis1 part1]$ ls
hello.sh  README.md
```

To read this file you can use your favourite command line text editor, `cat` or `less`
```bash
[USER@alvis1 part1]$ less README.md
```
to go out from `less` press `q`.

### Submitting a job
Before you submit a file you want to make sure that there are no obvious errors before you submit it, to do this it is absolutely OK to run small tests directly on the log-in node.

Take a look at the contents of `hello.sh` and if you think that it looks ok you can try running it on the log-in node.
```bash
bash hello.sh
```

Usually you wouldn't run everything that you are going to submit on the log-in node, what you could usually do is reduce the number of epochs and/or the size of the dataset etc. to see that it appears to run as you'd like.

Now there are three things to determine before we submit our script:
- The name of your project
- What GPUs to allocate
- How long the script is expected to run

#### Project name
To determine the name of your project use `projinfo`, e.g.
```bash
[USER@alvis1 part1]$ projinfo
 Project                Used[h]         Allocated[h]      Queue
    User
---------------------------------------------------------------
SNIC2021-X-YY             12.18                  100      alvis
    USER                    6.25   
```
in this case the project is `SNIC2021-X-YY`.

#### Deciding GPU type
When deciding what GPUs to allocate the main consideration is what demands the application has, the secondary is what GPUs are available right now and the price for GPUs should usually only be considered last if at all.

This script doesn't have any constraints on what GPU to use (in fact it doesn't use a GPU and doesn't technically belong on Alvis, but you can still learn the principles from it). Therefore, we should try to see what GPU type is most available right now to reduce how long we have to wait. This we can do with the command `jobinfo` e.g.
```
[USER@alvis1 part1]$ jobinfo -s
CLUSTER: alvis

Summary: 71 running jobs using 19 nodes, 1 waiting normal jobs wanting <= 1 nodes

Total node usage:
PARTITION        ALLOCATED       IDLE    OFFLINE      TOTAL
alvis                   19         19          0         38

Total GPU usage:
TYPE   ALLOCATED IDLE OFFLINE TOTAL
T4            52  108       0   160
A100           4    0       0     4
V100          16   28       0    44

Free nodes per number of GPU:s:
PARTITION  # NODES  GPU:s
alvis            2  T4:1   
alvis            1  T4:2   
alvis            1  T4:3   
alvis            1  T4:4   
alvis            2  T4:5   
alvis            1  T4:7   
alvis           10  T4:8   
alvis            2  V100:1 
alvis            5  V100:2 
alvis            4  V100:4
```
from this we can see that we have 108 idle T4s and 28 idle V100s, thus we could probably choose either one.

#### The time it takes
When choosing how long to allocate one should estimate an upper bound for how long the job will take. If the job does not finish within the allocated time everything will be lost (unless you are using checkpointing). On the other hand you might have to wait longer if you are allocating an unneccessarily time.

In this case the script is pretty much instantaneous so one minute should be enough. The maximum time you can allocate is seven days.

#### Interactive session
Now to submit our job interactively we will use `srun`.
```bash
[USER@alvis1 part1]$ srun -A SNIC2021-X-YY --gpus-per-node=T4:1 -t 00:01:00 --pty bash
srun: job 102893 queued and waiting for resources
srun: job 102893 has been allocated resources
[USER@alvisX-Y part1]$ bash hello.sh
Hello Alvis!
[USER@alvisX-Y part1]$ exit
```
Here an interactive (pseudo)-terminal was started by adding the flag `--pty` and note that `exit` or <kbd>Ctrl</kbd>+<kbd>D</kbd> is used to end the session.


#### Monitoring session
You should also make a habit of taking a look at the run statistics to see how the job has run, this can give hints for if something has gone wrong or is running inefficiently. To see these statistics run (but with your job ID)
```bash
[USER@alvis1 part1]$ job_stats.py 102893
https://scruffy.c3se.chalmers.se/d/alvis-job/alvis-job?var-jobid=102893&from=1632492049000&to=1632492074000
```

To see the current status of your job and find out your job ID you can also run
```bash
[USER@alvis1 part1]$ squeue -u $USER
```
if nothing shows up, the most likely reason is that your job has already finished. To also see accounting information from finished submissions use
```bash
[USER@alvis1 part1]$ sacct
```

#### Submitting a jobscript
Submitting jobscripts is done with `sbatch` and an example jobscript can be found in `jobscript.sh`

There are three parts to a successful jobscripts
1. A shebang at the very start of the script, usually `#!/bin/env bash`.
2. Specifying flags to sbatch. Either directly when calling sbatch or in the jobscript as `#SBATCH --flat-name value`.
3. The body of the script, this is where stuff happens.
    1. Setting up the environment e.g. loading modules.
    2. Calling what you want to run.

Now take a look at `jobscript.sh` and see that you understand what is going on. Then, when you feel comfortable you can submit the jobscript with
```bash
[USER@alvis1 part1]$ sbatch jobscript.sh
```

Next make sure to look at how it has gone for the script using what you learnt in the previous section.

### Setting up the environment
There are primarily two ways to set-up your environment on Alvis:
1. Modules
2. Containers

#### Loading modules
In this section we will go through the essentials for using modules to set up your preferred software for more details see the [C3SE documentation](https://www.c3se.chalmers.se/documentation/modules/).

The first command we will consider is
```bash
module purge
```
this commands will unload all modules you've loaded previously. Thus, we can get a clean slate before loading what we want.

Then to search for a module we will use `module spider`, e.g.
```bash
module spider pytorch
```
and finally following the instructions loading the wanted modules with `module load`.

There is one more thing that might be of interest and that is the existence of both flat and hierarchical module trees to switch between them use:
```bash
flat_modules
```
and
```bash
hierarchical_modules
```

There are two jobscripts `jobsubmit_flat_modules.sh` and `jobsubmit_hierarchical_modules.sh` take a look at them and see how you will use the two different module structures to load your software.
```bash
[USER@alvis1 part1]$ flat_modules
[USER@alvis1 part1]$ sbatch jobscript_flat_modules.sh
```
and
```bash
[USER@alvis1 part1]$ hierarchical_modules
[USER@alvis1 part1]$ sbatch jobscript_hierarchical_modules.sh
```

#### Using containers
Containers are a way for you to work with with a portable and reproducible environment for any HPC system that supports it. For more details about using containers see the [C3SE documentation](https://www.c3se.chalmers.se/documentation/applications/containers/).

In `/apps/containers/` we provide containers for your use, but if you want to build your own see the [build instructions](https://www.c3se.chalmers.se/documentation/applications/containers-building/building/).

See `jobscript_singularity.sh` for how to use a singularity container in a script and to submit use
```bash
[USER@alvis1 part1]$ sbatch jobscript_singularity.sh
```

If you'd like to do persistent changes to the environment that is available in a container then there is a possibility to use overlays for persistent storage. We provide ready to go overlays at `/apps/containers/overlay_<size>.img`.

One usage for these is to complement an existing container with a few extra packages. As an example we will look at how to add the python package Seaborn over a PyTorch container. The steps will be as follow:
```bash
[USER@alvis1 part1]$ cp /apps/containers/overlay_1G.img seaborn.img
[USER@alvis1 part1]$ singularity shell --overlay seaborn.img /apps/containers/PyTorch/PyTorch-1.10-NGC-21.08.sif
Singularity> conda install -y seaborn
...
Singularity> exit
```

These steps can be seen as:
1. Copy an empty overlay to your own storage
2. Open a singularity session with this overlay
3. Make the changes you want to do
4. You can now use this overlay together with the container that was used in step 2
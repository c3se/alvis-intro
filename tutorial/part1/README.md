# Getting started
This part contains

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
[CID@alvis1 ~]$ git clone https://github.com/c3se/alvis-intro.git
```
or download it as an archive
```bash
[CID@alvis1 ~]$ wget https://github.com/c3se/alvis-intro/archive/refs/heads/main.zip
[CID@alvis1 ~]$ mv alvis-intro-main alvis-intro
```

Now lets move to this file
```bash
[CID@alvis1 ~]$ cd alvis-intro/tutorial/part1
[CID@alvis1 part1]$ ls
hello.sh  README.md
```

To read this file you can use your favourite command line text editor, `cat` or `less`
```bash
[CID@alvis1 part1]$ less README.md
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
[CID@alvis1 part1]$ projinfo
 Project                Used[h]         Allocated[h]      Queue
    User
---------------------------------------------------------------
SNIC2021-X-YY             12.18                  100      alvis
    CID                    6.25   
```
in this case the project is `SNIC2021-X-YY`.

#### Deciding GPU type
When deciding what GPUs to allocate the main consideration is what demands the application has, the secondary is what GPUs are available right now and the price for GPUs should usually only be considered last if at all.

This script doesn't have any constraints on what GPU to use (in fact it doesn't use a GPU and doesn't technically belong on Alvis, but you can still learn the principles from it). Therefore, we should try to see what GPU type is most available right now to reduce how long we have to wait. This we can do with the command `jobinfo` e.g.
```
[CID@alvis1 part1]$ jobinfo -s
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
[CID@alvis1 part1]$ srun -A SNIC2021-X-YY --gpus-per-node=T4:1 -t 00:01:00 --pty bash
srun: job 102893 queued and waiting for resources
srun: job 102893 has been allocated resources
[CID@alvisX-Y part1]$ bash hello.sh
Hello Alvis!
[CID@alvisX-Y part1]$ exit
```
Here an interactive (pseudo)-terminal was started by adding the flag `--pty` and note that `exit` or <kbd>Ctrl</kbd>+<kbd>D</kbd> is used to end the session.

You should also make a habit of taking a look at the run statistics to see how the job has run, this can give hints for if something has gone wrong or is running inefficiently. To see these statistics run
```bash
[CID@alvis1 part1]$ job_stats.py 102893
https://scruffy.c3se.chalmers.se/d/alvis-job/alvis-job?var-jobid=102893&from=1632492049000&to=1632492074000
```

#### Submitting a jobscript
Submitting jobscripts is done with `sbatch` and an example jobscript can be found in `jobscript.sh`

%TODO continue from here
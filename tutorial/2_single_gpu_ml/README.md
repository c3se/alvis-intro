# Introduction to ML on Alvis
In this part we will implement a very simple regression model on a very simple
dataset using a single GPU. See below for specific instructions
for each language or library.

For this tutorial there is a MATLAB track for those that mainly work with
MATLAB, however, most of the tutorials are only Python based currently. If you
have a language you think should be included you can create an issue on this
GitHub repository and pull requests are welcome. With that said, if you want to
learn the different lessons in the different parts of the tutorial it might be
wise to get started with the Python version of the steps here as well.

## Opening a Jupyter Notebook
In this part you will have the possibility to use a Jupyter Notebook to follow
along some of the examples and excersices.

You can find some details on how to do this in the [C3SE
documentation](https://www.c3se.chalmers.se/documentation/software/frequently_used/jupyter/).

If you are using the Alvis OnDemand portal running jupyter notebooks is as easy as
1. Go to https://portal.c3se.chalmers.se and authenticate through supr
2. Select "Interactive Apps"
3. Click on "Jupyter"
4. Fill in details about the run as when writing a job script

To load user specified modules and/or containers see `/apps/jupyter` for how to
create a `jupyter1.sh` in your home directory. Note that the jupyter app in the
portal is always using compute nodes.

## PyTorch
For the following excercises you will need to load the following modules:
```bash
module purge
ml PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
ml matplotlib/3.5.2-foss-2022a
ml JupyterLab/3.5.0-GCCcore-11.3.0
```
if you are using the portal, copy alvis-intro-23.04.sh to a
specific path in your homedirectory for it to be available in the portal
drop-down menu
```bash
mkdir -p ~/portal/jupyter
cp -i alvis-intro-23.04.sh ~/portal/jupyter/
```
Then select it in the portal.


Now you should open up `regression-pytorch.ipynb` and follow the instructions there.

## TensorFlow
For the following excercises you will need to load the following modules:
```bash
module purge
ml TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
ml matplotlib/3.5.2-foss-2022a
ml JupyterLab/3.5.0-GCCcore-11.3.0
```
if you are using the portal, copy alvis-intro-23.04.sh to a
specific path in your homedirectory for it to be available in the portal
drop-down menu
```bash
mkdir -p ~/portal/jupyter
cp -i alvis-intro-23.04.sh ~/portal/jupyter/
```
Then select it in the portal.

Now you should open up `regression-tensorflow.ipynb` and follow the instructions there.

## MATLAB
If you want to use the graphical version of MATLAB you can use the Alvis
OnDemand portal and the MATLAB proxy app. Otherwise you can probably get the
key takeaways from following along with your favourite command line text
editor.

Generally you can use Thinlinc on the log-in nodes while you write your code
and if you want to do some light testing of your code.

To open the MATLAB GUI open your desktop environment (portal or Thinlinc) and
then in the upper left corner you can click to get a dropdown menu and under
C3SE you can find MATLAB which will open the latest available MATLAB version on
that machine.

In `regression.m` there is an example script that currently runs on CPUs, we
will build on that to see what we can do.

### Training on GPU
To make the trainNetwork function to use the GPU there are two things that are
needed as is:
1. An available GPU
2. Input data that is located on the GPU in question

Matlab makes use of `gpuArray` objects to run on the GPU and then many things
will just work with GPU accelaration. Documentation about `gpuArray`s are
available at <https://se.mathworks.com/help/parallel-computing/gpuarray.html>.

Examples:
```matlab
a = [1, 2]
a_gpu = gpuArray(a)
b_gpu = zeros(2, 4, 'gpuArray')
```

### MATLAB excercises
1.  Open MATLAB and navigate to this folder and open `regression.m`.
2.  Read through the script and see how it is structured.
3.  If it looks ok you can run it.
4.  Note that `trainingOptions(...)` contains most of the details with regards
    to training.
5.  Take a short while and explore the possibilities that exist in
    trainingOptions.
6.  Try running with a few different options and hyperparameters to see how it
    does.
7.  Increase dataset size to 1 million data points and make the training use
    the GPU. Then submit a job with `sbatch jobscript-matlab.sh` after
    investigating the details of the jobscript.
7.  Use `sacct` to find the job-id of your latest job.
8.  Issue `job_stats.py JOBID` where JOBID is the job-id that you found with
    `sacct`.
9.  Does your script use the GPU well?
10. If you want more frequent information on how your job is doing than what
    you get through the Grafana page:
    1. Use `squeue -u $USER` to see what node your job is running on.
    2. SSH to that node e.g. `ssh alvis7-02`.
    3. Load nvtop: `ml nvtop`.
    4. Run `nvtop` too see current GPU usage.

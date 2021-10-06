# Introduction to PyTorch on Alvis
This will introduce the very basics of using PyTorch on Alvis.

The dataset we will construct ourselves and the model will be a simple linear
regression model.

## Set-up
For the following excercises you will need to load the following modules:
```bash
flat_modules
ml PyTorch/1.9.0-fosscuda-2020b matplotlib/3.3.3-fosscuda-2020b JupyterLab/2.2.8-GCCcore-10.2.0
```

## Introduction
In this part simply get yourself acquainted with the existing examples
`regression.py` and `regression.ipynb`. Note that the notebook may not be very
readable in its raw text format, it is more easily investigated running a jupyter
notebook.

### Excercises
1. Run `python regression.py`
2. Run `jupyter lab` or `jupyter notebook` and go through `regression.ipynb`

## Your own model
In PyTorch the main way to construct a neural network model is by inheriting
from PyTorch
[Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).
In many cases it is enough to implement a forward method, this is what you will do now.

### Excercises
1. Modify `LinearModel` by using a fixed bias, this can be done in several
different ways. Depending on your approach you might want to take a look at the
options for the
[Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)
layer.

## Running on a single GPU
For this example you will need access to a GPU, on Alvis there are four T4 GPUs
available on the login node, to see their status you can use the command
`nvidia-smi`. If they seem to be available then you can go ahead and use one of
them for the following excercises, otherwise you will have to submit a job.

If you are going to submit a job you can modify the `jobscript.sh` file, if you
have forgotten what to think about when constructing a job script you can take a
look at part 1 and/or the introduction slides.



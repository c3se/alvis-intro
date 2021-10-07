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
In many cases it is enough to implement a forward method, this is what you will
do now.

As a side note, if are doing computations with tensors that you are not planning
to perform backpropagation or differentiation over then you can detach them from
the current graph with
```python
my_free_tensor = my_tensor.detach()
```
or simply specify that they do not require gradients directly
```python
# For specific tensor
my_tensor.requires_grad = False
# For an entire context
with torch.no_grad():
    validation_accuracy = (validation_labels == predicted_labels).float().mean()
```
This will reduce the load of these computations.



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

Now for the actual coding. In PyTorch the way to move computations to the GPU is
to move the objects that are part of the computation to the GPU. First create a
variable for the device you want to use
```python
dev = torch.device("cuda:0") 
```
you can change the zero to any other GPU that is available. Note that even if
you only have access to a part of a node the GPUs you have access to will still
always start from 0.

The second step is to move the data and model to the GPU this can be done by
calling
```pytorch
x_gpu = x.to(dev)
y_gpu = y.to(dev)
model = model.to(dev)
```
note that you can't use tensors on the GPU to plot with, for these you will have to send them to CPU first 

### Excercises
1. Use `nvidia-smi` to find out about current GPU usage
2. Decide if you will do the following excercises on the log-in node or if you
will submit a job
3. Modify `regression.py` or `regression.ipynb`
4. When you think you've succeded submit it with the jobscript.sh
5. Use `sacct` to find the job ID and then run `job_stats.py`
6. TODO more data see difference also above and below

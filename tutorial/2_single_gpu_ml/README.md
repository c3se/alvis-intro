# Introduction to ML on Alvis
In this part we will implement a very simple regression model on a very simple
dataset using a single GPU. See below for specific instructions
for each language or library.

## Opening a Jupyter Notebook
In this part you will have the possibility to use a Jupyter Notebook to follow
along some of the examples and excersices.

You can find some details on how to do this in the [C3SE
documentation](https://www.c3se.chalmers.se/documentation/applications/jupyter/).

If you are using the Alvis OnDemand portal running jupyter notebooks is as easy as
1. Go to https://portal.c3se.chalmers.se and log-in through supr
2. Select "Interactive Apps"
3. Click on "Jupyter"
4. Fill in details about the run as when writing a job script

To load user specified modules and/or containers see `/apps/jupyter` for how to create a `jupyter1.sh` in your home directory. Note that the jupyter app in the portal is always using compute nodes.

## PyTorch
For the following excercises you will need to load the following modules:
```bash
flat_modules
ml PyTorch/1.8.1-fosscuda-2020b matplotlib/3.3.3-fosscuda-2020b JupyterLab/2.2.8-GCCcore-10.2.0
```
if you are using the portal copy jupyter3.sh to your home directory
```
cp -i jupyter3.sh ~
```
and select it in the portal.


Now you should open up `regression-pytorch.ipynb` and follow the instructions there.

## TensorFlow
For the following excercises you will need to load the following modules:
```bash
flat_modules
ml TensorFlow/2.5.0-fosscuda-2020b matplotlib/3.3.3-fosscuda-2020b JupyterLab/2.2.8-GCCcore-10.2.0
```
if you are using the portal copy jupyter3.sh to your home directory
```
cp -i jupyter3.sh ~
```
and select it in the portal.

Now you should open up `regression-tensorflow.ipynb` and follow the instructions there.
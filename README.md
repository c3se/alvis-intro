# Alvis introduction
This repository contains examples and excercises for how to run ML applications on the
SNIC Alvis AI/ML HPC system. The aim is to get let users get their feet wet
quickly with something that they can adapt into their own jobs.

The examples are written in Python with the frameworks TensorFlow and PyTorch. More
frameworks can be added if interest exists.

Accompaning slides are available at:
 * <https://www.c3se.chalmers.se/documentation/intro-alvis/presentation.html>
 * <https://www.c3se.chalmers.se/documentation/intro-alvis/slides>

The accompanying slides for these problems can be found 
***TODO***
## Tutorial overview
Doing the excercises are voluntary, but make sure to read the associated READMEs for
each part to make sure that you're not missing something.

1. Connecting and submitting jobs
2. ***TODO*** A simple ML example on the GPU
3. ***TODO*** Loading data: provided, your own and from the web
4. ***TODO*** Checkpointing
5. ***TODO*** Profiling w/ TensorBoard
6. ***TODO*** Scaling beyond a single node w/ Horovod

## A note on notebooks
Large parts of this tutorial are written in Jupyter Notebooks, this fileformat is good for teaching and early development. However, when submitting jobs or handling a larger code-base it might be more convenient to use python files. To convert a notebook to python file use
```bash
jupyter nbconvert --to script my_notebook.ipynb
```
or on older versions of jupyter
```bash
jupyter nbconvert --to python my_notebook.ipynb
```
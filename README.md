# Alvis introduction
This repository contains examples and excercises for how to run ML applications on the
SNIC Alvis AI/ML HPC system. The aim is to let users get their feet wet
quickly with something that they can adapt into their own jobs.

The examples are written in Python with the frameworks TensorFlow and PyTorch. More
frameworks can be added if interest exists.

Accompanying introduction slides are available at:
 * <https://www.c3se.chalmers.se/documentation/intro-alvis/presentation.html>
 * <https://www.c3se.chalmers.se/documentation/intro-alvis/slides>

The accompanying slides for these workshop problems can be found 
***TODO***

## Tutorial overview
Doing the excercises are voluntary, but make sure to read the associated READMEs
for each part to make sure that you're not missing something. The tutorial is
located under
[alvis-intro/tutorial/](https://github.com/c3se/alvis-intro/tree/main/tutorial)
there are also some examples under
[alvis-intro/examples/](https://github.com/c3se/alvis-intro/tree/main/examples)
with similar content but can in some cases give a new perspective.

1. Connecting and submitting jobs
2. A simple ML example on the GPU
3. Loading data: provided, your own and from the web
4. Checkpointing
5. Profiling w/ TensorBoard
6. Single node, multiple GPUs
7. Multiple nodes, multiple GPUs

## A note on notebooks
Large parts of this tutorial are written in Jupyter Notebooks, this fileformat
is good for teaching and early development. However, when submitting jobs or
handling a larger code-base it might be more convenient to use python files. To
convert a notebook to python file use
```bash
jupyter nbconvert --to script my_notebook.ipynb
```
or with older versions of jupyter
```bash
jupyter nbconvert --to python my_notebook.ipynb
```
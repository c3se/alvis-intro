# Alvis introduction
This repository contains examples and excercises for how to run ML applications on the
SNIC Alvis AI/ML HPC system. The aim is to let users get their feet wet
quickly with something that they can adapt into their own jobs.

The examples are written in Python with the frameworks TensorFlow and PyTorch. More
frameworks can be added if interest exists. You can express interest by creating an
issue for this repository or by contacting support.

Accompanying introduction slides are available at:
 * <https://www.c3se.chalmers.se/documentation/for_users/intro-alvis/slides>

## Tutorial overview
Doing the excercises are voluntary, but make sure to read the associated READMEs
for each part to make sure that you're not missing something. The tutorial is
located in the
[tutorial/](https://github.com/c3se/alvis-intro/tree/main/tutorial) folder.

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

The automatic conversion can be used as a start to refactor the code from the
notebook into more suitable format.

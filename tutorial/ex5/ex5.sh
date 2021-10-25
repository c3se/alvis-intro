#!/bin/env bash
#SBATCH -A C3SE-STAFF -p alvis # find your project with the "projinfo" command
#SBATCH -t 0-00:05:00
#SBATCH -J tensorboard_example
#SBATCH --gpus-per-node=T4:1

ml GCC/8.3.0 \
   CUDA/10.1.243 \
   OpenMPI/3.1.4 \
   TensorFlow/2.3.1-Python-3.7.4
 
# Download MNIST in pickled-format 
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz -P $TMPDIR

export MNIST_DIR=$TMPDIR
python ex5.py

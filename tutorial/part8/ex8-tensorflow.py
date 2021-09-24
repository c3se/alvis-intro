#!/usr/bin/env python
# This example aims to illustrate how to use the TensorBoard profiling callback
# for later visualizations using TensorBoard.
# ---
# Requirements:
#   1. At least 1 GPU.
#   2. Local access to the MNIST-dataset in the "pickled format" mnist.npz.
#   3. TensorBoard version 2.2 or later including the tensorboard_plugin_profile
# ---
# Preparations:
#   2) Download the MNIST-dataset by running:
#      $ wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
#   3) Download the TensorBoard Profile Plugin
#      $ pip install --upgrade --user tensorboard_plugin_profile==<tensorflow_version>
#      where <tensorflow_version> is printed during execution (see code below)
#      Note that if the exact version is missing you can try the major release
#      e.g. replace tensorboard_plugin_profile==2.3.1 with tensorboard_plugin_profile==2.3.0.
import os
import sys
import argparse

from datetime import datetime
from packaging import version

import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard

parser = argparse.ArgumentParser(description='Alvis Example 7 with Horovod',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', type=int, default=80,
                    help='number of epochs to train')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=8,
                    help='input batch size for validation')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')

args = parser.parse_args()


# Initialize Horovod.
hvd.init()

# Horovod: print logs only on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) < 1:
  print("This example requires at least 1 GPU.")

for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


dataset = "mnist.npz"
if not os.path.exists(dataset):
  raise FileNotFoundError("This example requires {} to be in the working directory".format(dataset))

# Load, normalize and batch
(x_train, y_train), (x_test, y_test) = mnist.load_data(path=dataset)
(x_train, x_test) = (x_train/255.0, x_test/255.0)
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.val_batch_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(args.batch_size, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])


# Horovod: add Horovod Distributed Optimizer.
opt = tf.optimizers.Adam(lr=args.base_lr * hvd.size())
opt = hvd.DistributedOptimizer(opt)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy']
)

# Create a TensorBoard callback
logs = "logs/tensorflow-" + datetime.now().strftime("%Y%m%d-%H%M%S")


callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    hvd.keras.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.keras.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose),

    # Horovod: after the warmup reduce learning rate by 10 on the 15th, 25th and 35th epochs.
    hvd.keras.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, end_epoch=15, multiplier=1.),
    hvd.keras.callbacks.LearningRateScheduleCallback(start_epoch=15, end_epoch=25, multiplier=1e-1),
    hvd.keras.callbacks.LearningRateScheduleCallback(start_epoch=25, end_epoch=35, multiplier=1e-2),
    hvd.keras.callbacks.LearningRateScheduleCallback(start_epoch=35, multiplier=1e-3),
]

# Horovod: enable tensorboard only on the first worker
if hvd.rank() == 0:
    # This is were the magic happens - we add the profile_batch argument to
    # TensorBoard. profile_batch accepts a non-negative integer or pair of
    # integers (a range) to select which of the batches that should get profiled.
    callbacks.append(TensorBoard(log_dir=logs,
                              histogram_freq=1,
                              profile_batch='500,520'))


# Train the model. The training will randomly sample 1 / N batches of training data and
# 3 / N batches of validation data on every worker, where N is the number of workers.
# Over-sampling of validation data, which helps to increase the probability that every
# validation example will be evaluated.
model.fit(ds_train.repeat(), # The original training set is too small for running with horovod on multiple GPUs, here we're faking a larger set by repeating it
          epochs=args.epochs,
          steps_per_epoch=len(x_train) // args.batch_size // hvd.size(), # Each horodov task should handle a part of the steps
          workers=4,
          validation_data=ds_test.repeat(),
          validation_steps=3 * len(x_test) // args.val_batch_size // hvd.size(), # Each horodov task should handle a part of the steps
          callbacks = callbacks) # <-- Remember to add the callback

if verbose:
    print("\n-- Training completed --")
    print("Profiling data generated using TensorFlow version {}".format(tf.__version__))
    print("Start tensorboard by running the following command in your shell:")
    print("$ tensorboard --logdir logs")
    print("and visit the TensorBoard URL")

# Evaluate the model on the full data set.
score = model.evaluate(ds_test, workers=4)
if verbose:
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


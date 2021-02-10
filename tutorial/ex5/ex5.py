#!/usr/bin/env python
# This example aims to illustrate how to use the TensorBoard profiling callback
# for later visualizations using TensorBoard.
# ---
# Requirements:
#   1. At least 1 GPU.
#   2. Local access to the MIST-dataset in the "pickled format" mnist.npz.
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
from datetime import datetime
from packaging import version
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard

def main():
  gpus = tf.config.list_physical_devices("GPU")
  if len(gpus) < 1:
    print("This example requires at least 1 GPU.")

  dataset = "mnist.npz"
  if not os.path.exists(dataset):
    raise FileNotFoundError("This example requires {} to be in the working directory".format(dataset))

  # Load, normalize and batch
  (x_train, y_train), (x_test, y_test) = mnist.load_data(path=dataset)
  (x_train, x_test) = (x_train/255.0, x_test/255.0)
  ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
  ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=['accuracy']
  )
  # Create a TensorBoard callback
  logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

  # This is were the magic happens - we add the profile_batch argument to
  # TensorBoard. profile_batch accepts a non-negative integer or pair of
  # integers (a range) to select which of the batches that should get profiled.
  tb_callback = TensorBoard(log_dir=logs,
                            histogram_freq=1,
                            profile_batch='500,520')
  model.fit(ds_train,
            epochs=2,
            validation_data=ds_test,
            callbacks = [tb_callback]) # <-- Remember to add the callback

  print("\n-- Training completed --")
  print("Profiling data generated using TensorFlow version {}".format(tf.__version__))
  print("Start tensorboard by running the following command in your shell:")
  print("$ tensorboard --logdir logs")
  print("and visit the TensorBoard URL")

if __name__ == "__main__":
  try:
    sys.exit(main())
  except Exception as e:
    print("Unrecoverable error: {}".format(e))

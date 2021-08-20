#!/usr/bin/env python
# This example aims to illustrate how to use the TensorBoard profiling callback
# for later visualizations using TensorBoard.
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

  dataset = os.getenv("MNIST_DIR", os.getcwd()) + "/" + "mnist.npz"
  if not os.path.exists(dataset):
    raise FileNotFoundError("This example requires {} which could not be found".format(dataset))

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
            epochs=5,
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

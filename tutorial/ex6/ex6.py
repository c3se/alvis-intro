#!/usr/bin/env python
# This example aims to illustrate how to use the TensorBoard profiling callback
# for later visualizations using TensorBoard.
# ---
# Requirements:
#   1. At least 1 GPU.
#   2. Local access to the MNIST-dataset in the "pickled format" mnist.npz.
# ---
# Preparations:
#   2) Download the MNIST-dataset by running:
#      $ wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
import os
import sys
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint

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

	# Save the model
  model.save("my_seq_fdd_model")

  my_ckpts = "training/cp-{epoch:04d}.ckpt"
  checkpoint_callback = ModelCheckpoint(
    filepath=my_ckpts,
    monitor='val_loss',
    verbose=1,
    save_best_only=False,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch',
    options=None
  )
  model.fit(ds_train,
            epochs=8,
            validation_data=ds_test,
            callbacks = [checkpoint_callback]) # <-- Remember to add the callback

  print("\n-- Training completed --")

if __name__ == "__main__":
  try:
    sys.exit(main())
  except Exception as e:
    print("Unrecoverable error: {}".format(e))

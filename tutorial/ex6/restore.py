#!/usr/bin/env python
import sys
import tensorflow as tf

def main():
  model = tf.keras.models.load_model("my_seq_fdd_model")
  latest = tf.train.latest_checkpoint("training")
  model.load_weights(latest)
  # Continue evaluating model

if __name__ == "__main__":
  try:
    sys.exit(main())
  except Exception as e:
    print("Unrecoverable error: {}".format(e))

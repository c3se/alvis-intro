#!/usr/bin/env python
import os
import sys
from datetime import datetime
from packaging import version
import tensorflow as tf

def main():
  model = tf.keras.Model("my_seq_fdd")
  latest = tf.train.latest_checkpoint("training")
  model.load_weights(latest)
  # Continue evaluation model
if __name__ == "__main__":
  try:
    sys.exit(main())
  except Exception as e:
    print("Unrecoverable error: {}".format(e))

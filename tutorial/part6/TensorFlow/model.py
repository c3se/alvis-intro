import tensorflow as tf
from tensorflow.keras import layers


class Model(tf.keras.Model):

    def __init__(self, input_shape, output_shape, verbose=False):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.Input(input_shape),
            layers.Dense(5),
            layers.Dense(5),
            layers.Dense(output_shape),
        ])
        self.verbose = verbose

    @tf.function
    def call(self, inputs):
        if self.verbose:
            print("Hello from", inputs.device, "with input shape", inputs.shape)

        return self.model(inputs)

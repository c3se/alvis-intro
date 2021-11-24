import tensorflow as tf
from tensorflow.data import Dataset


def get_random_dataset(n_samples, sample_shape):
    '''Note that this is entirely different from tf.data.experimental.RandomDataset'''
    total_shape = (n_samples,) + tuple(sample_shape)
    x = tf.random.normal(total_shape)
    y = tf.random.uniform((n_samples, 1))
    return Dataset.from_tensor_slices((x, y))

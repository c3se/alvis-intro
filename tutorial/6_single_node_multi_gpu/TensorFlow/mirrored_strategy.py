from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from model import Model
from dataset import get_random_dataset


if __name__=="__main__":
    # Initialize dataset
    input_shape = 500
    output_shape = 1
    data_size = 100

    dataset = get_random_dataset(data_size, (input_shape,))

    # Create parallalized model
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = Model(input_shape, output_shape, verbose=False)


    # TensorBoard profiling
    profiling_callback = tf.keras.callbacks.TensorBoard(
        log_dir='logs/' + datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S"),
        histogram_freq=1,
        profile_batch="15,25",
    )

    # Compile and train
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss='mse',
    )

    batch_size = 30
    global_batch_size = strategy.num_replicas_in_sync

    model.fit(dataset.batch(global_batch_size), epochs=10, callbacks=[profiling_callback])

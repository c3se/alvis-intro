from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from model import Model
from dataset import get_random_dataset


def main():
    # Initialize dataset
    input_shape = 500
    output_shape = 1
    data_size = 100

    dataset = get_random_dataset(data_size, (input_shape,))

    # Create parallalized model
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = Model(input_shape, output_shape, verbose=False)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
            loss='mse',
        )

    # TensorBoard profiling
    profiling_callback = tf.keras.callbacks.TensorBoard(
        log_dir='logs/' + datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S"),
        histogram_freq=1,
        profile_batch="15,25",
    )

    # Train
    batch_size = 30
    global_batch_size = strategy.num_replicas_in_sync

    model.fit(
        dataset.batch(global_batch_size),
        epochs=10,
        callbacks=[profiling_callback],
    )


if __name__=="__main__":
    main()

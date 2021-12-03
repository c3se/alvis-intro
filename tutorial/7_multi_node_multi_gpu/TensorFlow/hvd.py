from datetime import datetime

import tensorflow as tf
import horovod.tensorflow as hvd

from model import Model
from dataset import get_random_dataset

def main():
    # Initialize Horovod
    hvd.init()

    # Horovod: pin GPU to be used to process local rank
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) < 1:
        raise RuntimeError("No GPUs were found.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    # Initialize dataset
    input_shape = 5
    output_shape = 1
    data_size = 100
    batch_size = 30
    data_batches = get_random_dataset(data_size, (input_shape,)).batch(batch_size)
    
    # Create parallalized model
    model = Model(input_shape, output_shape, verbose=False)

    # Compile and train
    optimizer = hvd.DistributedOptimizer(
        tf.keras.optimizers.SGD(learning_rate=0.01 * hvd.size()),
    )
    model.compile(
        optimizer=optimizer,
        loss="mse",
    )

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        hvd.keras.callbacks.MetricAverageCallback(),
    ]

    model.fit(data_batches, epochs=10, callbacks=callbacks)

if __name__=="__main__":
    main()

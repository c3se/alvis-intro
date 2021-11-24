import argparse

import tensorflow as tf
from tensorflow.distribute.experimental import CommunicationImplementation

from model import Model
from dataset import get_random_dataset


parser = argparse.ArgumentParser(
    description="Alvis Tutorial: Multi Node Distributed Training with TensorFlow",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--communicator',
    default="AUTO",
    choices=["AUTO", "NCCL", "RING"],
    help='the communication implementation to use',
)
args = parser.parse_args()


if __name__=="__main__":
    # Create parallalized model
    cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
    communication_options = tf.distribute.experimental.CommunicationOptions(
        implementation=CommunicationImplementation[args.communicator],
    )
    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver,
        communication_options=communication_options,
    )

    input_shape = 5
    output_shape = 1
    with strategy.scope():
        model = Model(input_shape, output_shape, verbose=False)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01) 

    # Initialize dataset
    data_size = 100
    batch_size = 30
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    data_batches = get_random_dataset(data_size, (input_shape,)).batch(global_batch_size)


    # Compile and train
    model.compile(optimizer=optimizer, loss='mse')

    model.fit(data_batches, epochs=10)

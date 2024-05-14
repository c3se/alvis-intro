import argparse

import tensorflow as tf

from model import Model
from dataset import get_random_dataset


parser = argparse.ArgumentParser(
    description="Alvis Tutorial: Multi Node Distributed Training with TensorFlow",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--communicator',
    default="NCCL",
    choices=["NCCL"],  # NCCL is the only one that makes use of IB
    help='the communication implementation to use',
)
args = parser.parse_args()


class AlvisResolver(tf.distribute.cluster_resolver.SlurmClusterResolver):
    '''Workaround for TensorFlow bug prior to version 2.12, see:
    https://github.com/tensorflow/tensorflow/commit/66e587c780c59f6bad2ddae5c45460440002dc68'''

    def _resolve_hostlist(self):
        hosts = super()._resolve_hostlist()
        def rename(host):
            group, num = host.split('-')
            return f'{group}-{int(num):02d}'
        return [rename(host) for host in hosts]


if __name__=="__main__":
    # Create parallalized model
    cluster_resolver = AlvisResolver(port_base=12345)
    communication_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation[
            args.communicator
        ],
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

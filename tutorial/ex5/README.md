# Introduction

In this tutorial we will show how to profile a TensorFlow model using the
built-in TensorBoard profiler.  We make use of the `tensorboard` command-line
utility for visualization and the `tensorflow.keras.callbacks.TensorBoard`
API-callback to collect profiling data. This example is only to show the built-in profiler
in TensorBoard can easily be used it conjunctin with TensorFlow. The model trains the
MNIST-database, but you can of course experiment with other datasets and models
as well. Lastly, TensorBoard supports other ML-libraries as well, such as PyTorch.

## Environment setup
You need to complete a few steps before you can run this example. The environment only
needs to load the TensorFlow module as TensorBoard comes bundled with TensorFlow.
### The following modules needs to be loaded:
```
ml GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4
ml load TensorFlow/2.3.1-Python-3.7.4
```
### The following dataset needs to be available
We provide equivalent datasets in `/cephyr/NOBACKUP/Datasets/MNIST` but to
simplify the code in this example we download the dataset in pickled format.
For production use you should always check first if the dataset is already
available in `/cephyr/NOBACKUP/Datasets` before you download it. The size
for MIST should only be around 11M compressed.
```
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

### Update the TensorBoard Profiler Plugin
The current installed versions of TensorFlow provides an older version of
the TensorBoard Profiler plugin - we need to update it. The `--user` flag
implies we install a local copy of the package.
```console
pip install --upgrade --user tensorboard_plugin_profile==2.3.0
```

## Running the code
We are now ready to train and profile our model. This is a very light model and
can be trained directly on the login node, which also simplifies how you access
the TensorBoard UI. For real models you should not be running on the login nodes.

**Note: Make sure you complete all tasks in the environment setup before this step!**

## Generate the profiling data
The profiling data will be created inside a directory `logs` in you current
working directory.
```
python ex5.py
[TensorFlow output - lots of it!]
```

## Start TensorBoard
Once the profiling data has been genereated we can start TensorBoard.
The`tensorboard` command-line utility starts a web server listening on
localhost, as seen below. 
```
tensorboard --logdir logs
2021-02-09 11:54:44.785607: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.3.0 at http://localhost:6007/ (Press CTRL+C to quit)
```
In the above example we need to visit `http://localhost:6007/` on the login
node - your port may be different.

You will need to either use ThinLinc (see [Connecting with ThinLinc](https://www.c3se.chalmers.se/documentation/remote_graphics/))
and connect to the Alvis logi node, or (recommended) setup a [SSH tunnel](https://www.c3se.chalmers.se/documentation/connecting/#use-ssh-to-access-services)
to access the UI from your computer.

On the TensorBoard UI you select "Profile" in the drop-down menu next to the UPLOAD button.
![TensorBoard Profile](tb_profile.png)


# Introduction

In this tutorial we will demonstrate how to checkpoint model. The general idea
of checkpointing is to save (or *checkpoint*) the state of a running job or
application at regular time intervals to provide:
* fault-tolerance - in case something should go wrong and your
  training job gets prematurely terminated. If for example training
  takes several hours, or days, you can use checkpointing to save an
  intermediate state, lets say every couple of hours. Then if your job would crash,
  or the compute node suddenly panics due to an machine-check exception, you would
  be able to resume your job at the last checkpoint instead of starting over from
  the beginning.
* easier debugging of your model. If your model introduces strange behavior at
  some point in trainng it can be very useful to checkpoint at regular
  intervals and inspect the progress.
* models or weights for e.g. transfer learning.
* sharing and/or collaboration.

![Checkpointing illustration](checkpointing.png)

Checkpointing is not exclusive to TensorFlow (or scientific computing) but is a
common feature in HPC applications and workflows due to the common occurance of
long-running and resource heavy jobs. It can however be very tricky to restore
checkpoints taken from large distributed applications which has consumed many
external resources (some which may not be available when the checkpoint is
restored), hence checkpointing support is not universal - and we should really
look into it when it is available for us! Technically a checkpoint is one, or
more commonly, multiple files written to disk, usually in a binary or
compressed format.

## PyTorch
In PyTorch the main way to perform checkpointing is to save the state
dictionaries of the objects that are relevant to continue training. PyTorch have
written a good introduction to checkpointing which you can find
[here](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

In short, in PyTorch you can use `torch.save` to save objects. This isn't
limited to models but what is special with models is that you usually want to
save the state dictionary of the model `my_model.state_dict()`. The reverse
would be to use `torch.load` and `my_model.load_state_dict()`.

Putting this together would look something like:
```python
# Saving a model
my_model = MyModel()
torch.save(my_model.state_dict(), "my_model.pt")

# Loading a model
my_model = MyModel()
my_model.load_state_dict(torch.load("my_model.pt"))
```

There are two further caveats, when used as checkpointing you will need to save
all that is needed to continue the training. The perhaps easiest thing to do is
perform the save after you've taken a step with the optimizer but before the
next forward pass, then you do not need to keep track of the gradients. Besides
the model state dictionary you might also want to save the optimizer state
dictionary and some other variables like the epoch to know were to continue
from. 

The second caveat is that beside the checkpoint file you will also need to have
access to the class definition of your model to be able to load it properly.
Usually this means saving the state dictionary and then initialising the model
before loading the checkpointed state dictionary.

If you want save the entire model without relying on having access to the
original code, then for some use cases an alternative is to export it to
[ONNX](https://pytorch.org/docs/stable/onnx.html) or similar. 

### Set-up and excercise
To set up you'll need to load
```
flat_modules
ml PyTorch/1.8.1-fosscuda-2020b torchvision/0.9.1-fosscuda-2020b-PyTorch-1.8.1 JupyterLab/2.2.8-GCCcore-10.2.0 matplotlib/3.3.3-fosscuda-2020b
```
and you'll probably want to run it on a compute node as well, because you'll want to access the TMPDIR for faster file I/O.

To do this you can use the prepared jobscript `jobscript-pytorch.sh` or use the
Alvis OnDemand portal. If you're submitting with `sbatch`, just make sure to
open the proxy link that appears in the output file.

## TensorFlow 2 TODO
Checkpointing in TensorFlow 2.x is supported in the API-classes
`tf.train.Checkpoint`, `tf.train.CheckpointManager`, and, as this example
builds on, as a callback from `tf.keras.callbacks.ModelCheckpoint`. The
`tf.train.Checkpoint` and `tf.train.CheckpoingManger` generalizes checkpointing
and provides more options for checkpointing basically all *trackable resources*
(such as `tf.Variablel`s objects). You have the option to writing your own
checkpoints checkpointing triggers but the use of these API-calls is outside
the scope of this example, but you should know that they exist. Instead we will
look at an example of using checkpointing provided by Keras.

Checkpointing in TensorFlow is not necessarily the same as saving your model,
but as we will see, you will be able to checkpoint by saving your entire model,
if wanted. If you choose the former note that you of course need to have the
model (or an architecturally equivalent model) available in order for you to
restore and make use of the checkpoints (i.e. weights). Due to the somewhat
close relation between a checkpoint and a model it can be helpful to
have basic understanding how TensorFlow saves models.

| Format    | Description | File extension |
| --------- | ----------- |----------------|
| SaveModel | The default model format in TensorFlow 2. The SaveModel format has the most general support for saving models. The model is split into several files and directories. | N/A (both files and directories) |
| HDF5      | *Hierarchical Data Format version 5* is the default model format for standalone Keras models and was the default model format for TensorFlow 1. A model becomes a single file. In TensorFlow 2 you need to explicity use the HDF5 file extension to save your model as HDF5. | .h5 or .keras |
| TensorFlow checkpoint | A TensorBoard checkpoint file stores the model weights. | .ckpt, .ckpt.data,.ckpt.index | 

We can examine a the contents of a `my_seq_fdd_model` saved in the SaveModel format:
```
$ tree my_seq_fdd_model/
my_seq_fdd_model/
├── assets
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00001
    └── variables.index

2 directories, 3 files
```

The key difference between a HDF5 and SaveModel is according to the TensorFlow
documentation
([link](https://www.tensorflow.org/tutorials/keras/save_and_load#manually_save_weights)):
>The key difference between HDF5 and SavedModel is that HDF5 uses object
>configs to save the model architecture, while SavedModel saves the execution
>graph. Thus, SavedModels are able to save custom objects like subclassed
>models and custom layers without requiring the original code.

Automatic checkpointing can be added to your model (`model.fit` as a callback,
similar to how we added a callback for profiling in [example
4](https://github.com/c3se/alvis-intro/tree/main/tutorial/ex5).

```
my_ckpts = "training/cp-{epoch:04d}.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=my_ckpts,
    monitor='val_loss',
    verbose=1,
    save_best_only=False,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch',
    options=None,
    **kwargs
)
model = create_model()
[...]
model.fit(train_X,
          train_Y,
          epochs=128, 
          callbacks=[checkpoint_callback],
          validation_data=(test_X, test_Y),
          verbose=1)
```

We have defined a `tf.keras.callbacks.ModelCheckpoint` callback which
continously produces a checkpoint at the end of every epoch (`save_freq` can
also take an integer to save at the end of <int> many batches). The
`save_weights_only=True` only saves the model weights and not the entire model
(SaveModel-format). We must remenber to add the callback to the `callbacks`
parameter in `model.fit()`. As this is an example to illustrate checkpointing
we run with verbosity (`verbose=1`) in the calls to `model.fit()` and
`ModelCheckpoint`, for long-running production training this might not be
ideal.

### Environment setup
You need to complete a few steps before you can run this example. The environment only
needs to load the TensorFlow module as TensorBoard comes bundled with TensorFlow.
#### The following modules needs to be loaded:
```
ml GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4
ml load TensorFlow/2.3.1-Python-3.7.4
```

#### The following dataset needs to be available
We provide equivalent datasets in `/cephyr/NOBACKUP/Datasets/MNIST` but to
simplify the code we download the dataset in pickled format. The size should
only be around 11M compressed.
```
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```
### Running the code
We are now ready to train and profile our model. This is a very light model and
can be trained directly on the login node. For real models you should not be
running on the login nodes.

**Note: Make sure you complete all tasks in the environment setup before this step!**

```
$ python ex6.py
[Initialization logs - excluded for this example]
460/469 [============================>.] - ETA: 0s - loss: 0.3762 - accuracy: 0.8967
Epoch 00001: saving model to training/cp-0001.ckpt
469/469 [==============================] - 1s 2ms/step - loss: 0.3721 - accuracy: 0.8979 - val_loss: 0.2052 - val_accuracy: 0.9411
Epoch 2/8
445/469 [===========================>..] - ETA: 0s - loss: 0.1773 - accuracy: 0.9485
Epoch 00002: saving model to training/cp-0002.ckpt
469/469 [==============================] - 1s 2ms/step - loss: 0.1734 - accuracy: 0.9498 - val_loss: 0.1465 - val_accuracy: 0.9560
Epoch 3/8
450/469 [===========================>..] - ETA: 0s - loss: 0.1270 - accuracy: 0.9633
Epoch 00003: saving model to training/cp-0003.ckpt
469/469 [==============================] - 1s 2ms/step - loss: 0.1251 - accuracy: 0.9640 - val_loss: 0.1197 - val_accuracy: 0.9633
Epoch 4/8
436/469 [==========================>...] - ETA: 0s - loss: 0.0990 - accuracy: 0.9720
Epoch 00004: saving model to training/cp-0004.ckpt
469/469 [==============================] - 1s 2ms/step - loss: 0.0968 - accuracy: 0.9727 - val_loss: 0.1042 - val_accuracy: 0.9672
Epoch 5/8
445/469 [===========================>..] - ETA: 0s - loss: 0.0796 - accuracy: 0.9775
Epoch 00005: saving model to training/cp-0005.ckpt
469/469 [==============================] - 1s 2ms/step - loss: 0.0781 - accuracy: 0.9780 - val_loss: 0.0960 - val_accuracy: 0.9700
Epoch 6/8
456/469 [============================>.] - ETA: 0s - loss: 0.0653 - accuracy: 0.9816
Epoch 00006: saving model to training/cp-0006.ckpt
469/469 [==============================] - 1s 2ms/step - loss: 0.0647 - accuracy: 0.9818 - val_loss: 0.0916 - val_accuracy: 0.9717
Epoch 7/8
460/469 [============================>.] - ETA: 0s - loss: 0.0548 - accuracy: 0.9845
Epoch 00007: saving model to training/cp-0007.ckpt
469/469 [==============================] - 1s 2ms/step - loss: 0.0545 - accuracy: 0.9846 - val_loss: 0.0892 - val_accuracy: 0.9722
Epoch 8/8
460/469 [============================>.] - ETA: 0s - loss: 0.0465 - accuracy: 0.9869
Epoch 00008: saving model to training/cp-0008.ckpt
469/469 [==============================] - 1s 2ms/step - loss: 0.0462 - accuracy: 0.9870 - val_loss: 0.0874 - val_accuracy: 0.9721
```

Notice from the above run (using 8 epochs) that we get the expect single checkpoint per epoch.

If we inspect the checkpointing directory we will see that we in addition to the `.ckpt`-files got associated metadata.
```console
$ ls training/
checkpoint			  cp-0002.ckpt.data-00000-of-00001  cp-0003.ckpt.index		      cp-0005.ckpt.data-00000-of-00001	cp-0006.ckpt.index		  cp-0008.ckpt.data-00000-of-00001
cp-0001.ckpt.data-00000-of-00001  cp-0002.ckpt.index		    cp-0004.ckpt.data-00000-of-00001  cp-0005.ckpt.index		cp-0007.ckpt.data-00000-of-00001  cp-0008.ckpt.index
cp-0001.ckpt.index		  cp-0003.ckpt.data-00000-of-00001  cp-0004.ckpt.index		      cp-0006.ckpt.data-00000-of-00001	cp-0007.ckpt.index
```

#### Restoring from checkpoints
Restoring from checkpoint is as easy as loading the model and then loading the weights.
```
model = tf.keras.Model("my_seq_fdd_model")
latest = tf.train.latest_checkpoint("training")
model.load_weights(latest)
```
You can run the example `restore.py` to restore from the earlier checkpoints.

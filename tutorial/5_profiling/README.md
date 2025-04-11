# Introduction
In this tutorial, we'll show you how to profile a model and view the results using TensorBoard 
with TensorFlow and HolisticTraceAnalysis (HTA) with PyTorch.

## PyTorch
For profiling with PyTorch you'll need PyTorch 1.8.1 or higher to access the
`torch.profiler` module. To see the results of the profiling with HTA tool
you'll also need to load HolisticTraceAnalysis module.

### Environment setup
To run the code you can either use the module tree
```
module purge
ml PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
ml matplotlib/3.7.2-gfbf-2023a
ml HolisticTraceAnalysis/0.2.0-gfbf-2023a
ml JupyterLab/4.0.5-GCCcore-12.3.0

jupyter lab
```
or a provided singularity container
```
singularity exec /apps/containers/PyTorch/PyTorch-1.14-NGC-23.02.sif jupyter notebook 
```

You'll probably want to run the code on a compute node to get access to the
TMPDIR for faster file I/O. To do this you can use
`sbatch jobscript-pytorch.sh`.
Open up the jupyter server and follow the instructions in `profiling-pytorch.ipynb`

### Profiling
Once the environment is set up, profiling your PyTorch code with HTA is straightforward. Simply wrap
the code you want to profile with a profile context manager and export the trace data for HTA analysis:
```python
import torch.profiler
from hta.trace_analysis import TraceAnalysis

with torch.profiler.profile(
    on_trace_ready=torch.profiler.schedule(),
) as prof:
    # Code to profile
    #...
analyzer = TraceAnalysis(trace_dir=trace_dir)
# Code to HTA
#...
```

## Tensorflow
In this tutorial we will show how to profile a TensorFlow model using the
built-in TensorBoard profiler.  We make use of the `tensorboard` command-line
utility for visualization and the `tensorflow.keras.callbacks.TensorBoard`
API-callback to collect profiling data. This example is only to show the built-in profiler
in TensorBoard can easily be used it conjunctin with TensorFlow. The model trains the
MNIST-database, but you can of course experiment with other datasets and models
as well. Lastly, TensorBoard supports other ML-libraries as well, such as PyTorch.

### Environment setup
You need to complete a few steps before you can run this example. The environment only
needs to load the TensorFlow module as TensorBoard comes bundled with TensorFlow.

```
flat_modules
ml purge
ml TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
ml matplotlib/3.7.2-gfbf-2023a
ml JupyterLab/4.0.5-GCCcore-12.3.0
```

### Running the code
You'll probably want to run the code on a compute node to get access to the
TMPDIR for faster file I/O. To do this you can use
`sbatch jobscript-tensorflow.sh`.

Open up the jupyter server and follow the instructions in `profiling-tensorflow.ipynb`

### Generate the profiling data
The profiling data will be created inside a directory `logs` in you current
working directory.

### Start TensorBoard
You've got three alternatives for how to access TensorBoard on the cluster:

- the Alvis OnDemand portal (TensorBoard app)
- using port forwarding with SSH
- and opening it in a browser in desktop session

these instructions are currently being updated, see our [TensorBoard guide](https://www.c3se.chalmers.se/documentation/software/machine_learning/tensorboard/).

In the TensorBoard UI you select "Profile" in the drop-down menu next to the UPLOAD button.
![TensorBoard Profile](tb_profile.png)


# This script is used for the Open Ondemand portal.
# You can use it as a reference for creating a custom ~/jupyter1.sh file

module purge

ml purge
ml torchdata/0.3.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
ml torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
ml TensorFlow-Datasets/4.7.0-foss-2021a-CUDA-11.3.1
ml matplotlib/3.4.2-foss-2021a

# You need to have a version of jupyter (e.g. JupyterLab or IPython  from the module system)
ml JupyterLab/3.0.16-GCCcore-10.3.0

# You can launch jupyter notebook or lab, but you must specify the config file as below: 
jupyter lab --config="${CONFIG_FILE}"

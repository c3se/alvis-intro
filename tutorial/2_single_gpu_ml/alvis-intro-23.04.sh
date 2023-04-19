# This script is used for the Open Ondemand portal.
# You can use it as a reference for creating a custom ~/portal/jupyter/my_jupyter_env.sh file

module purge

module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load matplotlib/3.5.2-foss-2022a

# You need to have a version of jupyter (e.g. JupyterLab or IPython  from the module system)
module load JupyterLab/3.5.0-GCCcore-11.3.0

# You can launch jupyter notebook or lab, but you must specify the config file as below: 
jupyter lab --config="${CONFIG_FILE}"

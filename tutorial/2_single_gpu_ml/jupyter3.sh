# This script is used for the Open Ondemand portal.
# You can use it as a reference for creating a custom ~/jupyter1.sh file

flat_modules

ml purge
ml PyTorch/1.8.1-fosscuda-2020b TensorFlow/2.5.0-fosscuda-2020b
ml torchvision/0.9.1-fosscuda-2020b-PyTorch-1.8.1

# You need to have a version of jupyter (e.g. JupyterLab or IPython  from the module system)
ml JupyterLab/2.2.8-GCCcore-10.2.0 matplotlib/3.3.3-fosscuda-2020b Pillow/8.0.1-GCCcore-10.2.0

# You can launch jupyter notebook or lab, but you must specify the config file as below: 
jupyter lab --config="${CONFIG_FILE}"

#!/bin/bash

# Setup env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate votenet
echo "hello from $(python --version) in $(which python)"
echo "running modelnet_preprocessor.py"

# Run the experiments
python modelnet_preprocessor.py --zip-dataset-path ModelNet10.zip --num-points 256
python modelnet_preprocessor.py --zip-dataset-path ModelNet40.zip --num-points 256
python modelnet_preprocessor.py --zip-dataset-path ModelNet10.zip --num-points 1024
python modelnet_preprocessor.py --zip-dataset-path ModelNet40.zip --num-points 1024
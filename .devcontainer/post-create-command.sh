#!/usr/bin/env sh

set -e

CONDA_NUMBER_CHANNEL_NOTICES=0
CONDA_TOS_ACCEPTED=yes
export CONDA_NUMBER_CHANNEL_NOTICES CONDA_TOS_ACCEPTED

conda config --set remote_read_timeout_secs 120
conda --no-plugins run -n waft pip install --disable-pip-version-check -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
conda --no-plugins init bash

# Add conda activate waft to bashrc so it's default in every bash terminal
echo "conda activate waft" >> /home/ubuntu/.bashrc
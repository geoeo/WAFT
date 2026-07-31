#!/usr/bin/env sh

set -e

CONDA_NUMBER_CHANNEL_NOTICES=0
CONDA_TOS_ACCEPTED=yes
export CONDA_NUMBER_CHANNEL_NOTICES CONDA_TOS_ACCEPTED

conda config --set remote_read_timeout_secs 120
conda --no-plugins create -y -q --solver classic --name waft python=3.12
conda --no-plugins run -n waft pip install --disable-pip-version-check -r requirements.txt
conda --no-plugins init bash
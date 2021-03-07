#!/bin/sh

# python3 -m venv ./.env
# source ./.env/bin/activate

pip install --upgrade pip
pip install 'numpy' 'keras==2.1.6' 'tensorflow==1.13.0rc1' 'scikit-learn'
# https://stackoverflow.com/a/65717773
pip install 'h5py==2.10.0' --force-reinstall


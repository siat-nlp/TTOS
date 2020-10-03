#!/bin/bash

# set gpu id to use
export CUDA_VISIBLE_DEVICES=1

# set python path according to your actual environment
pythonpath='python3'

# set parameters
data_dir=./data/KVR
data_name=kvr
save_dir=./models-kvr
ckpt=
num_epochs=15
pre_epochs=15

${pythonpath} ./main.py --data_dir=${data_dir} --ckpt=${ckpt} --save_dir=${save_dir} --num_epochs=${num_epochs} --pre_epochs=${pre_epochs}

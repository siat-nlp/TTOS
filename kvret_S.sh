#!/bin/bash

# set gpu id to use
export CUDA_VISIBLE_DEVICES=1

# set python path according to your actual environment
pythonpath_test='python3'
pythonpath_eval='python3 -m'

# set parameters
data_name='kvr'
data_dir=./data/KVR
save_dir=./models-kvr
test_model=S
output_dir=./outputs-kvr-${test_model}
ckpt=best.model
beam_size=2

mkdir -p ${output_dir}/${ckpt}

${pythonpath_test} ./main.py --test --test_model=${test_model} --data_dir=${data_dir} --save_dir=${save_dir} --ckpt=${ckpt} --beam_size=${beam_size} --save_file=${output_dir}/${ckpt}/output.txt

${pythonpath_eval} tools.eval --data_name=${data_name} --data_dir=${data_dir} --eval_dir=${output_dir}/${ckpt}

for i in {1..15}
do
  ckpt=state_epoch_${i}.model
  mkdir -p ${output_dir}/${ckpt}
  ${pythonpath_test} ./main.py --test --test_model=${test_model} --data_dir=${data_dir} --save_dir=${save_dir} --ckpt=${ckpt} --beam_size=${beam_size} --save_file=${output_dir}/${ckpt}/output.txt
  ${pythonpath_eval} tools.eval --data_name=${data_name} --data_dir=${data_dir} --eval_dir=${output_dir}/${ckpt}
done

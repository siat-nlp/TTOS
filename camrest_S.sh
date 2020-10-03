#!/bin/bash

# set gpu id to use
export CUDA_VISIBLE_DEVICES=0

# set python path according to your actual environment
pythonpath_test='python3'
pythonpath_eval='python3 -m'

# set parameters
data_name='camrest'
data_dir=./data/CamRest
save_dir=./models-camrest
test_model=S
output_dir=./outputs-camrest-${test_model}
ckpt=best.model
beam_size=2
max_dec_len=25

mkdir -p ${output_dir}/${ckpt}

${pythonpath_test} ./main.py --test --test_model=${test_model} --data_dir=${data_dir} --save_dir=${save_dir} --ckpt=${ckpt} --beam_size=${beam_size} --max_dec_len=${max_dec_len} --save_file=${output_dir}/${ckpt}/output.txt

${pythonpath_eval} tools.eval --data_name=${data_name} --data_dir=${data_dir} --eval_dir=${output_dir}/${ckpt}

for i in {1..15}
do
  ckpt=state_epoch_${i}.model
  mkdir -p ${output_dir}/${ckpt}
  ${pythonpath_test} ./main.py --test --test_model=${test_model} --data_dir=${data_dir} --save_dir=${save_dir} --ckpt=${ckpt} --beam_size=${beam_size} --max_dec_len=${max_dec_len} --save_file=${output_dir}/${ckpt}/output.txt
  ${pythonpath_eval} tools.eval --data_name=${data_name} --data_dir=${data_dir} --eval_dir=${output_dir}/${ckpt}
done

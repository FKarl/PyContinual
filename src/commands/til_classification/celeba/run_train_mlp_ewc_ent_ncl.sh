#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --note random$id,100,online \
    --ntasks 10 \
    --nclasses 2 \
    --task celeba \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_ewc_ncl \
    --train_batch_size 128 \
    --train_data_size 100 \
    --data_size full \
    --image_size 32 \
    --image_channel 3 \
    --nepochs 300 \
    --lamb=5000 \
    --ent_id \
    --model_path "./models/fp32/til_classification/celeba/mlp_ewc_ncl_$id" \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/til_classification/celeba/mlp_ewc_ncl_$id" \
    --resume_from_task 9 \
    --eval_batch_size 1
done
#--nepochs 1


#nepochs=100,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,lamb=5000
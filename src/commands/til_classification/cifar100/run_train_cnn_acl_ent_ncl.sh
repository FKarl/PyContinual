#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id,online \
    --ntasks 10 \
    --task cifar100 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_acl_ncl \
    --ent_id \
    --model_path "./models/fp32/til_classification/cifar100/cnn_acl_$id" \
    --resume_from_file "./models/fp32/til_classification/cifar100/cnn_acl_$id"
done
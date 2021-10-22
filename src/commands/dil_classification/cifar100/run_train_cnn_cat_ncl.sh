#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  4
do
    python run.py \
    --note random$id,last_id \
    --ntasks 10 \
    --task cifar100 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_cat_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
	--last_id\
    --model_path "./models/fp32/dil_classification/cifar100/cnn_cat_$id" \
    --save_model
done

#TODO: cat's hyper-parameters?

#128 is the total maximum batch size
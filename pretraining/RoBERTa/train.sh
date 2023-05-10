#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
CURRENT_DIR=$(pwd)
ROOT_DIR=$(realpath ../..)
DATA_DIR=${ROOT_DIR}/pretraining/data

OUT_DIR=$CURRENT_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUs=8

python -m torch.distributed.launch --nproc_per_node $NUM_GPUs --use_env run_mlm_no_trainer.py \
    --model_name_or_path roberta-base \
    --train_file $DATA_DIR/train.txt \
    --validation_file $DATA_DIR/test.txt \
    --output_dir $OUT_DIR \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --learning_rate 6e-4 \
    --beta_1 0.9 \
    --beta_2 0.98 \
    --max_train_steps 12500 \
    --num_warmup_steps 600 \
    --save_interval_updates 625 \
    2>&1 | tee $OUT_DIR/training.log

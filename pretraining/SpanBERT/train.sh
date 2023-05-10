#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
CURRENT_DIR=$(pwd)
ROOT_DIR=$(realpath ../..)
DATA_DIR=${ROOT_DIR}/pretraining/data

SAVE_DIR=$CURRENT_DIR/spanbert_checkpoints
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

function prepare() {
    mkdir -p $CURRENT_DIR/data
    python bpe_tokenize.py $DATA_DIR/train.txt $CURRENT_DIR/data/train_tokenized.txt
    python bpe_tokenize.py $DATA_DIR/test.txt $CURRENT_DIR/data/test_tokenized.txt
    python preprocess.py \
        --only-source \
        --trainpref $CURRENT_DIR/data/train_tokenized.txt \
        --validpref $CURRENT_DIR/data/test_tokenized.txt \
        --srcdict dict.txt \
        --destdir $CURRENT_DIR/data \
        --padding-factor 1 \
        --workers 48
}

function train() {
    python train.py $CURRENT_DIR/data \
        --total-num-update 100000 \
        --max-update 100000 \
        --save-interval 1 \
        --pretrained-bert-path spanbert-base-cased \
        --arch cased_bert_pair \
        --task span_bert \
        --optimizer adam \
        --lr-scheduler polynomial_decay \
        --lr 0.0001 \
        --min-lr 1e-09 \
        --criterion span_bert_loss \
        --batch-size 16 \
        --update-freq 2 \
        --tokens-per-sample 512 \
        --weight-decay 0.01 \
        --skip-invalid-size-inputs-valid-test \
        --log-format json \
        --log-interval 1000 \
        --save-interval-updates 5000 \
        --keep-interval-updates 5 \
        --seed 1234 \
        --save-dir $SAVE_DIR \
        --warmup-updates 1000 \
        --schemes [\"pair_span\"] \
        --span-lower 1 \
        --span-upper 10 \
        --validate-interval 1 \
        --clip-norm 1.0 \
        --geometric-p 0.2 \
        --adam-eps 1e-8 \
        --short-seq-prob 0.0 \
        --replacement-method span \
        --clamp-attention \
        --no-nsp \
        --pair-loss-weight 1.0 \
        --max-pair-targets 15 \
        --pair-positional-embedding-size 200 \
        --endpoints external
}

prepare
train

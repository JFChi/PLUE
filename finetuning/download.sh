#!/usr/bin/env bash

CURRENT_DIR=$PWD
MODEL_DIR=${CURRENT_DIR}/models
mkdir -p $MODEL_DIR

function wget_gdrive() {
    GDRIVE_FILE_ID=$1
    DEST_PATH=$2
    if [[ ! -f "$DEST_PATH" ]]; then
        echo "Downloading AtCoder test cases from https://drive.google.com/file/d/${GDRIVE_FILE_ID}"
        wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$GDRIVE_FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' >confirm.txt
        wget --load-cookies cookies.txt -O $DEST_PATH 'https://docs.google.com/uc?export=download&id='$GDRIVE_FILE_ID'&confirm='$(<confirm.txt)
        rm cookies.txt confirm.txt
    fi
}

fileid="1hsErogLA6YR2ap7GSCMEU_9RQIsSf3lH"
filename=policy-bert-base-uncased.tar.gz
if [[ ! -f "$MODEL_DIR/$filename" ]]; then
    wget_gdrive $fileid $MODEL_DIR/$filename
    tar -xvzf $MODEL_DIR/$filename -C $MODEL_DIR
fi

fileid="12X8KJQpmiujuwTQecJvgcS4ufLpc1ruY"
filename=policy-roberta-base.tar.gz
if [[ ! -f "$MODEL_DIR/$filename" ]]; then
    wget_gdrive $fileid $MODEL_DIR/$filename
    tar -xvzf $MODEL_DIR/$filename -C $MODEL_DIR
fi

fileid="1usv6iH7cemit6eVz5bmuuht_cBXgAvTK"
filename=policy-electra-base-discriminator.tar.gz
if [[ ! -f "$MODEL_DIR/$filename" ]]; then
    wget_gdrive $fileid $MODEL_DIR/$filename
    tar -xvzf $MODEL_DIR/$filename -C $MODEL_DIR
fi

fileid="1UjNY7l-2RxhgagT05DxACVXZbCeXP4LY"
filename=policy-spanbert-base-uncased.tar.gz
if [[ ! -f "$MODEL_DIR/$filename" ]]; then
    wget_gdrive $fileid $MODEL_DIR/$filename
    tar -xvzf $MODEL_DIR/$filename -C $MODEL_DIR
fi

fileid="1O6aA8SGguJV5ANVsORmxwh4BG9IA78Wk"
filename=policy-roberta-large.tar.gz
if [[ ! -f "$MODEL_DIR/$filename" ]]; then
    wget_gdrive $fileid $MODEL_DIR/$filename
    tar -xvzf $MODEL_DIR/$filename -C $MODEL_DIR
fi
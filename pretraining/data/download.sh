#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
DATA_DIR=$(pwd)

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

fileid="140E1pTD5pgvOzsr_fhH3NjOvfHiMItVu"
filename=train.txt
if [[ ! -f "$filename" ]]; then
    wget_gdrive $fileid $DATA_DIR/$filename
fi

fileid="1TLa5ZtsRjRrhx5GyOo2iU8AkF-77Q7cd"
filename=test.txt
if [[ ! -f "$filename" ]]; then
    wget_gdrive $fileid $DATA_DIR/$filename
fi

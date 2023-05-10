#!/usr/bin/env bash

CURRENT_DIR=$PWD

function piextract() {
    target_dir=$CURRENT_DIR/piextract
    python prepare.py \
        --dataset piextract \
        --source_dir $target_dir
}

function privacyqa() {
    target_dir=$CURRENT_DIR/privacyqa
    mkdir -p $target_dir
    git clone https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP.git
    cp -r PrivacyQA_EMNLP/data/*.csv $target_dir
    rm -rf PrivacyQA_EMNLP
}

function opp115() {
    target_dir=$CURRENT_DIR/opp115
    mkdir -p $target_dir
    if [ ! -d $target_dir/OPP-115 ]; then
        wget --no-check-certificate https://usableprivacy.org/static/data/OPP-115_v1_0.zip -P $target_dir
        unzip $target_dir/OPP-115_v1_0.zip -d $target_dir
    else
        echo "Data exist. No need to download"
    fi
    python prepare.py \
        --dataset opp115 \
        --source_dir $target_dir
    rm -rf $target_dir/OPP-115_v1_0.zip $target_dir/OPP-115 $target_dir/__MACOSX
}

function app350() {
    target_dir=$CURRENT_DIR/app350
    mkdir -p $target_dir
    if [ ! -d $target_dir/APP-350_v1.1 ]; then
        wget --no-check-certificate https://usableprivacy.org/static/data/APP-350_v1.1.zip -P $target_dir
        unzip $target_dir/APP-350_v1.1.zip -d $target_dir
    else
        echo "Data exist. No need to download"
    fi
    python prepare.py \
        --dataset app350 \
        --source_dir $target_dir
    rm -rf $target_dir/APP-350_v1.1.zip $target_dir/APP-350_v1.1 $target_dir/__MACOSX
}

piextract
privacyqa
app350
opp115

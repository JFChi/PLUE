#!/usr/bin/env bash

CURRENT_DIR=$(pwd)
DATA_DIR=${CURRENT_DIR}/policyie

# ****************** DATA PREPARATION ****************** #

function prepare() {
    unzip sanitized_split.zip

    # pre-processing data from json to BIO tagging format
    python process.py \
        --input_paths sanitized_split/train \
        --out_dir ${DATA_DIR}/train

    python process.py \
        --input_paths sanitized_split/test \
        --out_dir ${DATA_DIR}/test

    for split in train test; do
        python del_partial_slots.py --input_dir ${DATA_DIR}/${split}
        python del_partial_slots.py --input_dir ${DATA_DIR}/${split}
    done

    # Shorten labels
    python shorten_labels.py --input_path ${DATA_DIR}/train --out_dir ${DATA_DIR}/train
    python shorten_labels.py --input_path ${DATA_DIR}/test --out_dir ${DATA_DIR}/test

    # sanity check for BIO tagging data
    for split in train test; do
        python BIO_sanity_check.py --input_file ${DATA_DIR}/${split}/seq_type_I.out
        python BIO_sanity_check.py --input_file ${DATA_DIR}/${split}/seq_type_II.out
        python BIO_sanity_check.py --input_file ${DATA_DIR}/${split}/short_seq_type_I.out
        python BIO_sanity_check.py --input_file ${DATA_DIR}/${split}/short_seq_type_II.out
    done

    # convert the data into compositional form
    python flattener.py --data_dir ${DATA_DIR}
    # generate label (intent, slots and postag) files
    python labeler.py --data_dir ${DATA_DIR}

    rm -rf sanitized_split
    rm -rf __MACOSX
}

# ****************** BIO-tagging DATA FORMATTING ****************** #

function bio_formatting() {

    dest_dir=bio_format
    mkdir -p $dest_dir

    mkdir -p $dest_dir
    mkdir -p $dest_dir/train
    mkdir -p $dest_dir/valid
    mkdir -p $dest_dir/test

    cp $DATA_DIR/train/seq.in $dest_dir/temp.seq
    cp $DATA_DIR/train/seq_type_I.out $dest_dir/temp.seq_type_I
    cp $DATA_DIR/train/seq_type_II.out $dest_dir/temp.seq_type_II
    cp $DATA_DIR/train/label $dest_dir/temp.label
    cp $DATA_DIR/train/pos_tags.out $dest_dir/temp.pos_tags

    # valid
    tail -100 $dest_dir/temp.seq >$dest_dir/valid/seq.in
    tail -100 $dest_dir/temp.seq_type_I >$dest_dir/valid/seq_type_I.out
    tail -100 $dest_dir/temp.seq_type_II >$dest_dir/valid/seq_type_II.out
    tail -100 $dest_dir/temp.label >$dest_dir/valid/label
    tail -100 $dest_dir/temp.pos_tags >$dest_dir/valid/pos_tags.out

    # train
    head -n -100 $dest_dir/temp.seq >$dest_dir/train/seq.in
    head -n -100 $dest_dir/temp.seq_type_I >$dest_dir/train/seq_type_I.out
    head -n -100 $dest_dir/temp.seq_type_II >$dest_dir/train/seq_type_II.out
    head -n -100 $dest_dir/temp.label >$dest_dir/train/label
    head -n -100 $dest_dir/temp.pos_tags >$dest_dir/train/pos_tags.out

    rm $dest_dir/temp.seq && rm $dest_dir/temp.label
    rm $dest_dir/temp.seq_type_I && rm $dest_dir/temp.seq_type_II
    rm $dest_dir/temp.pos_tags

    # test files
    cp $DATA_DIR/test/seq.in $dest_dir/test/seq.in
    cp $DATA_DIR/test/seq_type_I.out $dest_dir/test/seq_type_I.out
    cp $DATA_DIR/test/seq_type_II.out $dest_dir/test/seq_type_II.out
    cp $DATA_DIR/test/label $dest_dir/test/label
    cp $DATA_DIR/test/pos_tags.out $dest_dir/test/pos_tags.out

    # label files
    cp $DATA_DIR/type_II_slot_label.txt $dest_dir/type_II_slot_label.txt
    cp $DATA_DIR/type_I_slot_label.txt $dest_dir/type_I_slot_label.txt
    cp $DATA_DIR/intent_label.txt $dest_dir/intent_label.txt
    cp $DATA_DIR/postag_label.txt $dest_dir/postag_label.txt

}

prepare
bio_formatting
python prepare.py
rm -rf bio_format && rm -rf policyie

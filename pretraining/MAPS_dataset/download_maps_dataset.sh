#!/bin/sh

DATA_PATH=data
mkdir -p $DATA_PATH

if [ ! -f "${DATA_PATH}/MAPS_Policies_Dataset_v1.0.zip" ]
    then
    echo "Downloading and unzipping MAP dataset"
    wget --no-check-certificate https://usableprivacy.org/static/data/MAPS_Policies_Dataset_v1.0.zip -P ${DATA_PATH}/
    unzip data/MAPS_Policies_Dataset_v1.0.zip -d $DATA_PATH
    mv ${DATA_PATH}/MAPS\ Policies\ Dataset/* $DATA_PATH
    rm -r ${DATA_PATH}/MAPS\ Policies\ Dataset/
    rm -r ${DATA_PATH}/__MACOSX
else
    echo "MAP dataset file found."
fi





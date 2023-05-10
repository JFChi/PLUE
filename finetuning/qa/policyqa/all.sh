#!/usr/bin/env bash

# ps aux | grep ec2-user | grep 'python main.py' | awk '{ print $2 }' | xargs -n1 sudo kill -9

nohup bash run.sh 0 bert_large 1e-5 >/dev/null 2>&1 &
nohup bash run.sh 1 bert_large 3e-5 >/dev/null 2>&1 &

nohup bash run.sh 2 spanbert_large 1e-5 >/dev/null 2>&1 &
nohup bash run.sh 3 spanbert_large 3e-5 >/dev/null 2>&1 &
nohup bash run.sh 4 spanbert_large 5e-5 >/dev/null 2>&1 &

nohup bash run.sh 5 electra_large 1e-5 >/dev/null 2>&1 &
nohup bash run.sh 6 electra_large 3e-5 >/dev/null 2>&1 &
nohup bash run.sh 7 electra_large 5e-5 >/dev/null 2>&1 &

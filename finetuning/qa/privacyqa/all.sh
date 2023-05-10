#!/usr/bin/env bash

# ps aux | grep ec2-user | grep 'python main.py' | awk '{ print $2 }' | xargs -n1 sudo kill -9

#nohup bash run.sh 0 spanbert 1e-5 >/dev/null 2>&1 &
#nohup bash run.sh 1 spanbert 3e-5 >/dev/null 2>&1 &
#nohup bash run.sh 2 spanbert 5e-5 >/dev/null 2>&1 &
#nohup bash run.sh 3 spanbert 1e-4 >/dev/null 2>&1 &
#nohup bash run.sh 4 spanbert 5e-6 >/dev/null 2>&1 &
#nohup bash run.sh 5 spanbert 3e-6 >/dev/null 2>&1 &
#nohup bash run.sh 6 spanbert 1e-6 >/dev/null 2>&1 &

#nohup bash run.sh 0 bert_large 2e-5 >/dev/null 2>&1 &
#nohup bash run.sh 1 bert_large 3e-5 >/dev/null 2>&1 &
#nohup bash run.sh 2 spanbert_large 5e-5 >/dev/null 2>&1 &
#nohup bash run.sh 3 spanbert_large 1e-4 >/dev/null 2>&1 &
#nohup bash run.sh 4 electra_large 5e-6 >/dev/null 2>&1 &
#nohup bash run.sh 5 electra_large 3e-6 >/dev/null 2>&1 &

#nohup bash run.sh 0 bert_large 1e-5 >/dev/null 2>&1 &
#nohup bash run.sh 1 bert_large 5e-6 >/dev/null 2>&1 &
#nohup bash run.sh 2 bert_large 3e-6 >/dev/null 2>&1 &
#
#nohup bash run.sh 3 spanbert_large 3e-5 >/dev/null 2>&1 &
#nohup bash run.sh 4 spanbert_large 1e-5 >/dev/null 2>&1 &
#nohup bash run.sh 5 spanbert_large 5e-6 >/dev/null 2>&1 &
#
#nohup bash run.sh 6 electra_large 1e-6 >/dev/null 2>&1 &
#nohup bash run.sh 7 electra_large 1e-5 >/dev/null 2>&1 &

#nohup bash run.sh 0 bert >/dev/null 2>&1 &
#nohup bash run.sh 1 spanbert >/dev/null 2>&1 &
#nohup bash run.sh 2 electra >/dev/null 2>&1 &
#
#nohup bash run.sh 3 policy_bert >/dev/null 2>&1 &
#nohup bash run.sh 4 policy_spanbert >/dev/null 2>&1 &
#nohup bash run.sh 5 policy_electra >/dev/null 2>&1 &
#
#nohup bash run.sh 6 legal_bert >/dev/null 2>&1 &
#nohup bash run.sh 0 bert_large 5e-6 >/dev/null 2>&1 &

nohup bash run.sh 0 roberta_large 1e-5 >/dev/null 2>&1 &
nohup bash run.sh 1 roberta_large 3e-5 >/dev/null 2>&1 &
nohup bash run.sh 2 roberta_large 5e-5 >/dev/null 2>&1 &
nohup bash run.sh 3 roberta_large 1e-6 >/dev/null 2>&1 &
nohup bash run.sh 4 roberta_large 3e-6 >/dev/null 2>&1 &
nohup bash run.sh 5 roberta_large 5e-6 >/dev/null 2>&1 &

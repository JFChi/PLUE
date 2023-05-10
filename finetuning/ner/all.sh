#!/usr/bin/env bash

# ps aux | grep ec2-user | grep 'python main.py' | awk '{ print $2 }' | xargs -n1 sudo kill -9

#nohup bash run.sh 0 bert >/dev/null 2>&1 &
#nohup bash run.sh 1 roberta >/dev/null 2>&1 &
#nohup bash run.sh 2 electra >/dev/null 2>&1 &
#nohup bash run.sh 3 spanbert >/dev/null 2>&1 &
#nohup bash run.sh 4 policy_bert >/dev/null 2>&1 &
#nohup bash run.sh 5 policy_roberta >/dev/null 2>&1 &
#nohup bash run.sh 6 policy_electra >/dev/null 2>&1 &
#nohup bash run.sh 7 policy_spanbert >/dev/null 2>&1 &

#nohup bash run.sh 0 electra_large 3e-5 >/dev/null 2>&1 &
#nohup bash run.sh 1 electra_large 1e-5 >/dev/null 2>&1 &
#nohup bash run.sh 2 electra_large 5e-6 >/dev/null 2>&1 &
#nohup bash run.sh 3 electra_large 3e-6 >/dev/null 2>&1 &

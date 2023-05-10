#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
CURRENT_DIR=$(pwd)
ROOT_DIR=$(realpath ../../..)

declare -A model_zoo
model_zoo["bert"]="bert-base-uncased"
model_zoo["roberta"]="roberta-base"
model_zoo["spanbert"]="SpanBERT/spanbert-base-cased"
model_zoo["electra"]="google/electra-base-discriminator"
model_zoo["bert_large"]="bert-large-cased"
model_zoo["roberta_large"]="roberta-large"
model_zoo["spanbert_large"]="SpanBERT/spanbert-large-cased"
model_zoo["electra_large"]="google/electra-large-discriminator"
model_zoo["legal_bert"]="nlpaueb/legal-bert-base-uncased"
model_zoo["policy_bert"]="${ROOT_DIR}/finetuning/models/policy-bert-base-uncased/best"
model_zoo["policy_roberta"]="${ROOT_DIR}/finetuning/models/policy-roberta-base/best"
model_zoo["policy_spanbert"]="${ROOT_DIR}/finetuning/models/policy-spanbert-base-uncased/spanbert"
model_zoo["policy_electra"]="${ROOT_DIR}/finetuning/models/policy-electra-base-discriminator/best"
model_zoo["policy_roberta_large"]="${ROOT_DIR}/finetuning/models/policy-roberta-large/best"

declare -A tokenizer_zoo
tokenizer_zoo["bert"]="bert-base-uncased"
tokenizer_zoo["roberta"]="roberta-base"
tokenizer_zoo["spanbert"]="SpanBERT/spanbert-base-cased"
tokenizer_zoo["electra"]="google/electra-base-discriminator"
tokenizer_zoo["bert_large"]="bert-large-cased"
tokenizer_zoo["roberta_large"]="roberta-large"
tokenizer_zoo["spanbert_large"]="SpanBERT/spanbert-large-cased"
tokenizer_zoo["electra_large"]="google/electra-large-discriminator"
tokenizer_zoo["legal_bert"]="nlpaueb/legal-bert-base-uncased"
tokenizer_zoo["policy_bert"]="bert-base-uncased"
tokenizer_zoo["policy_roberta"]="roberta-base"
tokenizer_zoo["policy_spanbert"]="SpanBERT/spanbert-base-cased"
tokenizer_zoo["policy_electra"]="google/electra-base-discriminator"
tokenizer_zoo["policy_roberta_large"]="roberta-large"

declare -A config_zoo
config_zoo["bert"]="bert-base-uncased"
config_zoo["roberta"]="roberta-base"
config_zoo["spanbert"]="SpanBERT/spanbert-base-cased"
config_zoo["electra"]="google/electra-base-discriminator"
config_zoo["bert_large"]="bert-large-cased"
config_zoo["roberta_large"]="roberta-large"
config_zoo["spanbert_large"]="SpanBERT/spanbert-large-cased"
config_zoo["electra_large"]="google/electra-large-discriminator"
config_zoo["legal_bert"]="nlpaueb/legal-bert-base-uncased"
config_zoo["policy_bert"]="bert-base-uncased"
config_zoo["policy_roberta"]="roberta-base"
config_zoo["policy_spanbert"]="SpanBERT/spanbert-base-cased"
config_zoo["policy_electra"]="google/electra-base-discriminator"
config_zoo["policy_roberta_large"]="roberta-large"

declare -A learning_rate
learning_rate["bert"]=3e-5
learning_rate["roberta"]=3e-5
learning_rate["spanbert"]=5e-5
learning_rate["electra"]=5e-5
learning_rate["bert_large"]=5e-5
learning_rate["roberta_large"]=5e-5
learning_rate["spanbert_large"]=3e-5
learning_rate["electra_large"]=1e-5
learning_rate["legal_bert"]=3e-5
learning_rate["policy_bert"]=3e-5
learning_rate["policy_roberta"]=3e-5
learning_rate["policy_spanbert"]=5e-5
learning_rate["policy_electra"]=5e-5
learning_rate["policy_roberta_large"]=5e-5

GPU=${1:-0}
MODEL_NAME=${2:-'bert'}
LEARNING_RATE=${3:-"${learning_rate["$MODEL_NAME"]}"}

export CUDA_VISIBLE_DEVICES=$GPU
export DATA_DIR=${ROOT_DIR}/data/opp115

function train() {
    python main.py \
        --config_name "${config_zoo["$MODEL_NAME"]}" \
        --tokenizer_name "${tokenizer_zoo["$MODEL_NAME"]}" \
        --model_name_or_path "${model_zoo["$MODEL_NAME"]}" \
        --do_train \
        --do_eval \
        --train_file $DATA_DIR/train.json \
        --validation_file $DATA_DIR/valid.json \
        --label_list_file $DATA_DIR/label_list.txt \
        --learning_rate $LEARNING_RATE \
        --warmup_ratio 0.05 \
        --num_train_epochs 20 \
        --balance_classes True \
        --max_seq_length 128 \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir \
        --overwrite_cache \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --save_steps 200 \
        --save_total_limit 1 \
        --seed $SEED \
        2>&1 | tee $OUTPUT_DIR/finetuning.log
}

function evaluate() {
    python main.py \
        --config_name "${config_zoo["$MODEL_NAME"]}" \
        --tokenizer_name "${tokenizer_zoo["$MODEL_NAME"]}" \
        --model_name_or_path $OUTPUT_DIR \
        --do_predict \
        --test_file $DATA_DIR/test.json \
        --label_list_file $DATA_DIR/label_list.txt \
        --max_seq_length 128 \
        --output_dir $OUTPUT_DIR \
        --overwrite_cache \
        --per_device_eval_batch_size 32 \
        --seed $SEED \
        2>&1 | tee $OUTPUT_DIR/evaluation.log
}

for SEED in 1234 2022 42; do
    OUTPUT_DIR=${CURRENT_DIR}/outputs_${SEED}/${MODEL_NAME}_lr$LEARNING_RATE
    mkdir -p $OUTPUT_DIR
    train
    evaluate
done

printf "\033c" # clear screen
echo "$MODEL_NAME"
for SEED in 1234 2022 42; do
    echo "SEED: $SEED"
    OUTPUT_DIR=${CURRENT_DIR}/outputs_${SEED}/${MODEL_NAME}_lr$LEARNING_RATE
    tail -17 $OUTPUT_DIR/evaluation.log | head -3
done

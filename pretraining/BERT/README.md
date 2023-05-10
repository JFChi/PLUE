## Install Environment

```
pip install -r requirements.txt
```

## Prepare training corpus for MLM training

Run the following script to perform train/test split and other pre-processing steps for MLM tasks. Note that we should
put the pretraining_corpus folder as the data_dir argument. In our case, the pretraining corpus folder should contain
both maps_txt_policies and privacy_policy_history_txt_policies folders, which could be downloaded in our Google drive.

```
PRETRAINING_CORPUS=data/pretraining_corpus
mkdir -p $PRETRAINING_CORPUS # then put the maps_txt_policies and privacy_policy_history_txt_policies inside the folder
python preprocess_pretraining_corpus.py --data_dir $PRETRAINING_CORPUS --out_dir $PRETRAINING_CORPUS
```

After running the script, we will have both train.txt and test.txt. We can also get them from google drive (train.txt
and test.txt in the folder name pretraining_corpus)

## Run MLM Pre-training

If you want to run with one gpu (be careful when the gpu memory is not enough for this task):

```
OUT_DIR=<MODEL_CHECKPOINT_PATH>
python run_mlm_no_trainer.py \
    --train_file data/pretraining_corpus/train.txt \
    --validation_file data/pretraining_corpus/test.txt \
    --model_name_or_path bert-base-uncased \
    --output_dir $OUT_DIR \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --max_train_steps 100000 \
    --num_warmup_steps 1000 \
    --save_interval_updates 5000
```

Otherwise you can use the following command to run the script on multiple gpus (recommended).

```
OUT_DIR=<MODEL_CHECKPOINT_PATH>
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUs=4
python -m torch.distributed.launch --nproc_per_node $NUM_GPUs --use_env run_mlm_no_trainer.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/pretraining_corpus/train.txt \
    --validation_file data/pretraining_corpus/test.txt \
    --output_dir $OUT_DIR \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --max_train_steps 100000 \
    --num_warmup_steps 1000 \
    --save_interval_updates 5000
```

## Example of Model Loading

```
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

model_name_or_path=<MODEL_CHECKPOINT_PATH>

config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForMaskedLM.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
    ) # depending on the task, we can change the auto model loading type (If this is the case, the prediction of MLM model will be discarded)
```

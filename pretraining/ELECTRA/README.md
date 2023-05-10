## Install Environment

```
pip install -r requirements.txt
```

## Run ELECTRA Pre-training

First, download the train.txt and text.txt, which are generated during BERT pretraining, and place them into the data
folder.

If you want to run with one gpu (be careful when the gpu memory is not enough for this task):

```
OUT_DIR=<MODEL_CHECKPOINT_PATH>
python train.py \
    --train_file data/train.txt \
    --validation_file data/test.txt \
    --model_size base \
    --output_dir $OUT_DIR \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.01 \
    --learning_rate 2e-4 \
    --max_train_steps 100000 \
    --num_warmup_steps 1000 \
    --save_interval_updates 5000 \
    --generator_size_divisor 3 \
    --tie_weights
```

Otherwise you can use the following command to run the script on multiple gpus (recommended).

```
OUT_DIR=<MODEL_CHECKPOINT_PATH>
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUs=4
python -m torch.distributed.launch --nproc_per_node $NUM_GPUs --use_env train.py \
    --model_size base \
    --train_file data/train.txt \
    --validation_file data/test.txt \
    --output_dir $OUT_DIR \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --max_train_steps 100000 \
    --num_warmup_steps 1000 \
    --save_interval_updates 5000 \
    --generator_size_divisor 3 \
    --tie_weights
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
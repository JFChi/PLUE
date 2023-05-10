## Install Environment

```
pip install -r requirements.txt
```

## Run RoBERTa aMLM Pre-training

```
bash train.sh
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

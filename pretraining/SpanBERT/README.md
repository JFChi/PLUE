# Install Environment

```
pip install -r requirements.txt
```

# Preprocessing

* Tokenization
  Download the train.txt and test.txt from Google drive pretraining_corpus (those files are obtained via running
  pretraining/BERT/preprocess_pretraining_corpus.py).

```
python bpe_tokenize.py /path-to-pretraining-corpus/train.txt /path-to-pretraining-corpus/train_tokenized.txt
python bpe_tokenize.py /path-to-pretraining-corpus/test.txt /path-to-pretraining-corpus/test_tokenized.txt
```

* Fairseq Preprocessing

```
python preprocess.py --only-source --trainpref /path-to-pretraining-corpus/train_tokenized.txt --validpref /path-to-pretraining-corpus/test_tokenized.txt --srcdict dict.txt --destdir /path-to-preprocessed-pretraining-corpus --padding-factor 1 --workers 48
```

Make sure dict.txt contains two columns. The first is the word peice and the second is the dummy frequency in descending
order.

# Pre-training

First download the spanbert model from huggingface, e.g.,

```
git lfs install
git clone https://huggingface.co/SpanBERT/spanbert-base-cased
```

Use this command to start pretraining. See fairseq/options.py for more information on SpanBERT configuration.

```
CUDA_VISIBLE_DEVICES=0,1,3,4 python train.py /path-to-preprocessed-pretraining-corpus \
    --total-num-update 100000 \
    --max-update 100000 \
    --save-interval 1 \
    --pretrained-bert-path spanbert-base-cased \
    --arch cased_bert_pair \
    --task  span_bert \
    --optimizer adam \
    --lr-scheduler polynomial_decay \
    --lr 0.0001 \
    --min-lr 1e-09  \
    --criterion span_bert_loss  \
    --batch-size 16 --update-freq 4 \
    --tokens-per-sample 512 \
    --weight-decay 0.01 \
    --skip-invalid-size-inputs-valid-test \
    --log-format json \
    --log-interval 1000 \
    --save-interval-updates 5000 \
    --keep-interval-updates 5 \
    --seed 1234 \
    --save-dir plue_spanbert_checkpoint \
    --warmup-updates 1000 \
    --schemes [\"pair_span\"] --span-lower 1 --span-upper 10 --validate-interval 1  --clip-norm 1.0 --geometric-p 0.2 --adam-eps 1e-8 --short-seq-prob 0.0 --replacement-method span --clamp-attention --no-nsp --pair-loss-weight 1.0 --max-pair-targets 15 --pair-positional-embedding-size 200 --endpoints external
```

Be careful with GPU memory comsumption! If there is a OOM problem, you might see errors like "_pickle.UnpicklingError".

# Convert Checkpoint to Huggingface Format

```
python convert_checkpoint_to_hfmodel.py --checkpoint checkpoint_dir/plue_spanbert --out_dir plue_spanbert_hf
```

Running the script will generate config.json and pytorch.bin in the output dir.

#### [Optional]

Next, we also need to put the vocab.txt and tokenizer config files to the output dir.
For the vocabulary information, please
download [here](https://huggingface.co/SpanBERT/spanbert-base-cased/blob/main/vocab.txt) (the vacab.txt is the same as
dict.txt in *Preprocessing* step if we do not change the dict.txt).
For tokenizer configuation, since SpanBERT uses 'bert-base-cased' tokenizer by default, we can download
the [tokenizer.json](https://huggingface.co/bert-base-cased/blob/main/tokenizer.json)
and [tokenizer_config.json](https://huggingface.co/bert-base-cased/blob/main/tokenizer_config.json) in bert-base-cased
in huggingface.
**Note that** if we change the vocabulary or tokenizer configuration in the *preprocessing* step, we might need to
hard-code
those config files.

# Model Loading

```
from transformers import AutoTokenizer, AutoModel

model_name_or_path=<MODEL_CHECKPOINT_PATH>

model = AutoModel.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
```

# Important Files

* `fairseq/tasks/span_bert.py`: Main task file which also contains the all the task-specific options.
* `fairseq/data/no_nsp_span_bert_dataset.py`: This is where the data preprocessing happens.
* `fairseq/data/masking.py`: All the masking schemes are defined here. These are called from the dataset files above.
* `fairseq/criterions` -- `span_bert_loss`: The losses are defined here. **Make sure `--no_-nsp` is set to true when
  using the no_-nsp losses**
* `fairseq/models/pair_bert.py`: Transformer model.



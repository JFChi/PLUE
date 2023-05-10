import os
import argparse
from collections import OrderedDict
import json

import torch

parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
parser.add_argument(
    "--checkpoint",
    default="checkpoint_dir/privacy_spanbert",
    type=str,
    help="Path to pretrained SpanBERT model",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="hf_privacy_spanbert",
    help="Path to store huggingface pytorch_model.bin",
)
args = parser.parse_args()

# sanity check
assert os.path.exists(args.checkpoint)
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

input_obj = torch.load(os.path.join(args.checkpoint, "checkpoint_last.pt"))
input_model_obj = input_obj['model']
config = input_obj['args'].config.to_dict()

try:
    best_loss = input_obj['extra_state']['best']
    best_epoch = input_obj['extra_state']['train_iterator']['epoch']
    best_num_updates = input_obj['optimizer_history'][0]['num_updates']
    valid_nll_loss = input_obj['extra_state']['train_meters']['valid_nll_loss'].avg
    valid_ppl = 2 ** valid_nll_loss

    print(
        f"The best model is found during epoch {best_epoch} (step {best_num_updates}), with "
        f"validation loss: {best_loss}, nll_loss {valid_nll_loss} and ppl {valid_ppl}"
    )
except KeyError:
    print("key missing: cannot print the stats of the best model ...")


# rename model parameter
def key_transformation(old_key):
    '''remove the name 'decoder.'''
    return old_key.replace('decoder.', '', 1)


output_model_obj = OrderedDict()

for key, value in input_model_obj.items():
    new_key = key_transformation(key)
    # we discard the MLM and SBO prediction head 
    if new_key.startswith("cls."):
        continue
    output_model_obj[new_key] = value

torch.save(output_model_obj, os.path.join(args.out_dir, "pytorch_model.bin"))
print(f"Model is saved at {args.out_dir}")

# save config.json
## remove unnecessary config parameters 
config.pop("clamp_attention", None)

# add additional config parameters (format follows https://huggingface.co/SpanBERT/spanbert-base-cased/blob/main/config.json)
config["directionality"] = "bidi"
config["layer_norm_eps"] = 1e-12  # NOTE: check the parameter in fairseq/models/pair_bert.py
config["model_type"] = "bert"
config["pad_token_id"] = 0  # NOTE: check the dict.txt to see pad_token_id

with open(os.path.join(args.out_dir, "config.json"), 'w') as f:
    json.dump(config, f, indent=2)

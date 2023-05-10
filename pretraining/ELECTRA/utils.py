

import logging

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Tuple


logger = logging.getLogger(__name__)

def update_model_config(gen_config, disc_config, args):
    gen_config.hidden_size = disc_config.hidden_size // args.generator_size_divisor
    gen_config.num_attention_heads = disc_config.num_attention_heads // args.generator_size_divisor
    gen_config.intermediate_size = disc_config.intermediate_size // args.generator_size_divisor

    assert gen_config.hidden_size > 0
    assert gen_config.num_attention_heads > 0
    assert gen_config.intermediate_size > 0

    return gen_config, disc_config


def tie_weights(generator, discriminator):
    discriminator.electra.embeddings.word_embeddings = generator.electra.embeddings.word_embeddings
    discriminator.electra.embeddings.position_embeddings = generator.electra.embeddings.position_embeddings
    discriminator.electra.embeddings.token_type_embeddings = generator.electra.embeddings.token_type_embeddings

    generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "pt":
            return self.torch_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    <Tip>
    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.
    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    orginal_prob: float = 0.15
    replace_prob: float = 0.0
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def torch_call(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs, special_tokens_mask: None):
        # Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # (1 - replace_prob - orginal_prob) of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        mask_prob = 1 - self.replace_prob - self.orginal_prob
        indices_replaced = torch.bernoulli(torch.full(labels.shape, mask_prob)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # replace_prob of the time, we replace masked input tokens with random word
        if abs(self.replace_prob - 0.0) > 1e-6:
            rep_prob = self.replace_prob / (self.replace_prob + self.orginal_prob)
            indices_random = torch.bernoulli(torch.full(labels.shape, rep_prob)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]
        
        # The rest of the time, we keep the masked input tokens unchanged
        return inputs, labels
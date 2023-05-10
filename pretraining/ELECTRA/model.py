from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from transformers.utils import ModelOutput

def log(t, eps=1e-9):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

@dataclass
class ElectraModelOutput(ModelOutput):
    """
    Output type of [`ElectraModelOutput`].
    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss of the ELECTRA objective.
    """

    loss: Optional[torch.FloatTensor] = None
    generator_loss: Optional[torch.FloatTensor] = None
    discriminator_loss: Optional[torch.FloatTensor] = None
    gen_acc: Optional[torch.FloatTensor] = None
    masked_disc_acc: Optional[torch.FloatTensor] = None
    unmasked_disc_acc: Optional[torch.FloatTensor] = None
    disc_ones_ratio: Optional[torch.FloatTensor] = None


class ELECTRAModel(nn.Module):
    def __init__(
        self, 
        generator, 
        discriminator,
        gen_weight = 1.,
        disc_weight = 50.,
    ):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        self.generator_loss_fct = nn.CrossEntropyLoss() # -100 index = padding token
        self.discriminator_loss_fct = nn.BCEWithLogitsLoss()

        # loss weights
        self.gen_weight = gen_weight
        self.disc_weight = disc_weight

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):

        # get generator output and get mlm loss
        generator_logits = self.generator(input_ids, attention_mask, token_type_ids)[0]
        generator_loss = self.generator_loss_fct(generator_logits.view(-1, self.generator.config.vocab_size), labels.view(-1))

        with torch.no_grad():
            # generate mask index boolean matrix from labels
            masked_indices = labels.clone()
            masked_indices[masked_indices != -100] = 1
            masked_indices[masked_indices == -100] = 0
            masked_indices = masked_indices.bool()

            # use mask from before to select logits that need sampling 
            sampled_generator_logits = generator_logits[masked_indices] # ( #mlm_positions, vocab_size)
            
            # sample 
            sampled_tokens = gumbel_sample(sampled_generator_logits)

            # scatter the sampled values back to the input
            disc_input_ids = input_ids.clone()
            disc_input_ids[masked_indices] = sampled_tokens.detach()

            # generate discriminator labels, with replaced as True and original as False
            disc_labels = masked_indices.clone()
            disc_labels[masked_indices] = (sampled_tokens != labels[masked_indices])
            disc_labels = disc_labels.float().detach()

        # get discriminator output and binary cross entropy loss
        disc_logits = self.discriminator(disc_input_ids, attention_mask, token_type_ids)[0]

        num_tokens = torch.numel(input_ids)
        if attention_mask is not None:
            active_disc_token = attention_mask.view(-1, num_tokens) == 1
            active_disc_logits = disc_logits.view(-1, num_tokens)[active_disc_token]
            active_disc_labels = disc_labels.view(-1, num_tokens)[active_disc_token]
            discriminator_loss = self.discriminator_loss_fct(active_disc_logits, active_disc_labels)
        else:
            discriminator_loss = self.discriminator_loss_fct(disc_logits.view(-1, num_tokens), disc_labels.view(-1, num_tokens))
        
        loss = self.gen_weight * generator_loss + self.disc_weight * discriminator_loss
        
        # gather metrics
        with torch.no_grad():
            gen_predictions = torch.argmax(generator_logits, dim=-1)
            disc_predictions = torch.round((torch.sign(disc_logits) + 1.0) * 0.5)
            gen_acc = (labels[masked_indices] == gen_predictions[masked_indices]).float().sum() / labels[masked_indices].numel()
            masked_disc_acc = (disc_labels[masked_indices] == disc_predictions[masked_indices]).float().sum() / disc_predictions[masked_indices].numel()
            unmasked_disc_acc = (disc_labels[~masked_indices] == disc_predictions[~masked_indices]).float().sum() / disc_predictions[~masked_indices].numel()
            disc_ones_ratio = disc_labels.float().sum() / num_tokens


        return ElectraModelOutput(
            loss=loss,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            gen_acc=gen_acc,
            masked_disc_acc=masked_disc_acc,
            unmasked_disc_acc=unmasked_disc_acc,
            disc_ones_ratio=disc_ones_ratio,
        )
        

        

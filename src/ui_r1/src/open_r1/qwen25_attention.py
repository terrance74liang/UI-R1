import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import dist
from typing import List, Tuple, Union, Optional

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
)


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


class QwenVLWithFeaturesOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    """
    Same as Qwen2_5_VLCausalLMOutputWithPast, but adds:

    - vision_hidden_states:  vision encoder outputs (image patch features)
    - text_hidden_states:    last hidden states of the text+image sequence
                             (before lm_head)
    """
    def __init__(
        self,
        vision_hidden_states=None,
        text_hidden_states=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.vision_hidden_states = vision_hidden_states
        self.text_hidden_states = text_hidden_states

class Qwen2_5_VLForConditionalGenerationWithAttention(Qwen2_5_VLForConditionalGeneration):
    """
    A thin wrapper around Qwen2_5_VLForConditionalGeneration that:

    - Behaves like the base model (loss, logits, past_key_values, hidden_states, attentions, rope_deltas)
    - Additionally returns:
        * vision_hidden_states: outputs of the vision encoder (image patch features)
        * text_hidden_states: last hidden states of the decoder (before lm_head)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # (batch_size, seq_len)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Union[Tuple, QwenVLWithFeaturesOutputWithPast]:

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if verbose:
            rank0_print(f"input_ids: {input_ids.shape}, {input_ids[0][:5]}...")
            if labels is not None:
                rank0_print(f"labels: {labels.shape}, {labels[0][:5]}...")
            if pixel_values is not None:
                rank0_print(f"pixel_values: {pixel_values.shape}")
            if image_grid_thw is not None:
                rank0_print(f"image_grid_thw: {image_grid_thw.shape}, {image_grid_thw}")

        vision_hidden_states = None  # we will store image_embeds here

        # ---- 1) Build inputs_embeds manually (same as GUI-Actor, but no pointer stuff) ---- #

        if inputs_embeds is None:
            # Text embeddings
            inputs_embeds = self.model.embed_tokens(input_ids)  # (batch_size, seq_len, d_model)

            # Vision encoder: images
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)  # (n_image_tokens_total, d_model)
                # Keep a copy as "vision hidden states"
                vision_hidden_states = image_embeds

                # Replace <image> token embeddings by image_embeds
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # ---- 2) RoPE position ids (copied from your existing code) ---- #
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # ---- 3) Run the transformer backbone ---- #
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # outputs[0] is the last hidden state of decoder (batch, seq_len, d_model)
        last_hidden_state = outputs[0]
        logits = self.lm_head(last_hidden_state)
        
        vision_tensor_list = []
        text_tensor_list = []

        # if vision_hidden_states is not None:
        batch_size, *_ = inputs_embeds.shape
        if vision_hidden_states is not None:
            for i in range(batch_size):
                token_ids = input_ids[i]
                visual_mask = (token_ids == self.config.image_token_id)
                visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1)

                target_mask = ((token_ids == self.config.pointer_start_token_id) | (token_ids == self.config.pointer_end_token_id))
                target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)

                vision_tensor_list.append(inputs_embeds[i][visual_indices])
                text_tensor_list.append(last_hidden_state[i][target_indices[0] + 1:target_indices[1]])
        
        # Standard LM loss (same as base Qwen2.5-VL LM head logic)
        lm_loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)

        # ---- 4) Return outputs ---- #
        if not return_dict:
            # HF tuple ordering: (loss?), logits, past_key_values, hidden_states, attentions, ...
            # We append our extra tensors at the end:
            if lm_loss is not None:
                return (
                    lm_loss,
                    logits,
                    *outputs[2:],  # past_key_values, hidden_states, attentions, ...
                    vision_tensor_list,
                    text_tensor_list,  # text_hidden_states
                )
            else:
                return (
                    logits,
                    *outputs[1:],  # past_key_values, hidden_states, attentions, ...
                    vision_tensor_list,
                    text_tensor_list,
                )

        # return_dict=True: build our extended output object
        return QwenVLWithFeaturesOutputWithPast(
            loss=lm_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            vision_hidden_states=vision_tensor_list,
            text_hidden_states=text_tensor_list,
        )

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import dist
# from typing import List, Tuple, Union, Optional

# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
#     Qwen2_5_VLCausalLMOutputWithPast,
#     Qwen2_5_VLForConditionalGeneration,
# )


# def rank0_print(*args):
#     if dist.is_initialized():
#         if dist.get_rank() == 0:
#             print(f"Rank {dist.get_rank()}: ", *args)
#     else:
#         print(*args)


# class QwenVLWithFeaturesOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
#     """
#     Same as Qwen2_5_VLCausalLMOutputWithPast, but adds:

#     - vision_hidden_states:  vision encoder outputs (image patch features)
#     - text_hidden_states:    last hidden states of the text+image sequence
#                              (before lm_head)
#     """
#     def __init__(
#         self,
#         vision_hidden_states=None,
#         text_hidden_states=None,
#         *args,
#         **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.vision_hidden_states = vision_hidden_states
#         self.text_hidden_states = text_hidden_states

# class Qwen2_5_VLForConditionalGenerationWithAttention(Qwen2_5_VLForConditionalGeneration):
#     """
#     A thin wrapper around Qwen2_5_VLForConditionalGeneration that:

#     - Behaves like the base model (loss, logits, past_key_values, hidden_states, attentions, rope_deltas)
#     - Additionally returns:
#         * vision_hidden_states: outputs of the vision encoder (image patch features)
#         * text_hidden_states: last hidden states of the decoder (before lm_head)
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.post_init()

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,  # (batch_size, seq_len)
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         pixel_values: Optional[torch.Tensor] = None,
#         pixel_values_videos: Optional[torch.FloatTensor] = None,
#         image_grid_thw: Optional[torch.LongTensor] = None,
#         video_grid_thw: Optional[torch.LongTensor] = None,
#         rope_deltas: Optional[torch.LongTensor] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         second_per_grid_ts: Optional[torch.Tensor] = None,
#         verbose: bool = False,
#     ) -> Union[Tuple, QwenVLWithFeaturesOutputWithPast]:

#         output_attentions = (
#             output_attentions if output_attentions is not None else self.config.output_attentions
#         )
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if verbose:
#             rank0_print(f"input_ids: {input_ids.shape}, {input_ids[0][:5]}...")
#             if labels is not None:
#                 rank0_print(f"labels: {labels.shape}, {labels[0][:5]}...")
#             if pixel_values is not None:
#                 rank0_print(f"pixel_values: {pixel_values.shape}")
#             if image_grid_thw is not None:
#                 rank0_print(f"image_grid_thw: {image_grid_thw.shape}, {image_grid_thw}")

#         vision_hidden_states = None  # we will store image_embeds here

#         # ---- 1) Build inputs_embeds manually (same as GUI-Actor, but no pointer stuff) ---- #

#         if inputs_embeds is None:
#             # Text embeddings
#             inputs_embeds = self.model.embed_tokens(input_ids)  # (batch_size, seq_len, d_model)

#             # Vision encoder: images
#             if pixel_values is not None:
#                 pixel_values = pixel_values.type(self.visual.dtype)
#                 image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)  # (n_image_tokens_total, d_model)
#                 # Keep a copy as "vision hidden states"
#                 vision_hidden_states = image_embeds

#                 # Replace <image> token embeddings by image_embeds
#                 n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#                 n_image_features = image_embeds.shape[0]
#                 if n_image_tokens != n_image_features:
#                     raise ValueError(
#                         f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                     )
#                 image_mask = (
#                     (input_ids == self.config.image_token_id)
#                     .unsqueeze(-1)
#                     .expand_as(inputs_embeds)
#                     .to(inputs_embeds.device)
#                 )
#                 image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#                 inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

#             if attention_mask is not None:
#                 attention_mask = attention_mask.to(inputs_embeds.device)

#         # ---- 2) RoPE position ids (copied from your existing code) ---- #
#         if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#             if (
#                 (cache_position is not None and cache_position[0] == 0)
#                 or self.rope_deltas is None
#                 or (past_key_values is None or past_key_values.get_seq_length() == 0)
#             ):
#                 position_ids, rope_deltas = self.get_rope_index(
#                     input_ids, image_grid_thw, video_grid_thw, attention_mask
#                 )
#                 self.rope_deltas = rope_deltas
#             else:
#                 batch_size, seq_length, _ = inputs_embeds.shape
#                 delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
#                 position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#                 position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#                 if cache_position is not None:
#                     delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#                     delta = delta.to(position_ids.device)
#                 position_ids = position_ids.add(delta)
#                 position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#         # ---- 3) Run the transformer backbone ---- #
#         outputs = self.model(
#             input_ids=None,
#             position_ids=position_ids,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             cache_position=cache_position,
#         )

#         # outputs[0] is the last hidden state of decoder (batch, seq_len, d_model)
#         last_hidden_state = outputs[0]
#         logits = self.lm_head(last_hidden_state)
        
#         vision_tensor_list = []
#         text_tensor_list = []

#         # if vision_hidden_states is not None:
#         batch_size, *_ = inputs_embeds.shape
#         if vision_hidden_states is not None:
#             for i in range(batch_size):
#                 token_ids = input_ids[i]
#                 visual_mask = (token_ids == self.config.image_token_id)
#                 visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1)

#                 target_mask = ((token_ids == self.config.pointer_start_token_id) | (token_ids == self.config.pointer_end_token_id))
#                 target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)

#                 vision_tensor_list.append(inputs_embeds[i][visual_indices])
#                 text_tensor_list.append(last_hidden_state[i][target_indices[0] + 1:target_indices[1]])
        
#         # Standard LM loss (same as base Qwen2.5-VL LM head logic)
#         lm_loss = None
#         if labels is not None:
#             logits = logits.float()
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             loss_fct = nn.CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             shift_labels = shift_labels.to(shift_logits.device)
#             lm_loss = loss_fct(shift_logits, shift_labels)

#         # ---- 4) Return outputs ---- #
#         if not return_dict:
#             # HF tuple ordering: (loss?), logits, past_key_values, hidden_states, attentions, ...
#             # We append our extra tensors at the end:
#             if lm_loss is not None:
#                 return (
#                     lm_loss,
#                     logits,
#                     *outputs[2:],  # past_key_values, hidden_states, attentions, ...
#                     vision_tensor_list,
#                     text_tensor_list,  # text_hidden_states
#                 ) 
#             else:
#                 return (
#                     logits,
#                     *outputs[1:],  # past_key_values, hidden_states, attentions, ...
#                     vision_tensor_list,
#                     text_tensor_list,
#                 )

#         # return_dict=True: build our extended output object
#         return QwenVLWithFeaturesOutputWithPast(
#             loss=lm_loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             rope_deltas=self.rope_deltas,
#             vision_hidden_states=vision_tensor_list,
#             text_hidden_states=text_tensor_list,
#         )

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLForConditionalGeneration
from typing import List, Tuple, Union, Optional
from dotenv import load_dotenv
import comet_ml
import os
from dataclasses import dataclass
from typing import Optional,List
# if dist.get_rank() == 0:
#     load_dotenv()
#     os.getenv('COMET_API_KEY')
#     env_key = os.getenv('ACTOR_EXP_NAME')
#     experiment_config = comet_ml.ExperimentConfig(name=env_key)
#     experiment = comet_ml.start(project_name="actor", experiment_config=experiment_config)
@dataclass
class QwenVLwithVisionHeadOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    """
    Output class for Qwen2_5_VL with pointer head, extending the base output class.
    
    Args:
        lm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss.
        pointer_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Vision pointer network loss.
        pointer_scores (`List[torch.FloatTensor]`, *optional*):
            Attention scores from the pointer network, one tensor per batch item.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Combined loss (weighted sum of lm_loss and pointer_loss).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores from the language modeling head.
        past_key_values, hidden_states, attentions, rope_deltas:
            Same as parent class.
    """

    lm_loss: Optional[torch.FloatTensor] = None
    pointer_loss: Optional[torch.FloatTensor] = None
    pointer_scores: Optional[List[torch.FloatTensor]] = None
    attention_pattern: Optional[List[torch.Tensor]] = None
    displaced_attention_pattern: Optional[List[torch.Tensor]] = None



class VisionHead_MultiPatch(nn.Module):
    def __init__(self,d_model, projection_dim, num_attention_heads=8, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Note: We omit additional normalization here because Qwen2VL
        # already normalizes hidden states using RMSNorm.
        self.projection_enc = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model)
        )
        self.projection_dec = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model)
        )

        # Add self-attention layer for visual features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization and residual connection
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                hidden_state_enc,  # shape: [n_enc, d_model] where n_enc can vary with image size
                hidden_state_dec,  # shape: [n_dec, d_model] there can be multiple query in one sample
                labels: Optional[torch.Tensor] = None,  # shape: [n_dec, n_enc], binary mask of patches in bbox
                do_single_patch: bool = False,
               ):
        
        enc_input = hidden_state_enc.unsqueeze(0)
        attn_output, _ = self.self_attention(
            query=enc_input,
            key=enc_input,
            value=enc_input,
            # attn_mask=attention_mask,
            need_weights=False
        )
        # Residual connection and layer normalization
        hidden_state_enc_ctx = self.layer_norm(enc_input + self.dropout(attn_output))
        # Remove batch dimension
        hidden_state_enc_ctx = hidden_state_enc_ctx.squeeze(0)  # [n_enc, d_model]

        # Apply the projection networks.
        proj_enc = self.projection_enc(hidden_state_enc_ctx)  # [n_enc, d_model]
        proj_dec = self.projection_dec(hidden_state_dec)  # [n_dec, d_model]
       
        # Compute scaled dot-product attention scores.
        # Scaling by sqrt(d_model) is critical regardless of variable n_enc.
        scaling = self.d_model ** 0.5
        patch_logits = torch.matmul(proj_dec, proj_enc.transpose(0, 1)) / scaling  # [n_dec, n_enc]
        
        # Softmax normalization is applied along the encoder dimension.
        attn_weights = F.log_softmax(patch_logits, dim=-1)

        loss = None
        supression_loss = None
        if (labels is not None) and (not do_single_patch):
            epsilon = 1e-8
            labels_float = labels.float()
            # Normalize each row to get target probability distribution
            target_dist = labels_float / (labels_float.sum(dim=-1, keepdim=True) + epsilon)

            # Apply log_softmax to logits
            pred_log_probs = F.log_softmax(patch_logits, dim=-1)
            # Use KL divergence as loss
            loss = F.kl_div(pred_log_probs, target_dist, reduction='batchmean')

            supression_loss = attn_weights* (labels == 0)

        # if do_single_patch and (labels is not None):
        #     loss = F.cross_entropy(attn_scores, labels)

        return attn_weights, loss, supression_loss
    
class VisionHead_MultiPatch_binary_heads(nn.Module):
    # d model is just the size of the hidden layer for qwen
    def __init__(self,d_model, projection_dim, num_attention_heads=8, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Note: We omit additional normalization here because Qwen2VL
        # already normalizes hidden states using RMSNorm.
        self.projection_enc = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, 1)
        )

        self.projection_dec_x = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model)
        )

        self.projection_dec_y = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model)
        )

        # Add self-attention layer for visual features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization and residual connection
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                thw_rows,
                thw_cols,
                hidden_state_enc,  # shape: [n_enc, d_model] where n_enc can vary with image size
                hidden_state_dec_x,
                hidden_state_dec_y,  # shape: [n_dec, d_model] there can be multiple query in one sample
                labels: Optional[torch.Tensor] = None,  # shape: [n_dec, n_enc], binary mask of patches in bbox
                do_single_patch: bool = False,
               ):
        
        enc_input = hidden_state_enc.unsqueeze(0)
        attn_output, _ = self.self_attention(
            query=enc_input,
            key=enc_input,
            value=enc_input,
            # attn_mask=attention_mask,
            need_weights=False
        )
        # Residual connection and layer normalization
        hidden_state_enc_ctx = self.layer_norm(enc_input + self.dropout(attn_output))
        # Remove batch dimension
        hidden_state_enc_ctx = hidden_state_enc_ctx.squeeze(0)  # [n_enc, 1]

        # n encoder is the number of visual patches and d model is the vector size attending to that patch 

        # Apply the projection networks.
        proj_enc = self.projection_enc(hidden_state_enc_ctx)  # [n_enc, d_model]
        proj_dec_x = self.projection_dec_x(hidden_state_dec_x)[:,0:thw_cols//2] # [n_dec, w]
        proj_dec_y = self.projection_dec_y(hidden_state_dec_y)[:,0:thw_rows//2] # [n_dec, h]

        coordinate_weights_with_dimension = torch.matmul(proj_dec_x[:,:,None],proj_dec_y[:,None, :]) # [ndec , w, h]

        coordinate_weights_without_dimension = coordinate_weights_with_dimension.reshape(coordinate_weights_with_dimension.shape[0], -1) #[ndec, h.w = n_enc]

        weighted_patches = coordinate_weights_without_dimension * proj_enc.squeeze(-1)
        
        # Compute scaled dot-product attention scores.
        # Scaling by sqrt(d_model) is critical regardless of variable n_enc.
        scaling = self.d_model ** 0.5
        patch_logits = weighted_patches / scaling  # [n_dec, n_enc]
        # n_dec is the number of trajectories and n_enc is the number f patches
        
        # Softmax normalization is applied along the encoder dimension.
        attn_weights = F.softmax(patch_logits, dim=-1)

        loss = None
        supression_loss = None
        if (labels is not None) and (not do_single_patch):
            epsilon = 1e-8
            labels_float = labels.float()
            # Normalize each row to get target probability distribution
            target_dist = labels_float / (labels_float.sum(dim=-1, keepdim=True) + epsilon)

            # Apply log_softmax to logits
            pred_log_probs = F.log_softmax(patch_logits, dim=-1)
            # Use KL divergence as loss
            loss = F.kl_div(pred_log_probs, target_dist, reduction='batchmean')

            supression_loss = attn_weights* (labels == 0)

        # if do_single_patch and (labels is not None):
        #     loss = F.cross_entropy(attn_scores, labels)

        return attn_weights, loss, supression_loss


class Qwen2_5_VLForConditionalGenerationWithAttention(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        # get rid of the extra arguemnts before passing to parent function
        self.mode = kwargs.pop("mode","v2p")
        super().__init__(*args, **kwargs)
        if self.mode == "binary_head":
            self.binary_multi_patch_pointer_head = VisionHead_MultiPatch_binary_heads(self.config.hidden_size, self.config.hidden_size)
        elif self.mode == "v2p" or self.mode == 'actor':
            self.multi_patch_pointer_head = VisionHead_MultiPatch(self.config.hidden_size, self.config.hidden_size)
        self.pointer_loss_weight = kwargs.get("pointer_loss_weight", 1.0)
        self.lm_loss_weight = kwargs.get("lm_loss_weight", 1.0)
        self.step = 0

        if dist.get_rank() == -1:
            load_dotenv()
            os.getenv('COMET_API_KEY')
            env_key = os.getenv('ACTOR_EXP_NAME')
            experiment_config = comet_ml.ExperimentConfig(name=env_key)
            self.experiment = comet_ml.start(project_name="actor", experiment_config=experiment_config)
        self.post_init()

    import torch

    def circular_shift(self,log_attn: torch.Tensor,H: int,W: int,dx: int,dy: int,jitter_std = 10,max_abs_shift = 10) -> torch.Tensor:
        """
        Circularly shift a flattened THW attention map stored as log-probs.

        Args:
            log_attn: (N,) log-probs, where N = T*H*W
            T,H,W: grid sizes (from image_grid_thw)
            dx: shift in x (cols), + = right, integer cells
            dy: shift in y (rows), + = down,  integer cells

        Returns:
            shifted_log_attn: (N,) log-probs after circular shift
        """
        H = H.to(torch.int16).item()
        W = W.to(torch.int16).item()
        device = log_attn.device
        N = log_attn.numel()
        expected =  H * W
        if N != expected:
            raise ValueError(f"log_attn has {N} elems but H*W={expected} (T,H,W={H,W})")
        
        jx = torch.randn((), device=device) * jitter_std
        jy = torch.randn((), device=device) * jitter_std

        mod_dx = int(torch.round(torch.tensor(dx, device=device) + jx).item())
        mod_dy = int(torch.round(torch.tensor(dy, device=device) + jy).item())

        # Clamp and also reduce modulo the grid size (wrap shift size doesn't need > W/H)
        mod_dx = max(-max_abs_shift, min(max_abs_shift, mod_dx))
        mod_dy= max(-max_abs_shift, min(max_abs_shift, mod_dy))

        # log-probs -> probs
        attn = log_attn.exp()              # (N,)

        # reshape to THW so roll is spatial (doesn't mix tokens incorrectly)
        attn_thw = attn.view( H, W)      # (T, H, W)

        # circular shift along H and W (leave T untouched)
        attn_thw = torch.roll(attn_thw, shifts=(mod_dy, mod_dx), dims=(0, 1))

        # renormalize (roll preserves sum, but this guards tiny numeric drift)
        attn_flat = attn_thw.reshape(-1)
        attn_flat = attn_flat / (attn_flat.sum() + 1e-6)

        # back to log
        return (attn_flat + 1e-12).log()

    def displace_attention(self,base_attn, coords, rl_head_out = [0.15, 0.10, -0.3, -0.3], stochastic=True):
        """
        base_attn: (N,) attention over patches, non-negative, sum~1
        coords:    (N, 2) patch coordinates in [0,1]
        rl_head_out: (4,) [Δμx, Δμy, Δlogσx, Δlogσy]
        """
        base_attn = base_attn.exp()
        device = base_attn.device
        base_attn = base_attn / (base_attn.sum() + 1e-6)  # normalize

        x = coords[0, :]
        y = coords[1, :]

        # --- 1) moments of original (possibly non-Gaussian) blob ---
        mu_x0 = (base_attn * x).sum()
        mu_y0 = (base_attn * y).sum()
        var_x0 = (base_attn * (x - mu_x0) ** 2).sum()
        var_y0 = (base_attn * (y - mu_y0) ** 2).sum()
        sigma_x0 = (var_x0 + 1e-6).sqrt()
        sigma_y0 = (var_y0 + 1e-6).sqrt()

        # --- 2) apply RL deltas to get new μ and σ ---
        delta_mu_x, delta_mu_y, delta_log_sigma_x, delta_log_sigma_y = torch.tensor(rl_head_out,device=device)

        # (optional) clamp for stability
        delta_mu_x = delta_mu_x.clamp(-0.3, 0.3)
        delta_mu_y = delta_mu_y.clamp(-0.3, 0.3)
        delta_log_sigma_x = delta_log_sigma_x.clamp(-2.0, 2.0)
        delta_log_sigma_y = delta_log_sigma_y.clamp(-2.0, 2.0)

        # new means (relative to original center)
        mu_x = (mu_x0 + delta_mu_x).clamp(0.0, 1.0)
        mu_y = (mu_y0 + delta_mu_y).clamp(0.0, 1.0)

        # new stds (scale original spread by exp(delta_log_sigma))
        log_sigma_x0 = sigma_x0.clamp(min=1e-3).log()
        log_sigma_y0 = sigma_y0.clamp(min=1e-3).log()
        sigma_x = (log_sigma_x0 + delta_log_sigma_x).exp().clamp(1e-3, 0.5)
        sigma_y = (log_sigma_y0 + delta_log_sigma_y).exp().clamp(1e-3, 0.5)

        # --- 3) sample center from N(μ, σ) (this is the "variation") ---
        if stochastic:
            eps_x = torch.randn((), device=coords.device)
            eps_y = torch.randn((), device=coords.device)
            c_x = (mu_x + sigma_x * eps_x).clamp(0.0, 1.0)
            c_y = (mu_y + sigma_y * eps_y).clamp(0.0, 1.0)
        else:
            c_x, c_y = mu_x, mu_y

        # --- 4) build Gaussian mask at sampled center ---
        gauss = torch.exp(
            -0.5 * (
                (x - c_x) ** 2 / (sigma_x ** 2 + 1e-6)
            + (y - c_y) ** 2 / (sigma_y ** 2 + 1e-6)
            )
        )
        gauss = gauss / (gauss.sum() + 1e-6)

        # --- 5) move the original blob by modulating with Gaussian ---
        moved_attn = base_attn * gauss
        moved_attn = moved_attn / (moved_attn.sum() + 1e-6)

        return moved_attn

        moved_log_attn = (moved_attn + 1e-12).log()  

        return moved_log_attn

    def shift_heatmap_circular(attn, dx, dy):
        """
        Circular (wrap-around) shift.
        attn: (H, W) or (B, H, W)
        dx: shift in x (columns), + = right, integer
        dy: shift in y (rows),   + = down, integer
        """
        if attn.dim() == 2:
            attn = attn.unsqueeze(0)  # (1, H, W)

        # torch.roll wraps by default
        out = torch.roll(attn, shifts=(dy, dx), dims=(-2, -1))

        # still a prob dist just in case of tiny numeric drift
        out = out / (out.sum(dim=(-1, -2), keepdim=True) + 1e-6)
        return out.squeeze(0)
    
    def reset_loss_weights(self, pointer_loss_weight, lm_loss_weight):
        self.pointer_loss_weight = pointer_loss_weight
        self.lm_loss_weight = lm_loss_weight
   
    def forward(self,
                input_ids: torch.LongTensor = None, # (batch_size, seq_len)
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
                # Grounding
                visual_token_indices_of_coordinates: Optional[torch.Tensor] = None, # shape: (batch_size, n_target); each element is the ground-truth index of the visual token that should be attended to for the corresponding target token
                multi_patch_labels: Optional[torch.Tensor] = None, # shape: list [(n_target, n_visual), ...]; binary mask of patches in bbox
                if_multi_patch: bool = True,
                coordinates: Optional[List[Tuple[float, float]]] = None,
                patch_indexes : Optional[dict] = None,
                patch_centers : Optional[list] = None,
                verbose: bool = False) -> Union[Tuple, QwenVLwithVisionHeadOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids) # shape: (batch_size, seq_len, d_model)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
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

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

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

        hidden_states = outputs[0] # shape: (batch_size, seq_len, d_model)

        logits = self.lm_head(hidden_states)

        lm_loss = None
        if labels is not None and self.lm_loss_weight > 0:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)


        # If vision supervision is requested, process the action head.
        pointer_loss = None
        supression_loss = None
        gaussian_loss = None
        pointer_scores = []

        batch_size = input_ids.shape[0]
        pointer_losses = []
        supression_losses = []
        gaussian_losses = []
        attention_pattern = []
        displaced_attention_pattern = []
        # Process each sample individually because the number of visual and target tokens may vary.
        if multi_patch_labels is not None:
            for i in range(batch_size):

                dummy_target = False

                # Get the token ids and corresponding hidden states for sample i.
                token_ids = input_ids[i]          # shape: (seq_length,)
                hs = hidden_states[i]             # shape: (seq_length, d_model)

                # Identify visual tokens indices.
                visual_mask = (token_ids == self.config.image_token_id)
                visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1) # shape: (n_visual,)

                # Identify target tokens (the ones that should attend to visual features).
                # contains all trajectories in a 1d tensor
                if self.mode == 'binary_head':
                    target_mask_x = (token_ids == self.config.pointer_x_pad_token_id)
                    target_indices_x = torch.nonzero(target_mask_x, as_tuple=False).squeeze(-1)
                    target_mask_y = (token_ids == self.config.pointer_y_pad_token_id)
                    target_indices_y = torch.nonzero(target_mask_y, as_tuple=False).squeeze(-1)
                elif self.mode == "v2p" or self.mode =='actor':
                    target_mask = (token_ids == self.config.pointer_pad_token_id)
                    target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
                    # print("target indices " + str(target_indices))
                    # print("target print" + str(token_ids[target_indices]))
                
                # If either visual or target tokens are missing, skip this sample.
                if visual_indices.numel() == 0:
                    raise ValueError(f"No visual or target tokens found for sample {i}.")
                
                if self.mode == "binary_head":
                    if target_indices_x.numel() == 0 or target_indices_y.numel() == 0:
                        target_indices_x = torch.tensor([hs.shape[0] - 1]) # take the last token as the dummy target token
                        target_indices_y = torch.tensor([hs.shape[0] - 1]) # take the last token as the dummy target token
                        if if_multi_patch:  # task the first 4 visual tokens as the ground truth
                            sample_labels = torch.zeros_like(visual_indices).unsqueeze(0)
                            sample_labels[0][:4] = 1

                        dummy_target = True
                    else:
                        if if_multi_patch:
                            sample_labels = multi_patch_labels[i]

                elif self.mode =='v2p' or self.mode =='actor':
                    if target_indices.numel() == 0:
                        target_indices = torch.tensor([hs.shape[0] - 1]) # take the last token as the dummy target token
                        if if_multi_patch:  # task the first 4 visual tokens as the ground truth
                            sample_labels = torch.zeros_like(visual_indices).unsqueeze(0)
                            sample_labels[0][:4] = 1

                        dummy_target = True
                    else:
                        # For supervision, we assume that visual_token_indices_of_coordinates[i] is a tensor of shape (n_target,)
                        # where each element is an integer in the range [0, n_visual-1] indicating the ground-truth visual token.
                        if if_multi_patch:
                            sample_labels = multi_patch_labels[i]
    
                
                # Gather the corresponding hidden state representations.
                # visual_hidden = hs[visual_indices]  # shape: (n_visual, d_model)
                # we have one vector for each token for targe tindices
                #  for visual embeds it is all the visual patches thw
                visual_embeds = inputs_embeds[i][visual_indices]
                # select the OS messages
                if self.mode == "binary_head":
                    target_hidden_x = hs[target_indices_x]
                    target_hidden_y = hs[target_indices_y]
                elif self.mode == "v2p" or self.mode =='actor':
                    target_hidden = hs[target_indices]  # shape: (n_target, d_model)
                    # print("target hidden result " + str(target_hidden))

                    # Ensure the number of targets matches between sample and labels
                if self.mode == "binary_head":
                    # if sample_labels.shape[0] != target_indices_x.shape[0]:
                    if len(sample_labels.shape) != len(target_indices_x.shape):
                        raise ValueError(f"Sample {i} has mismatched target counts: {sample_labels.shape[0]} labels but found {target_indices.shape[0]} target tokens")
    
                    attn_scores, loss_v, sup_loss = self.binary_multi_patch_pointer_head(image_grid_thw[i][1],
                                                                                            image_grid_thw[i][2],
                                                                                            visual_embeds,
                                                                                            target_hidden_x,
                                                                                            target_hidden_y,
                                                                                            labels=sample_labels)
                elif self.mode =="v2p" or self.mode =='actor':
                    # if sample_labels.shape[0] != target_indices.shape[0]:
                    if len(sample_labels.shape) != len(target_indices.shape):
                        # print("sample label shape " + str(sample_labels.shape))
                        # print("target indice shape " + str(target_indices.shape))
                        raise ValueError(f"Sample {i} has mismatched target counts: {sample_labels.shape[0]} labels but found {target_indices.shape[0]} target tokens")

                    # Process using VisionHead_MultiPatch
                    # we are currently here
                    attn_scores, loss_v, sup_loss = self.multi_patch_pointer_head(
                        visual_embeds,
                        target_hidden,
                        labels=sample_labels
                    )

                displaced_attn_score = self.circular_shift(attn_scores[0],image_grid_thw[i][1]/2,image_grid_thw[i][2]/2,torch.randint(-10,11,(1,)).item(),torch.randint(-10,11,(1,)).item())

                pointer_scores.append(attn_scores.detach().cpu())

                gaussian_scores = torch.sum((patch_indexes[i]*torch.log(patch_indexes[i]/attn_scores)), dim=1).mean()

                supression_scores = torch.sum(sup_loss, dim=1).mean()
            # this is a single value

                pointer_losses.append(loss_v * 0.0 if dummy_target else loss_v)

                supression_losses.append(supression_scores * 0.0 if dummy_target else supression_scores)

                gaussian_losses.append(gaussian_scores * 0.0 if dummy_target else gaussian_scores)

                attention_pattern.append(attn_scores[0])

                displaced_attention_pattern.append(displaced_attn_score)
                
                pointer_loss = torch.stack(pointer_losses).mean()

                supression_loss = torch.stack(supression_losses).mean()

                gaussian_loss = torch.stack(gaussian_losses).mean()

        if self.mode in ['v2p','binary_head'] and multi_patch_labels is not None:
            if lm_loss is None :
                total_loss = supression_loss + gaussian_loss
                if dist.get_rank() == 0:
                    # self.experiment.log_metrics({"supression loss":supression_loss,"gaussian loss":gaussian_loss,"total":total_loss}, step=self.step)
                    self.step += 1
            elif supression_loss is None or gaussian_loss is None:
                total_loss = self.lm_loss_weight * lm_loss
                if dist.get_rank() == 0:
                    # self.experiment.log_metrics({"lm loss":lm_loss,"total":total_loss}, step=self.step)
                    self.step += 1
            else:
                total_loss = self.lm_loss_weight * lm_loss + supression_loss + gaussian_loss
                if dist.get_rank() == 0:
                    # self.experiment.log_metrics({"supression loss":supression_loss,"gaussian loss":gaussian_loss,"lm loss":lm_loss,"total":total_loss}, step=self.step)
                    self.step += 1
        elif self.mode =="actor" and multi_patch_labels is not None:
            # Combine the LM loss and vision loss using the provided loss weights.
            if lm_loss is None:
                total_loss = pointer_loss
                # self.experiment.log_metrics({"pointer loss":pointer_loss,"total":total_loss}, step=self.step)
                self.step += 1
            elif pointer_loss is None:
                total_loss = lm_loss
                # self.experiment.log_metrics({"lm loss":lm_loss,"total":total_loss}, step=self.step)
                self.step += 1
            else:
                total_loss = self.lm_loss_weight * lm_loss + self.pointer_loss_weight * pointer_loss 
                # self.experiment.log_metrics({"pointer loss":pointer_loss,"lm loss":lm_loss,"total":total_loss}, step=self.step)
                self.step += 1
        else:
            total_loss = None

        if return_dict:
            return QwenVLwithVisionHeadOutputWithPast(
                lm_loss=lm_loss,
                pointer_loss=pointer_loss,
                pointer_scores=pointer_scores,
                loss=total_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
                attention_pattern=displaced_attention_pattern,
                displaced_attention_pattern=displaced_attention_pattern
            )
        else:
            # When labels are provided, parent's forward returns a tuple with loss as the first element.
            if labels is not None:
                # Replace the LM loss with the combined loss.
                output = (lm_loss, pointer_loss, logits, pointer_scores,) + outputs[1:]
                print(f"returning: total_loss, logits, pointer_scores, ...")
                return (total_loss,) + output if total_loss is not None else output
            else:
                return outputs


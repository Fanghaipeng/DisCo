from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision
from typing import Callable, Optional
from typing import Tuple, Optional, List, Union, Any, Type
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward

from einops import rearrange, repeat
import math


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(
    in_channels,
    motion_module_type: str, 
    motion_module_kwargs: dict
):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(in_channels=in_channels, **motion_module_kwargs,)    
    else:
        raise ValueError


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 2,
        attention_block_types              = ( False, True, ),
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 16,
        temporal_attention_dim_div         = 1,
        zero_initialize                    = True,
    ):
        super().__init__()
        
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )
        
        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, temb, encoder_hidden_states, attention_mask=None, anchor_frame_idx=None):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask)

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,

        num_layers,
        attention_block_types              = ( False, True, ),        
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 16,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)    
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length)
        
        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        
        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types              = ( False, True, ),
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 16,
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        
        for do_shift in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    do_shift=do_shift,
                    
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
        
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))
            
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)


    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                video_length=video_length,
            ) + hidden_states
            
        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        
        output = hidden_states  
        return output


class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 16
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VersatileAttention(CrossAttention):
    def __init__(
            self,
            do_shift                           = False,
            temporal_position_encoding         = False,
            temporal_position_encoding_max_len = 16,
            # NOTE：Window-related Hyperparameter
            window_size: Tuple[int, int, int] = (4, 4, 4),       
            feat_size: Tuple[int, int, int] = (16, 32, 32),
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.do_shift = do_shift
        
        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            dropout=0., 
            max_len=temporal_position_encoding_max_len
        ) if temporal_position_encoding else None

        # NOTE：Window-related Hyperparameter
        self.has_init = False
        self.window_size: Tuple[int, int, int] = window_size
        self.shift_size = (window_size[0]//2, window_size[1]//2, window_size[2]//2)

        self.feat_size: Tuple[int, int, int] = feat_size
        self.target_shift_size: Tuple[int, int, int] = self.shift_size
        self.window_size, self.shift_size = self._calc_window_shift(self.window_size)
        self.window_area = self.window_size[0] * self.window_size[1] * self.window_size[2]
        self.device = None
        self._make_attention_mask()

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}"

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    
    def window_partition(self, x, window_size: Tuple[int, int, int]):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, F, H, W, C = x.shape
        x = x.view(B, F//window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
        return windows

    def window_reverse(self, windows, window_size: Tuple[int, int, int], feat_size: Tuple[int, int, int]):
        """
        Args:
            windows: (num_windows * B, window_size[0], window_size[1], C)
            window_size (Tuple[int, int]): Window size
            img_size (Tuple[int, int]): Image size

        Returns:
            x: (B, H, W, C)
        """
        F, H, W = feat_size
        C = windows.shape[-1]
        x = windows.view(-1, F//window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(-1, H, W, C)
        return x

    def update_input_size(self, feat_size: Tuple[int, int, int], **kwargs: Any) -> None:
        """Method updates the window size and so the pair-wise relative positions

        Args:
            new_window_size (int): New window size
            kwargs (Any): Unused
        """
        # Set new window size and new pair-wise relative positions
        self.feat_size: Tuple[int, int, int] = feat_size
        self.target_shift_size: Tuple[int, int, int] = self.shift_size
        self.window_size, self.shift_size = self._calc_window_shift(self.window_size)
        self.window_area = self.window_size[0] * self.window_size[1] * self.window_size[2]

        self._make_attention_mask()

    def _calc_window_shift(self, target_window_size):
        window_size = [f if f <= w else w for f, w in zip(self.feat_size, target_window_size)]
        shift_size = [0 if f <= w else s for f, w, s in zip(self.feat_size, window_size, self.target_shift_size)]
        return tuple(window_size), tuple(shift_size)
    
    def _make_attention_mask(self) -> None:
        """Method generates the attention mask used in shift case."""
        # Make masks for shift case
        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            F, H, W = self.feat_size
            img_mask = torch.zeros((1, F, H, W, 1), device=self.device)  # 1 H W 1
            cnt = 0
            for f in (
                    slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None)):
                for h in (
                        slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None)):
                    for w in (
                            slice(0, -self.window_size[2]),
                            slice(-self.window_size[2], -self.shift_size[2]),
                            slice(-self.shift_size[2], None)):
                        img_mask[:, f, h, w, :] = cnt
                        cnt += 1
            mask_windows = self.window_partition(img_mask, self.window_size)  # num_windows, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    # def _attention(self, query, key, value, attn_mask=None):
    #     if self.upcast_attention:
    #         query = query.float()
    #         key = key.float()

    #     # attention_scores = torch.baddbmm(
    #     #     torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
    #     #     query,
    #     #     key.transpose(-1, -2),
    #     #     beta=0,
    #     #     alpha=self.scale,
    #     # )
    #     # if attn_mask is not None:
    #     #     attention_scores = attention_scores + attn_mask

    #     # if self.upcast_softmax:
    #     #     attention_scores = attention_scores.float()

    #     # attention_probs = attention_scores.softmax(dim=-1)

    #     # # cast back to the original dtype
    #     # attention_probs = attention_probs.to(value.dtype)

    #     attention_probs = self.get_attention_scores(query, key, attn_mask)

    #     # compute attention output
    #     hidden_states = torch.bmm(attention_probs, value)

    #     # reshape hidden_states
    #     hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
    #     return hidden_states

    def window_attention(self, hidden_states):

        batch_size, sequence_length, dim = hidden_states.shape

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        B = batch_size // self.attn_mask.shape[0]
        attn_mask = self.attn_mask.repeat_interleave(B * self.heads, dim=0)

        attn_mask = attn_mask.to(query.dtype)
        attention_probs = self.get_attention_scores(query, key, attn_mask)

        hidden_states = torch.bmm(attention_probs, value)

        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states
    
    def _shifted_window_attn(self, x, do_shift = False):

        B, F, H, W, C = x.shape

        if do_shift:
            sf, sh, sw = self.shift_size
            # FIXME PyTorch XLA needs cat impl, roll not lowered
            # x = torch.cat([x[:, sh:], x[:, :sh]], dim=1)
            # x = torch.cat([x[:, :, sw:], x[:, :, :sw]], dim=2)
            x = torch.roll(x, shifts=(-sf, -sh, -sw), dims=(1, 2, 3))

        # partition windows
        x_windows = self.window_partition(x, self.window_size)  # num_windows * B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_area, C)

        # W-MSA/SW-MSA
        attn_windows = self.window_attention(hidden_states=x_windows)  # num_windows * B, window_area, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        x = self.window_reverse(attn_windows, self.window_size, self.feat_size)  # BF H' W' C

        # reverse cyclic shift
        if do_shift:
            # FIXME PyTorch XLA needs cat impl, roll not lowered
            # x = torch.cat([x[:, -sh:], x[:, :-sh]], dim=1)
            # x = torch.cat([x[:, :, -sw:], x[:, :, :-sw]], dim=2)
            x = torch.roll(x, shifts=(sf, sh, sw), dims=(1, 2, 3))

        return x

    def forward(self, hidden_states, attention_mask=None, video_length=None):

        BF, HW, C = hidden_states.shape
        F = video_length
        B = BF // F
        H = W = int(HW ** 0.5)

        hidden_states = rearrange(hidden_states, "(b f) hw c -> (b hw) f c", f=F)
            
        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)
            
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        hidden_states = rearrange(hidden_states, "(b h w) f c -> b f h w c", b=B,h=H,w=W)
        if not self.has_init:
            feat_size: Tuple[int, int, int] = (F, H, W)
            self.device = hidden_states.device
            self.update_input_size(feat_size)
            self.has_init = True
        hidden_states = self._shifted_window_attn(hidden_states, do_shift=self.do_shift)
        hidden_states = rearrange(hidden_states, "(bf) h w c -> (bf) (h w) c")

        return hidden_states


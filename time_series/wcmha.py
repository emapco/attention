from typing import Literal

import numpy as np
import scipy.interpolate
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Code based on: "Powerformer: A Transformer with Weighted Causal Attention for Time-series Forecasting"

# https://arxiv.org/abs/2502.06151


# A.2 Code to Calculate the Gain
def butterworth_filter(scale, order, times):
    b, a = scipy.signal.butter(order, 0.8, "lowpass", analog=False)
    t, decay = scipy.signal.freqz(b, a)
    t = scale * t / 2
    dc = 5 * np.log(np.abs(decay))
    decay_interp = scipy.interpolate.interp1d(t, dc)
    return decay_interp(times)


# 3.1 Weighted Causal Multihead Attention
class WeightedCausalMultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        locality_func: Literal[
            "similarity_power_law", "weighted_power_law", "butterworth_filter"
        ] = ("similarity_power_law"),
        alpha=1.0,
        butterworth_tc=None,
    ):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

        self.f_type = locality_func
        self.alpha = alpha
        if locality_func == "butterworth_filter":
            assert butterworth_tc is not None, "Butterworth filter requires cutoff time"
            self.butterworth_tc = butterworth_tc

    def forward(self, x):
        _, T, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = rearrange(Q, "b t (h d) -> b h t d", h=self.num_heads)
        K = rearrange(K, "b t (h d) -> b h t d", h=self.num_heads)
        V = rearrange(V, "b t (h d) -> b h t d", h=self.num_heads)

        # S_h
        scores = torch.einsum("b h i d, b h j d -> b h i j", Q, K) / (
            self.head_dim**0.5
        )

        # Causal mask to prevent future leakage
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        # S_h^(C) = S_h + M^(C)
        scores.masked_fill_(causal_mask, float("-inf"))

        time_idx = torch.arange(T, device=x.device)
        delta = time_idx.unsqueeze(1) - time_idx.unsqueeze(0)  # delta[i,j] = i - j
        if self.f_type == "similarity_power_law":
            decaying_mask = -(delta.float() ** self.alpha)
            decaying_mask = torch.where(
                delta < 0, torch.zeros_like(decaying_mask), decaying_mask
            )
        elif self.f_type == "weighted_power_law":
            safe_delta = torch.where(
                delta > 0, delta.float(), torch.ones_like(delta.float())
            )
            decaying_mask = -self.alpha * torch.log(safe_delta)
            decaying_mask = torch.where(
                delta <= 0, torch.zeros_like(decaying_mask), decaying_mask
            )
        elif self.f_type == "butterworth_filter":
            positive_delta = delta.float().clamp(min=0).cpu().numpy()
            butter_mask = butterworth_filter(
                self.butterworth_tc, int(self.alpha), positive_delta
            )
            decaying_mask = torch.tensor(butter_mask, device=x.device, dtype=x.dtype)
        else:
            raise ValueError("Unknown locality function")

        scores = scores + decaying_mask.unsqueeze(0).unsqueeze(
            0
        )  # S_h^(C, D) = S_h + M^(C) + M^(D)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, V)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.W_o(out)

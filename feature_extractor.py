import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

class FeatureExtractor:
    def __init__(self, embedding_matrix: Optional[torch.Tensor] = None,
                 kernel_target: str = 'logits',
                 pooling_method: str = 'max',
                 top_k: int = 0,
                 seq_len_scale: int = 64,
                 use_confidence_weighting: bool = True,
                 ignore_token_ids: List = []):

        self.embedding_matrix = embedding_matrix
        self.kernel_target = kernel_target
        self.pooling_method = pooling_method
        self.top_k = top_k
        self.seq_len_scale = seq_len_scale
        self.use_confidence_weighting = use_confidence_weighting
        self.ignore_token_ids = ignore_token_ids

    def extract(self, logit_k: torch.Tensor, mask_k: torch.Tensor, x_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if logit_k.dim() == 2: logit_k = logit_k.unsqueeze(0)
        if mask_k.dim() == 1: mask_k = mask_k.unsqueeze(0)
        if x_k.dim() == 1: x_k = x_k.unsqueeze(0)

        probs = torch.softmax(logit_k, dim=-1)

        if self.ignore_token_ids:
            probs[..., self.ignore_token_ids] = 0.0

        if self.top_k > 0:
            vals, indices = torch.topk(probs, k=self.top_k, dim=-1)
            probs_in_ = torch.zeros_like(probs)
            probs_in_.scatter_(2, indices, vals)
        else:
            probs_in_ = probs

        probs_in = torch.zeros_like(probs_in_)
        probs_in[mask_k] = probs_in_[mask_k]

        if (~mask_k).any():
            one_hot_tokens = F.one_hot(x_k[~mask_k], num_classes=probs_in.shape[-1])
            probs_in[~mask_k] = one_hot_tokens.to(dtype=probs_in.dtype)

        if self.kernel_target == "embeddings" and self.embedding_matrix is not None:
            W = self.embedding_matrix.to(probs_in.device).detach()
            features = torch.matmul(probs_in, W)
        else:
            features = probs_in

        if self.pooling_method == "max":
            vecs = features.max(dim=1).values
        elif self.pooling_method == "mean":
            vecs = features.mean(dim=1)
        elif self.pooling_method == "positional":
            seq_len_scale = max(self.seq_len_scale, x_k.shape[-1])
            omega = (torch.pi / 2.0) / seq_len_scale
            pos_indices = torch.arange(features.shape[1], device=logit_k.device).view(1, -1, 1)
            angles = pos_indices.float() * omega
            real_part = (features * torch.cos(angles)).sum(dim=1)
            imag_part = (features * torch.sin(angles)).sum(dim=1)
            vecs = torch.cat([real_part, imag_part], dim=-1)
        else:
            raise NotImplementedError(f"Pooling method {self.pooling_method} not implemented")

        norm_vec = F.normalize(vecs, p=2, dim=1)

        if self.use_confidence_weighting:
            all_max_vals = probs_in.max(dim=-1).values
            masked_max_vals = all_max_vals * mask_k.float()
            num_masked = mask_k.sum(dim=1).clamp(min=1.0)
            quality = masked_max_vals.sum(dim=1) / num_masked
        else:
            quality = torch.ones(logit_k.shape[0], device=logit_k.device)

        return norm_vec, quality
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from feature_extractor import FeatureExtractor


def _normalize_gradient(grad: torch.Tensor, protected_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
    if protected_tokens is not None:
        if grad.dim() == 3:
            grad.index_fill_(2, protected_tokens, 0.0)
        elif grad.dim() == 2:
            grad.index_fill_(1, protected_tokens, 0.0)

    token_norms = torch.norm(grad, p=2, dim=-1, keepdim=True)
    max_val_dim = 1 if grad.dim() == 3 else 0
    max_norms = token_norms.max(dim=max_val_dim, keepdim=True).values.clamp(min=1e-8)

    grad_safe = torch.where(max_norms > 0, grad / max_norms, grad)
    return grad_safe


class DiverseStrategy(ABC):
    def __init__(self, alpha: float, quality_scale: float, feature_extractor: FeatureExtractor):
        self.alpha = alpha
        self.quality_scale = quality_scale
        self.feature_extractor = feature_extractor

    @abstractmethod
    def apply(self, logits: torch.Tensor, mask_index: torch.Tensor, x: torch.Tensor,
              history_vecs: List[torch.Tensor], history_qualities: List[float],
              protected_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        pass


# baseline strategy with no diversity intervention
class BaselineStrategy(DiverseStrategy):
    def __init__(self):
        super().__init__(0.0, 0.0, None)
        self.alpha = 0

    def apply(self, logits, mask_index, x, history_vecs, history_qualities, protected_tokens=None):
        return logits, {}


# ODD strategy as outlined in paper
class ODDStrategy(DiverseStrategy):
    def apply(self, logits, mask_index, x, history_vecs, history_qualities, protected_tokens=None):
        metadata = {"force_map": []}
        active_logits = logits[mask_index].detach().clone().requires_grad_(True)
        logits_in = logits.detach().clone()
        logits_in[mask_index] = active_logits

        all_norm_vecs, all_quals = self.feature_extractor.extract(logits_in, mask_index, x)
        total_loss = 0
        current_basis = [h.detach().flatten() for h in history_vecs]

        for k in range(logits.shape[0]):
            with torch.no_grad():
                v_clean = all_norm_vecs[k].flatten()
                resid = v_clean.clone()
                for b in current_basis:
                    proj = torch.dot(resid, b)
                    resid = resid - proj * b

                norm = torch.norm(resid)
                target_dir = resid / norm if norm > 1e-6 else None

            if current_basis and target_dir is not None:
                align = torch.dot(all_norm_vecs[k].flatten(), target_dir)
                loss_k = -align * (self.quality_scale * all_quals[k])
                total_loss = total_loss + loss_k

            if target_dir is not None:
                current_basis.append(target_dir)

        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            active_grads = torch.autograd.grad(total_loss, active_logits)[0]
            with torch.no_grad():
                final_grads = torch.zeros_like(logits)
                final_grads[mask_index] = active_grads
                final_grads = _normalize_gradient(final_grads, protected_tokens)
                update = self.alpha * final_grads
                metadata["force_map"] = torch.norm(update, p=2, dim=-1).detach().float().cpu()
                logits.sub_(update)

        return logits, metadata


# DPP strategy adapted from DiverseFlow
class DPPStrategy(DiverseStrategy):
    def apply(self, logits: torch.Tensor, mask_index: torch.Tensor, x: torch.Tensor,
              history_vecs: List[torch.Tensor], history_qualities: List[float],
              protected_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        metadata = {"entropy_map": [], "force_map": []}

        logits_in = logits.detach().clone().requires_grad_(True)
        norm_vecs, quals = self.feature_extractor.extract(logits_in, mask_index, x)

        K = torch.mm(norm_vecs, norm_vecs.t())
        identity = torch.eye(K.shape[0], device=K.device)
        jitter = 1e-4

        q_mat = torch.outer(quals, quals)
        L = K * (1 + self.quality_scale * q_mat)

        loss = -(torch.logdet(L + jitter * identity) - torch.logdet(L + identity + jitter * identity))

        raw_grads = torch.autograd.grad(loss, logits_in)[0]
        final_grads = _normalize_gradient(raw_grads, protected_tokens)

        update = self.alpha * final_grads
        metadata["force_map"] = torch.norm(update, p=2, dim=-1).detach().float().cpu()

        return logits - update, metadata


# strategy where we maximise projection onto random direction, rather than orthogonal component
# initial results show worse performance
class RandomProbeStrategy(DiverseStrategy):
    def apply(self, logits: torch.Tensor, mask_index: torch.Tensor, x: torch.Tensor,
              history_vecs: List[torch.Tensor], history_qualities: List[float],
              protected_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:

        metadata = {"entropy_map": [], "force_map": []}
        current_logits = logits.clone().detach()
        final_grads = torch.zeros_like(logits)

        local_history_vecs = list(history_vecs)

        for k in range(logits.shape[0]):
            logit_k = logits[k].unsqueeze(0).detach().clone().requires_grad_(True)
            norm_vec_k, qual_k = self.feature_extractor.extract(
                logit_k, mask_index[k].unsqueeze(0), x[k].unsqueeze(0)
            )

            if k > 0 or local_history_vecs:

                ortho_basis = []
                for q in local_history_vecs:
                    u = q.clone()
                    for b in ortho_basis:
                        u = u - torch.dot(u.view(-1), b.view(-1)) * b

                    u_norm = torch.norm(u)
                    if u_norm > 1e-6:
                        ortho_basis.append(u / u_norm)

                probe = torch.randn_like(norm_vec_k).detach()

                for b in ortho_basis:
                    proj = torch.dot(probe.view(-1), b.view(-1)) * b
                    probe = probe - proj

                probe_norm = torch.norm(probe)
                if probe_norm > 1e-6:
                    target_dir = probe / probe_norm
                    alignment = torch.dot(norm_vec_k.view(-1), target_dir.view(-1))
                    loss = -alignment * (self.quality_scale * qual_k)

                    if loss.requires_grad:
                        raw_grads = torch.autograd.grad(loss, logit_k)[0]
                        final_grads[k] = _normalize_gradient(raw_grads, protected_tokens).squeeze(0)

            if k > 0 or local_history_vecs:
                current_logits[k] -= (self.alpha * final_grads[k])

            with torch.no_grad():
                norm_vec_new, _ = self.feature_extractor.extract(
                    current_logits[k].unsqueeze(0),
                    mask_index[k].unsqueeze(0),
                    x[k].unsqueeze(0)
                )
                local_history_vecs.append(norm_vec_new)

        update = self.alpha * final_grads
        metadata["force_map"] = torch.norm(update, p=2, dim=-1).detach().float().cpu()
        return logits - update, metadata


# variant of ODD where we update the logits in the inner loop, so each step is aware of where the previous sample
# lands after update. Much more expensive (backprop B - 1 times), and results are similar
class OrthogonalProjectionStrategy(DiverseStrategy):
    def apply(self, logits, mask_index, x, history_vecs, history_qualities, protected_tokens=None):
        metadata = {"force_map": []}

        current_logits = logits.clone().detach()
        final_grads = torch.zeros_like(logits)

        basis_vectors = [h.detach().flatten() for h in history_vecs]

        def project_resid_mgs(target, basis_list):
            resid = target.clone()
            for b in basis_list:
                proj = torch.dot(resid, b)
                resid = resid - proj * b
            return resid

        for k in range(logits.shape[0]):
            if basis_vectors:
                logit_k_leaf = logits[k].unsqueeze(0).detach().clone().requires_grad_(True)

                norm_vec_k, qual_k = self.feature_extractor.extract(
                    logit_k_leaf,
                    mask_index[k].unsqueeze(0),
                    x[k].unsqueeze(0)
                )

                v_raw_detached = norm_vec_k.detach().flatten()
                v_ortho = project_resid_mgs(v_raw_detached, basis_vectors)

                target_norm = torch.norm(v_ortho)

                if target_norm > 1e-6:
                    target_dir = v_ortho / target_norm

                    alignment = torch.dot(norm_vec_k.flatten(), target_dir)
                    loss = -alignment * (self.quality_scale * qual_k)

                    g = torch.autograd.grad(loss, logit_k_leaf)[0]
                    final_grads[k] = _normalize_gradient(g, protected_tokens).squeeze(0)

                    del g, loss, logit_k_leaf, norm_vec_k

            if k > 0 or history_vecs:
                current_logits[k] -= (self.alpha * final_grads[k])

            with torch.no_grad():
                new_vec, _ = self.feature_extractor.extract(
                    current_logits[k].unsqueeze(0),
                    mask_index[k].unsqueeze(0),
                    x[k].unsqueeze(0)
                )
                new_vec_flat = new_vec.flatten()

                v_final = project_resid_mgs(new_vec_flat, basis_vectors)
                norm = torch.norm(v_final)

                if norm > 1e-6:
                    basis_vectors.append(v_final / norm)

        update = self.alpha * final_grads
        metadata["force_map"] = torch.norm(update, p=2, dim=-1).detach().float().cpu()

        return logits - update, metadata


def get_strategy(name: str, alpha: float, quality_scale: float, feature_extractor: FeatureExtractor) -> DiverseStrategy:
    if name == "orthogonal_projection":
        return OrthogonalProjectionStrategy(alpha, quality_scale, feature_extractor)
    elif name == "random_probe":
        return RandomProbeStrategy(alpha, quality_scale, feature_extractor)
    elif name == "dpp":
        return DPPStrategy(alpha, quality_scale, feature_extractor)
    elif name == "baseline":
        return BaselineStrategy()
    elif name == 'odd':
        return ODDStrategy(alpha, quality_scale, feature_extractor)
    else:
        raise ValueError(f"Unknown strategy: {name}")
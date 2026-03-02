from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F

from strategies import DPPStrategy


def sample_gumbel_efficient_paired(logits, logits_orig, temperature):
    if temperature == 0:
        return torch.argmax(logits, dim=-1), torch.argmax(logits_orig, dim=-1)

    output_indices = torch.empty(logits.shape[:-1], dtype=torch.long, device=logits.device)
    output_indices_orig = torch.empty(logits_orig.shape[:-1], dtype=torch.long, device=logits_orig.device)

    for i in range(logits.shape[0]):
        logit_slice = logits[i].to(torch.float64)
        logit_slice_orig = logits_orig[i].to(torch.float64)

        noise = torch.rand_like(logit_slice, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature

        noisy_logits = logit_slice.exp() / gumbel_noise
        output_indices[i] = torch.argmax(noisy_logits, dim=-1)

        noisy_logits_orig = logit_slice_orig.exp() / gumbel_noise
        output_indices_orig[i] = torch.argmax(noisy_logits_orig, dim=-1)

    return output_indices, output_indices_orig


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = min(steps, mask_num.max().item()) if mask_num.max().item() > 0 else steps
    if steps == 0: steps = 1
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


class AppGenerator:
    def __init__(self, model, tokenizer, strategy: DPPStrategy, mask_token_id: int):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.mask_token_id = mask_token_id
        self.device = model.device

    def generate(self, prompt: str, batch_size: int, steps: int, gen_length: int, temperature: float) -> Tuple[
        List[Dict], List[str]]:
        messages = [{"role": "user", "content": prompt}]
        prompt_str = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        encoded = self.tokenizer([prompt_str] * batch_size, return_tensors="pt", padding=True, add_special_tokens=False)
        prompt_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

        prompt_len = prompt_ids.shape[1]
        x = torch.full((batch_size, prompt_len + gen_length), self.mask_token_id, dtype=torch.long).to(self.device)
        x[:, :prompt_len] = prompt_ids.clone()

        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=self.device)],
            dim=-1)

        mask_index_init = (x[:, prompt_len:] == self.mask_token_id)
        num_transfer_tokens_schedule = get_num_transfer_tokens(mask_index_init, steps)
        protected_tokens = torch.tensor([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id], device=self.device)

        history_frames = []

        def _dec(tid):
            s = self.tokenizer.decode([tid]).replace("Ġ", " ").replace("\n", "⏎")
            if self.tokenizer.mask_token: s = s.replace(self.tokenizer.mask_token, "[MASK]")
            return s

        for i in range(steps):
            mask_index = (x == self.mask_token_id)

            with torch.no_grad():
                logits = self.model(x, attention_mask=attention_mask).logits
                gen_logits = logits[:, prompt_len:, :].clone()

                # Baseline tracking
                gen_logits_orig = gen_logits.clone()
                probs_original = torch.softmax(gen_logits_orig, dim=-1)
                k_val = 5
                topk_probs_orig, topk_indices_orig = torch.topk(probs_original, k=k_val, dim=-1)

            curr_alpha = self.strategy.alpha * (1 - (i / steps))
            original_alpha = self.strategy.alpha
            self.strategy.alpha = curr_alpha

            metadata = {"entropy_map": torch.zeros(batch_size, gen_length),
                        "force_map": torch.zeros(batch_size, gen_length)}

            if curr_alpha > 0.0:
                gen_logits_guided, meta = self.strategy.apply(
                    gen_logits, mask_index=mask_index[:, prompt_len:], x=x[:, prompt_len:],
                    history_vecs=[], history_qualities=[], protected_tokens=protected_tokens
                )
                logits[:, prompt_len:, :] = gen_logits_guided

                probs = torch.softmax(gen_logits_guided, dim=-1)
                log_probs = torch.log_softmax(gen_logits_guided, dim=-1)
                metadata["entropy_map"] = -torch.sum(probs * log_probs, dim=-1).detach().float().cpu()
                if "force_map" in meta: metadata["force_map"] = meta["force_map"]

            self.strategy.alpha = original_alpha

            with torch.no_grad():
                gen_logits_final = logits[:, prompt_len:, :]
                probs_final = torch.softmax(gen_logits_final, dim=-1)

                topk_probs_final, topk_indices_final = torch.topk(probs_final, k=k_val, dim=-1)
                topk_probs_original_at_final = torch.gather(probs_original, -1, topk_indices_final)
                topk_probs_final_at_orig = torch.gather(probs_final, -1, topk_indices_orig)

                full_logits_orig = logits.clone()
                full_logits_orig[:, prompt_len:, :] = gen_logits_orig

                x0, x0_orig = sample_gumbel_efficient_paired(logits, full_logits_orig, temperature)
                flips = (x0 != x0_orig)[:, prompt_len:]

                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf')).to(x0_p.device))

                p_orig = F.softmax(full_logits_orig, dim=-1)
                x0_orig_p = torch.squeeze(torch.gather(p_orig, dim=-1, index=torch.unsqueeze(x0_orig, -1)), -1)
                confidence_orig = torch.where(mask_index, x0_orig_p, torch.tensor(-float('inf')).to(x0_orig_p.device))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                transfer_index_orig = torch.zeros_like(x0_orig, dtype=torch.bool, device=x0_orig.device)

                for j in range(batch_size):
                    k = num_transfer_tokens_schedule[j, i]
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True

                        _, select_index_orig = torch.topk(confidence_orig[j], k=k)
                        transfer_index_orig[j, select_index_orig] = True

                x_display = x.clone()
                x_display[transfer_index] = x0[transfer_index]

                frame_data = {"step": i, "alpha": float(curr_alpha), "batches": []}

                for b in range(batch_size):
                    raw_ids = x_display[b, prompt_len:].tolist()
                    orig_ids = x0_orig[b, prompt_len:].tolist()

                    display_tokens = []
                    orig_sampled_tokens = []
                    special_mask = []

                    for tid, orig_tid in zip(raw_ids, orig_ids):
                        if tid == self.mask_token_id:
                            display_tokens.append("[MASK]")
                            special_mask.append(False)
                        else:
                            display_tokens.append(_dec(tid))
                            special_mask.append(
                                tid == self.tokenizer.eos_token_id or tid == self.tokenizer.pad_token_id)

                        if orig_tid == self.mask_token_id:
                            orig_sampled_tokens.append("[MASK]")
                        else:
                            orig_sampled_tokens.append(_dec(orig_tid))

                    frame_data["batches"].append({
                        "tokens": display_tokens,
                        "orig_sampled_tokens": orig_sampled_tokens,
                        "is_mask": [tid == self.mask_token_id for tid in x[b, prompt_len:].tolist()],
                        "is_special": special_mask,
                        "is_flip": flips[b].cpu().tolist(),
                        "is_unmasked_next": transfer_index[b, prompt_len:].cpu().tolist(),
                        "is_unmasked_next_orig": transfer_index_orig[b, prompt_len:].cpu().tolist(),
                        "top_k_tokens": [[_dec(t) for t in seq] for seq in topk_indices_final[b].tolist()],
                        "top_k_probs": topk_probs_final[b].tolist(),
                        "top_k_probs_original": topk_probs_original_at_final[b].tolist(),
                        "top_k_orig_tokens": [[_dec(t) for t in seq] for seq in topk_indices_orig[b].tolist()],
                        "top_k_orig_probs": topk_probs_orig[b].tolist(),
                        "top_k_orig_probs_final": topk_probs_final_at_orig[b].tolist(),
                        "entropy": metadata["entropy_map"][b].tolist() if len(metadata["entropy_map"]) > 0 else [],
                        "force": metadata["force_map"][b].tolist() if len(metadata["force_map"]) > 0 else []
                    })
                history_frames.append(frame_data)
                x[transfer_index] = x0[transfer_index]

        final_frame = {"step": steps, "alpha": 0.0, "batches": []}
        for b in range(batch_size):
            raw_ids = x[b, prompt_len:].tolist()
            display_tokens = []
            special_mask = []
            for tid in raw_ids:
                display_tokens.append(_dec(tid))
                special_mask.append(tid == self.tokenizer.eos_token_id or tid == self.tokenizer.pad_token_id)

            final_frame["batches"].append({
                "tokens": display_tokens,
                "is_mask": [tid == self.mask_token_id for tid in raw_ids],
                "is_special": special_mask,
                "entropy": [], "force": []
            })
        history_frames.append(final_frame)

        samples = self.tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)
        return history_frames, samples
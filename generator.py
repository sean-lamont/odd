import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from strategies import DPPStrategy


# loop through batch and cast to 64 bit, rather than run sequentially.
# Saves significant VRAM with low time cost, with same result.
def sample_gumbel_efficient(logits, temperature):
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    output_indices = torch.empty(logits.shape[:-1], dtype=torch.long, device=logits.device)
    for i in range(logits.shape[0]):
        logit_slice = logits[i].to(torch.float64)
        noise = torch.rand_like(logit_slice, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        noisy_logits = logit_slice.exp() / gumbel_noise
        output_indices[i] = torch.argmax(noisy_logits, dim=-1)
    return output_indices


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


class DiverseGenerator:
    def __init__(self, model, tokenizer, strategy: DPPStrategy, mask_token_id: int):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.mask_token_id = mask_token_id
        self.device = model.device

    def generate(self, prompt: str, batch_size: int, steps: int, gen_length: int, temperature: float) -> Tuple[List[Dict], List[str]]:
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

        for i in range(steps):
            mask_index = (x == self.mask_token_id)

            with torch.no_grad():
                logits = self.model(x, attention_mask=attention_mask).logits
                gen_logits = logits[:, prompt_len:, :].clone()

            curr_alpha = self.strategy.alpha * (1 - (i / steps))
            original_alpha = self.strategy.alpha
            self.strategy.alpha = curr_alpha

            if curr_alpha > 0.0:
                gen_logits_guided, _ = self.strategy.apply(
                    gen_logits,
                    mask_index=mask_index[:, prompt_len:],
                    x=x[:, prompt_len:],
                    history_vecs=[], history_qualities=[], protected_tokens=protected_tokens
                )
                logits[:, prompt_len:, :] = gen_logits_guided

            self.strategy.alpha = original_alpha

            with torch.no_grad():
                x0 = sample_gumbel_efficient(logits, temperature)

                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf')).to(x0_p.device))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(batch_size):
                    k = num_transfer_tokens_schedule[j, i]
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True

                x[transfer_index] = x0[transfer_index]

        samples = self.tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)
        return [], samples
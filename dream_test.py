import time

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from feature_extractor import FeatureExtractor
from strategies import get_strategy


def main():
    print(">>> Loading DREAM Model...")
    model_path = "Dream-org/Dream-v0-Instruct-7B"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModel.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )

    model.eval()


    # model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        for t in ["<|mask|>", "[MASK]", "<mask>"]:
            if t in tokenizer.get_vocab():
                mask_token_id = tokenizer.get_vocab()[t]
                break

    embedding_matrix = model.model.embed_tokens.weight.detach()

    # --- Dream Native Parameters ---
    BATCH_SIZE = 16
    STEPS = 32
    MAX_NEW_TOKENS = 64
    TEMPERATURE = 1.0
    # ALG_TEMP = 0.6
    BASE_ALPHA = 16.0

    feature_extractor = FeatureExtractor(
        embedding_matrix=embedding_matrix, kernel_target="logits",
        pooling_method="max", top_k=0, use_confidence_weighting=True,
        ignore_token_ids=[]  # FIXED: Empty list prevents in-place Softmax mutation
    )

    odd_strategy = get_strategy("odd", BASE_ALPHA, 1.0, feature_extractor)

    prompt_text = "Write a python function to compute the fibonacci sequence."
    messages = [{"role": "user", "content": prompt_text}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)

    input_ids = inputs.input_ids.repeat(BATCH_SIZE, 1).to(model.device)
    attention_mask = inputs.attention_mask.repeat(BATCH_SIZE, 1).to(model.device)
    prompt_len = input_ids.shape[1]

    def odd_logits_hook(step, x, logits):
        # FIXED: Temporarily restore graph building inside the hook
        with torch.enable_grad():
            gen_x = x[:, prompt_len:]
            gen_logits = logits[:, prompt_len:, :].clone()
            gen_mask = (gen_x == mask_token_id)

            if not gen_mask.any():
                return logits

            curr_alpha = BASE_ALPHA * (1.0 - (step / STEPS))
            odd_strategy.alpha = curr_alpha

            if curr_alpha > 0.0:
                guided_gen_logits, _ = odd_strategy.apply(
                    logits=gen_logits, mask_index=gen_mask, x=gen_x,
                    history_vecs=[], history_qualities=[], protected_tokens=None
                )

                # Copy the safe, detached logits back into the main pipeline tensor
                logits[:, prompt_len:, :] = guided_gen_logits.detach()

            return logits

    print(f"\n>>> Executing Dream ODD Generation (Steps: {STEPS}, Temp: {TEMPERATURE})...")
    start_t = time.time()

    # Outer block keeps Dream's forward pass memory-efficient
    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS, steps=STEPS,
            temperature=TEMPERATURE,
            # alg_temp=ALG_TEMP,
            top_p=1.0,
            alg="origin",
            return_dict_in_generate=True,
            # generation_logits_hook_func=odd_logits_hook
        )

    print(f">>> Generation completed in {time.time() - start_t:.2f}s\n")

    generations = [tokenizer.decode(g[prompt_len:].tolist(), skip_special_tokens=True) for g in output.sequences]

    print("=" * 60)
    for i, gen in enumerate(generations):
        print(f"[Sample {i + 1}]\n{gen.strip()}\n" + "=" * 60)


if __name__ == "__main__":
    main()
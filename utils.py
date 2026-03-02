import torch
from sentence_transformers import util
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


def calculate_diversity_score(eval_model, texts):
    if len(texts) < 2: return 0.0
    embeddings = eval_model.encode(texts, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings)
    mask = torch.eye(len(texts), dtype=torch.bool).to(cos_scores.device)
    cos_scores.masked_fill_(mask, 0.0)
    avg_sim = cos_scores.sum() / (len(texts) * (len(texts) - 1))
    return 1.0 - avg_sim.item()

def load_model(cfg):
    print(f"Loading {cfg.model.name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        cfg.model.name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.eval()
    tokenizer.padding_side = 'left'

    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        embedding_matrix = model.model.transformer.wte.weight
    else:
        embedding_matrix = None

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = cfg.model.mask_token_id

    return model, tokenizer, embedding_matrix, mask_token_id

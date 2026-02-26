import torch
from sentence_transformers import util

def calculate_diversity_score(eval_model, texts):
    if len(texts) < 2: return 0.0
    embeddings = eval_model.encode(texts, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings)
    mask = torch.eye(len(texts), dtype=torch.bool).to(cos_scores.device)
    cos_scores.masked_fill_(mask, 0.0)
    avg_sim = cos_scores.sum() / (len(texts) * (len(texts) - 1))
    return 1.0 - avg_sim.item()
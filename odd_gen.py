import hydra
from omegaconf import DictConfig

from feature_extractor import FeatureExtractor
from generator import DiverseGenerator
from strategies import get_strategy
from utils import load_model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    model, tokenizer, embedding_matrix, mask_token_id = load_model(cfg)

    feature_extractor = FeatureExtractor(
        embedding_matrix=embedding_matrix,
        kernel_target=cfg.strategy.target,
        pooling_method=cfg.strategy.pool,
        top_k=cfg.strategy.top_k
    )
    
    dpp_strategy = get_strategy(
        cfg.strategy.name, 
        cfg.strategy.alpha, 
        cfg.strategy.quality_scale, 
        feature_extractor
    )
    
    generator = DiverseGenerator(model, tokenizer, dpp_strategy, mask_token_id)
    
    print(f"Generating for prompt: {cfg.prompt}")
    history, samples = generator.generate(
        prompt=cfg.prompt, 
        batch_size=cfg.batch_size, 
        steps=cfg.steps, 
        gen_length=cfg.gen_length, 
        temperature=cfg.temperature,
    )

    for i, s in enumerate(samples):
        print(f"[{i+1}] {s}")


if __name__ == "__main__":
    main()

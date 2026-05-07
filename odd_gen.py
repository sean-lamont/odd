import hydra
from omegaconf import DictConfig

from feature_extractor import FeatureExtractor
from generator import DiverseGenerator
from utils import load_model
from strategies import get_strategy


# import functools
# import torch
# from transformers.configuration_utils import PretrainedConfig
# from transformers.modeling_utils import PreTrainedModel
# if not hasattr(PretrainedConfig, "use_cache"):
#     PretrainedConfig.use_cache = False
#
# _original_getattr = getattr(PreTrainedModel, "__getattr__", torch.nn.Module.__getattr__)
#
#
# def _patched_getattr(self, name):
#     if name == "all_tied_weights_keys": return {}
#     return _original_getattr(self, name)
#
#
# PreTrainedModel.__getattr__ = _patched_getattr
#
# if hasattr(PreTrainedModel, "_finalize_model_loading"):
#     _original_finalize = PreTrainedModel._finalize_model_loading
#
# # _original_finalize = PreTrainedModel._finalize_model_loading
#
#
# def _patched_finalize(self, *args, **kwargs):
#     if hasattr(self, "tie_weights"):
#         original_tie_weights = self.tie_weights
#
#         @functools.wraps(original_tie_weights)
#         def safe_tie_weights(*tw_args, **tw_kwargs):
#             tw_kwargs.pop("tied_weight_pointers", None)
#             tw_kwargs.pop("missing_keys", None)
#             tw_kwargs.pop("recompute_mapping", None)
#             return original_tie_weights(*tw_args, **tw_kwargs)
#
#         self.tie_weights = safe_tie_weights
#     return _original_finalize(self, *args, **kwargs)
#
#
# PreTrainedModel._finalize_model_loading = _patched_finalize
#
#


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
        print(f"[{i + 1}] {s}")


if __name__ == "__main__":
    main()

<h1 align="center">ODD: Orthogonal Diverse Diffusion</h1>

<p align="center">
  <strong>Free Lunch for Pass@k? Low Cost Diverse Sampling for Diffusion Language Models</strong>
</p>

<div align="center">

[![arXiv Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/TODO)
[![Project Website](https://img.shields.io/badge/Website-Project_Page-1f425f.svg?style=flat-square)](https://sean-lamont.github.io/odd/)
[![Live Demo](https://img.shields.io/badge/Demo-Hugging_Face-blue.svg?style=flat-square)](https://huggingface.co/spaces/sean-lamont/ODD-Demo)
[![Weights & Biases GSM8K](https://img.shields.io/badge/W&B-GSM8K-FFBE00.svg?style=flat-square&logo=weightsandbiases&logoColor=white)](https://wandb.ai/sean-a-lamont/odd_gsm8k)
[![Weights & Biases HumanEval](https://img.shields.io/badge/W&B-HumanEval-FFBE00.svg?style=flat-square&logo=weightsandbiases&logoColor=white)](https://wandb.ai/sean-a-lamont/odd_humaneval)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://github.com/sean-lamont/odd/blob/main/LICENCE)

</div>

<p align="center">
  <img src="docs/assets/demo.gif" alt="ODD Interactive Visualisation App" width="90%">
</p>
<p align="center">
  <em>Our interactive dashboard visualising ODD altering generation in real-time. It highlights counterfactuals, showing exactly what standard sampling would have unmasked (dashed) and where ODD forced a unique path (blue).</em>
</p>

---

## Overview
<p align="center">
  <img src="docs/assets/fig1.png" alt="Overview Diagram" width="80%">
</p>

This repository contains the official implementation of **ODD (Orthogonal Diverse Diffusion)**, a training-free inference strategy designed to enhance the diversity and sample efficiency of Diffusion Language Models (such as LLaDA).

By applying a lightweight, geometric repulsion term during the denoising process, ODD forces the model to explore distinct reasoning paths within a single batch, significantly improving **Pass@k** performance on reasoning and coding benchmarks like GSM8K and HumanEval with negligible computational overhead.


## Approach
<p align="center">
  <img src="docs/assets/fig2.png" alt="Approach Diagram" width="80%">
</p>

Unlike standard sampling, which treats every generation independently and often collapses into redundant modes, ODD exploits the intermediate states of the diffusion process. For each sample in a batch, it projects the latent features away from the subspace spanned by previous samples, enforcing structural diversity without requiring retraining or complex beam searches.


## Installation

Install the base conda and pip requirements:

```bash
conda env create -f environment.yml
conda activate odd
pip install -r requirements.txt
```

*Note: Install `flash_attn` and `triton` separately if supported by your system, with the versions we use commented out in `requirements.txt`.*

## Usage

Run `python odd_gen.py` to run a diversity augmented generation. The prompt and diversity settings can be configured in the config file `conf/config.yaml`.

## Interactive Visualisation (App)

To understand exactly how diversity interventions alter the model's generation trajectory, we provide an interactive visualisation tool.

### Local Generation
Run `python app.py` to launch the local Streamlit interface. This version allows you to specify custom prompts and generation settings (alpha, temperature, batch size, etc.)

**How to use:**
```bash
# To run local inference visualization
streamlit run app.py
```

## Repository Structure

The codebase is structured as follows:

### Core Logic
* **`feature_extractor.py`**: Contains the `FeatureExtractor`, which extracts features from model logits during diffusion. Baseline is max-pool over logits, however alternative feature extraction methods could improve performance.
* **`strategies.py`**: Contains the diversity strategy implementations:
    * `ODDStrategy`: The main **ODD** algorithm. Sequentially projects samples away from the history of the batch.
    * `DPPStrategy`: The **DiverseFlow** baseline (DPP-based global optimisation).
    * `BaselineStrategy`: Standard independent sampling.
* **`generator.py`**: Contains `DiverseGenerator`, which manages the iterative diffusion loop and applies the selected strategy at each timestep.
* **`app_generator.py`**: Contains `AppGenerator`, a specialised generator used exclusively by the Streamlit app to track counterfactuals and logging metrics.
* **`odd_gen.py`**: The primary entry point for single run text generation. It loads the model, configures the strategy via Hydra, and produces outputs for a given prompt.
* **`utils.py`**: Utility functions.

### Benchmarking & Evaluation
Run these scripts to replicate the experiments in the paper. They handle dataset loading, answer extraction, and Pass@k calculation, and log to Weights and Biases (WandB). Optuna is used to control and synchronize the sweeps in multi-node and multi-process setups, currently using a grid sweep for the paper results. This can easily be changed to e.g. TPESampler to find the best hyperparameters for a given setup more quickly.

* **`sweep_gsm8k.py`**: Experiments for the 200 problem subset we test on in GSM8K, extracts answers by the final numeric value in the output string.
* **`sweep_human_eval.py`**: Evaluation over the HumanEval coding benchmark. It interfaces with the local `human_eval` directory to execute and validate generated code samples.

### Visualisation & Analysis
* **`app.py`**: Interactive Streamlit application for local, real-time generation visualization.
* **`streamlit_app.py`**: Lightweight, zero-GPU Streamlit application for exploring pre-computed benchmark results.
* **`gen_demo_data.py`**: Generates examples for the lightweight `streamlit_app.py` to run.
* **`analyse_results/`**: Contains scripts to download WandB run data and generate the tables/plots found in the paper, as well as profiling the overhead.
* **`conf/`**: Stores the Hydra configuration files.
* **`human_eval/`**: A fork of the official HumanEval evaluation harness, used by `sweep_human_eval.py` to run code execution tests.

## Citation

If you find this code or our approach useful in your research, please consider citing:

```bibtex
@article{lamont2026odd,
  title={Free Lunch for Pass@k? Low Cost Diverse Sampling for Diffusion Language Models},
  author={Lamont, Sean and Walder, Christian and Montague, Paul and Dezfouli, Amir and Norrish, Michael},
  journal={arXiv preprint},
  year={2026}
}
```
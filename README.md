<h1 align="center">ODD: Orthogonal Diverse Diffusion</h1>

<p align="center">
  <strong>Free Lunch for Pass@k? Low Cost Diverse Sampling for Diffusion Language Models</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/TODO_YOUR_ARXIV_ID">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?style=flat-square" alt="arXiv Paper">
  </a>
  <a href="https://TODO_YOUR_GITHUB_PAGES_URL">
    <img src="https://img.shields.io/badge/Website-Project_Page-1f425f.svg?style=flat-square" alt="Project Website">
  </a>
  <a href="https://github.com/sean-lamont/odd/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License">
  </a>
  <img src="https://img.shields.io/badge/Python-3.9+-yellow.svg?style=flat-square" alt="Python Version">
</p>

<p align="center">
  <img src="docs/assets/demo.gif" alt="ODD Interactive Visualization App" width="90%">
</p>
<p align="center">
  <em>Our interactive dashboard visualising ODD altering generation in real-time. It highlights counterfactuals—showing exactly what standard sampling would have unmasked (dashed) and where ODD forced a unique reasoning path.</em>
</p>

---

## 🚀 Overview

This repository contains the official implementation of **ODD (Orthogonal Diverse Diffusion)**, a training-free inference strategy designed to enhance the diversity and sample efficiency of Diffusion Language Models (such as LLaDA). 

By applying a lightweight, geometric repulsion term during the denoising process, ODD forces the model to explore distinct reasoning paths within a single batch, significantly improving **Pass@k** performance on reasoning and coding benchmarks like GSM8K and HumanEval with negligible computational overhead.

<p align="center">
  <img src="docs/assets/fig1.png" alt="Overview Diagram" width="80%">
</p>

## 🧠 Approach

Unlike standard sampling, which treats every generation independently and often collapses into redundant modes, ODD exploits the intermediate states of the diffusion process. For each sample in a batch, it projects the latent features away from the subspace spanned by previous samples, enforcing structural diversity without requiring retraining or complex beam searches.

<p align="center">
  <img src="docs/assets/fig2.png" alt="Approach Diagram" width="80%">
</p>

## ⚙️ Installation 

Install the base conda and pip requirements: 

```bash 
conda env create -f environment.yml
conda activate odd
pip install -r requirements.txt
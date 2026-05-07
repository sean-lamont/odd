conda create -n dream_eval python=3.10 -y
conda activate dream_eval
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.2 accelerate sentence-transformers hydra-core wandb omegaconf datasets optuna scikit-learn
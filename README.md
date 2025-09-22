# VLA_MIPT
MIPT Internship Test Assignment

## Install

### First we need to install the OpenVLA, Maniskill, Training Pipeline:

``` bash
cd RL4VLA/

# create conda env: rlvla_env
conda create -n rlvla_env -y python=3.10
conda activate rlvla_env

# install dependencies
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
cd openvla && pip install -e . && cd ..
pip install -U tyro
pip install datasets==3.3.2

# special install for flash attention
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# install other dependencies
cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..

# optional: for ubuntu 2204
# sudo apt-get install libglvnd-dev

```

#### Optional: Create octo-env for collect data with octo-small

```bash
conda create -n octo_env -y python=3.10
conda activate octo_env

git clone https://github.com/octo-models/octo.git

cd ManiSkill && pip install -e . && cd ..

cd octo && pip install -e . && pip install -r requirements.txt && cd ..
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 "nvidia-cudnn-cu11>=8.7,<9.0" --index-url https://download.pytorch.org/whl/cu118
pip install -U tyro
pip install scipy==1.12.0

cd SimplerEnv && pip install -e . && cd ..
```

```bash

# 0) New env (Python 3.10) vlm_env
conda create -n vlm_env -y python=3.10
conda activate vlm_env

# 1) Core PyTorch (CUDA 12.1 wheels)
pip install --upgrade pip
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121


pip install \
  "transformers>=4.44.2" \
  accelerate>=0.33.0 \
  safetensors>=0.4.2 \
  bitsandbytes==0.43.1 \
  datasets==3.3.2 \
  peft==0.11.1 \
  numpy==1.26 \
  pillow>=10.2 \
  tqdm>=4.66 \
  matplotlib>=3.8 \
  scikit-learn>=1.4 \
  wandb>=0.17 \
  trl==0.23.0 

# Optional: tokenizers & timm sometimes help with VLMs
pip install tokenizers>=0.19 timm>=0.9.16

# 3) Optional: FlashAttention (prebuilt wheel for Linux, CUDA 12, torch 2.2, cp310)
#    If you're on Ubuntu with a supported GPU, this is the easiest drop-in.
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 5) (Ubuntu 22.04 OpenGL dev libs, if needed for some envs/viewers)
# sudo apt-get update && sudo apt-get install -y libglvnd-dev


# Test 
python - << 'PY'
import torch
from PIL import Image
from transformers.image_utils import load_image
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration, AutoModelForVision2Seq, TrainingArguments, Trainer
import json, os, pickle, numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import wandb
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import gc

print("All imports OK.")
print("torch:", torch.__version__)
PY

```

#### When you 
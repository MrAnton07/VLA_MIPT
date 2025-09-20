# VLA_MIPT
MIPT Internship Test Assignment

## Install

```

# 0) New env (Python 3.10), as requested: vlm_env
conda create -n vlm_env -y python=3.10
conda activate vlm_env

# 1) Core PyTorch (CUDA 12.1 wheels)
pip install --upgrade pip
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121

# 2) Core ML stack for your imports
# - transformers (recent enough for Idefics3 & vision2seq)
# - accelerate & safetensors for HF runtime
# - bitsandbytes for 4/8-bit (matches torch 2.2 well around 0.43.x)
# - datasets (you previously pinned 3.3.2)
# - peft for LoRA
# - numpy, pillow, tqdm, matplotlib, scikit-learn, wandb
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
  wandb>=0.17

# Optional: tokenizers & timm sometimes help with VLMs
pip install tokenizers>=0.19 timm>=0.9.16

# 3) Optional: FlashAttention (prebuilt wheel for Linux, CUDA 12, torch 2.2, cp310)
#    If you're on Ubuntu with a supported GPU, this is the easiest drop-in.
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 4) (If you also need ManiSkill / SimplerEnv in this env)
# cd ManiSkill && pip install -e . && cd ..
# cd SimplerEnv && pip install -e . && cd ..

# 5) (Ubuntu 22.04 OpenGL dev libs, if needed for some envs/viewers)
# sudo apt-get update && sudo apt-get install -y libglvnd-dev

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

После установки всех библиотек нам надо собрать датасет putonplate: запускаем скрипт а потом собираем json carrot_on_plate_files etc
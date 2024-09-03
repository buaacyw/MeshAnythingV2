#!/bin/bash

cd ~/

# Setup comfyui
git clone https://github.com/comfyanonymous/ComfyUI.git
pip install -r ./ComfyUI/requirements.txt

# Setup comfyui manager
git clone https://github.com/ltdrdata/ComfyUI-Manager.git ./ComfyUI/custom_nodes/ComfyUI-Manager
pip install -r ./ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt

# Download checkpoints
wget -O "sd3_medium_incl_clips_t5xxlfp8.safetensors" "https://huggingface.co/adamo1139/stable-diffusion-3-medium-ungated/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors?download=true"

# custom nodes
git clone https://github.com/SpiffGreen/comfyui_meshanything_v2.git ~/ComfyUI/custom_nodes/comfyui_meshanything_v2
pip install -r ~/ComfyUI/custom_nodes/comfyui_meshanything_v2
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn --no-build-isolation
pip install -U gradio

# Setup ngrok
sudo apt-get install ngrok
ngrok config add-authtoken 2l340923829382

# Run app
nohup python3 ~/ComfyUI/main.py & | ngrok http 8188 --url open-flamingo-lightly.ngrok-free.app

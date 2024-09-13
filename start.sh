apt-get update
# apt-get install -y wget curl git python3 python3-pip python3-dev build-essential virtualenv
sudo apt-get install -y virtualenv

virtualenv -p python3 venv
source ./venv/bin/activate
# pip install colabcode

# Setup comfyui
cd ~/
git clone https://github.com/comfyanonymous/ComfyUI.git
pip install -r ~/ComfyUI/requirements.txt

# Setup comfyui-manager
cd ~/ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# Setup custom node
git clone https://github.com/SpiffGreen/comfyui_meshanything_v2.git
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r ~/ComfyUI/custom_nodes/comfyui_meshanything_v2/requirements.txt
pip install flash-attn --no-build-isolation
pip install -U gradio

cd ~/ComfyUI
python3 main.py

# proxy server
# ssh -p 443 -R0:localhost:8188 a.pinggy.io
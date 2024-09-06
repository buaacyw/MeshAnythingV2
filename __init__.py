import importlib.util
import os
import sys
import folder_paths
import importlib

ROOT_PATH = os.path.join(folder_paths.base_path, "custom_nodes", "comfyui_meshanything_v2")
MODULE_PATH = os.path.join(ROOT_PATH, "MeshAnything")

sys.path.append(ROOT_PATH)
sys.path.append(MODULE_PATH)

from .nodes import NODE_CLASS_MAPPINGS
from . import MeshAnything

print('--------------')
print('*ComfyUI_MeshAnythingV2- nodes_loaded*')
print('--------------')

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__ALL__ = ['NODE_CLASS_MAPPINGS']
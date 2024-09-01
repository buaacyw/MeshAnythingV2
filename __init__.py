import os
import sys
import folder_paths

ROOT_PATH = os.path.join(folder_paths.get_folder_paths("custom_nodes")[0], "comfyui_meshanything_v2")
MODULE_PATH = os.path.join(ROOT_PATH, "MeshAnything")

sys.path.append(ROOT_PATH)
sys.path.append(MODULE_PATH)

from .nodes import NODE_CLASS_MAPPINGS

print('--------------')
print('*ComfyUI_MeshAnythingV2- nodes_loaded*')
print('--------------')

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__ALL__ = ['NODE_CLASS_MAPPINGS']
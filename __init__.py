# import importlib

# nodes_module = importlib.import_module("nodes", package=__name__)
# NODE_CLASS_MAPPINGS = getattr(nodes_module, "NODE_CLASS_MAPPINGS")
from .nodes import SaveMesh, GrayScale

NODE_CLASS_MAPPINGS = {
    "CMA_SaveMesh": SaveMesh,
    "CMA_GrayScale": GrayScale,
}

print('--------------')
print('*ComfyUI_MeshAnythingV2- nodes_loaded*')
print('--------------')

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__ALL__ = ['NODE_CLASS_MAPPINGS']
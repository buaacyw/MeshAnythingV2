import importlib

nodes_module = importlib.import_module(".MeshAnything.models.meshanything_v2", package=__name__)
NODE_CLASS_MAPPINGS = getattr(nodes_module, "NODE_CLASS_MAPPINGS")

print('--------------')
print('*ComfyUI_MeshAnythingV2- nodes_loaded*')
print('--------------')

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__ALL__ = ['NODE_CLASS_MAPPINGS']
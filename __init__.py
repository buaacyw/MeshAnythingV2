import importlib

nodes_module = importlib.import_module(".nodes", package=__name__)
SaveMesh = getattr(nodes_module, "SaveMesh")
GrayScale = getattr(nodes_module, "GrayScale")

NODE_CLASS_MAPPINGS = {
    # "CMA_MeshImage": MeshImage,
    "CMA_SaveMesh": SaveMesh,
    "CMA_GrayScale": GrayScale,
}

print('--------------')
print('*ComfyUI_MeshAnythingV2- nodes_loaded*')
print('--------------')

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__ALL__ = ['NODE_CLASS_MAPPINGS']
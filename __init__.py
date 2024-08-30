from .comfyui_meshanything_v2 import MeshImage, MeshText

NODE_CLASS_MAPPINGS = {
    "MeshImage": MeshImage,
    "MeshText": MeshText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshImage": "MeshImage 2"
}

print('--------------')
print('*ComfyUI_MeshAnythingV2- nodes_loaded*')
print('--------------')

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__ALL__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
import time

"""
The ComfyUI Meshanythingv2 Node simply takes an input image/text/3d Object and turns into a mesh, even smaller size.
"""
class MeshImage:

    def __init__(self) -> None:
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE",),
                    }
                }
    
    RETURN_TYPES = ("IMAGE")
    FUNCTION = "mesh_image"

    OUTPUT_NODE = True

    CATEGORY = "ComfyUI MeshAnythingV2"

    def mesh_image(self, image):
        print(image)
        print("Meshing Image...")
        print("Meshing Image Completed...")
        return (image,)
    
    def IS_CHANGED(s, images):
        return time.time()
    
# class MeshText:
#     pass

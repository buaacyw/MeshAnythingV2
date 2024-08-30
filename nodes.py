from PIL import Image

"""
The ComfyUI Meshanythingv2 Node simply takes an input image/text/3d Object and turns into a mesh, even smaller size.
"""
class MeshImage:
  @classmethod
  def INPUT_TYPES(s):
    return {
			"required": {
				"image": ("IMAGE",),
			}
		}
    
  RETURN_TYPES = ("IMAGE",)
  FUNCTION = "mesh_image"

  CATEGORY = "image/postprocessing"
  
  def mesh_image(self, image):
    converted_image = Image.fromarray(image)
    converted_image = converted_image.convert("L")
    return converted_image
    
NODE_CLASS_MAPPINGS = {
    "CMA_MeshImage": MeshImage,
}
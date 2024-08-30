import torch
from PIL import Image, ImageOps
import numpy as np

"""
The ComfyUI Meshanythingv2 Node simply takes an input image/text/3d Object and turns into a mesh, even smaller size.
"""
class MeshImage:
  @classmethod
  def INPUT_TYPES(s):
    return {
			"required": {
				"images": ("IMAGE",),
			}
		}
    
  RETURN_TYPES = ("IMAGE",)
  FUNCTION = "mesh_images"

  CATEGORY = "image/postprocessing"
  
  def mesh_images(self, images):
    converted_images = []
    for image in images:
      i = 255. * image.cpu().numpy()
      img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("L")
      converted_images.append(img)
      
    images = []
    for pil_image in converted_images:
        i = pil_image
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        images.append(image)
    
    if len(images) > 1:
        images = torch.cat(images, dim=0)
    else:
        images = images[0]

    return (images,)
    
NODE_CLASS_MAPPINGS = {
    "CMA_MeshImage": MeshImage,
}
from pathlib import Path
import torch
import datetime, time
import os
import folder_paths
import trimesh
from PIL import Image, ImageOps
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs
from MeshAnything.models.meshanything_v2 import MeshAnythingV2
from .utils import Dataset, conv_pil_tensor, conv_tensor_pil


"""
The ComfyUI Meshanythingv2 Node simply takes an input image/text/3d Object and turns into a mesh, even smaller size.
"""
class GrayScale:
  @classmethod
  def INPUT_TYPES(s):
    return {
			"required": {
				"image": ("IMAGE",),
			}
		}
    
  RETURN_TYPES = ("IMAGE",)
  FUNCTION = "grayscale_image"

  CATEGORY = "image/postprocessing"
  
  def grayscale_image(self, image):
    image = conv_pil_tensor(conv_tensor_pil(image).convert("L"))
    return (image,)
   

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
    cur_time = datetime.datetime.now().strftime("%d_%H-%M-%S")
    checkpoint_dir = os.path.join(os.getcwd(), cur_time)
    os.makedirs(checkpoint_dir, exist_ok=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16",
        project_dir=checkpoint_dir,
        kwargs_handlers=[kwargs]
    )
    model = MeshAnythingV2.from_pretrained("Yiwen-ntu/meshanythingv2")
    set_seed(0)
    dataset = Dataset("pc_normal", [MeshImage._resolve_path(image=image)], False, 7)
    
    
    # Start ---------
    dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=1,
      drop_last = False,
      shuffle = False,
    )

    if accelerator.state.num_processes > 1:
      model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    dataloader, model = accelerator.prepare(dataloader, model)
    begin_time = time.time()
    print("Generation Start!!!")
    
    with accelerator.autocast():
      for curr_iter, batch_data_label in enumerate(dataloader):
        outputs = model(batch_data_label['pc_normal'], sampling=False)
        batch_size = outputs.shape[0]
        device = outputs.device

        for batch_id in range(batch_size):
          recon_mesh = outputs[batch_id]
          valid_mask = torch.all(~torch.isnan(recon_mesh.reshape((-1, 9))), dim=1)
          recon_mesh = recon_mesh[valid_mask]  # nvalid_face x 3 x 3

          vertices = recon_mesh.reshape(-1, 3).cpu()
          vertices_index = np.arange(len(vertices))  # 0, 1, ..., 3 x face
          triangles = vertices_index.reshape(-1, 3)

          scene_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, force="mesh",
                                        merge_primitives=True)
          scene_mesh.merge_vertices()
          scene_mesh.update_faces(scene_mesh.nondegenerate_faces())
          scene_mesh.update_faces(scene_mesh.unique_faces())
          scene_mesh.remove_unreferenced_vertices()
          scene_mesh.fix_normals()
          save_path = os.path.join(checkpoint_dir, f'{batch_data_label["uid"][batch_id]}_gen.obj')
          num_faces = len(scene_mesh.faces)
          brown_color = np.array([255, 165, 0, 255], dtype=np.uint8)
          face_colors = np.tile(brown_color, (num_faces, 1))

          scene_mesh.visual.face_colors = face_colors
          scene_mesh.export(save_path)
          print(f"{save_path} Over!!")
    end_time = time.time()
    print(f"Total time: {end_time - begin_time}")
    
    return (image,)
  
  def _resolve_path(image) -> Path:
    image_path = Path(folder_paths.get_annotated_filepath(image))
    return image_path
    
NODE_CLASS_MAPPINGS = {
    "CMA_MeshImage": MeshImage,
    "CMA_GrayScale": GrayScale,
}
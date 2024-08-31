import trimesh
from mesh_to_pc import process_mesh_to_pc
import torch
import numpy as np
from PIL import Image

# Utils

def conv_pil_tensor(img):
	return (torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0),)

def conv_tensor_pil(tsr):
	return Image.fromarray(np.clip(255. * tsr.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class Dataset:
  def __init__(self, input_type, input_list, mc=False, mc_level = 7):
    super().__init__()
    self.data = []
    if input_type == 'pc_normal':
      for input_path in input_list:
        # load npy
        cur_data = np.load(input_path)
        # sample 4096
        assert cur_data.shape[0] >= 8192, "input pc_normal should have at least 4096 points"
        idx = np.random.choice(cur_data.shape[0], 8192, replace=False)
        cur_data = cur_data[idx]
        self.data.append({'pc_normal': cur_data, 'uid': input_path.split('/')[-1].split('.')[0]})

    elif input_type == 'mesh':
      mesh_list = []
      for input_path in input_list:
        # load ply
        cur_data = trimesh.load(input_path)
        mesh_list.append(cur_data)
      if mc:
        print("First Marching Cubes and then sample point cloud, need several minutes...")
      pc_list, _ = process_mesh_to_pc(mesh_list, marching_cubes=mc, mc_level=mc_level)
      for input_path, cur_data in zip(input_list, pc_list):
        self.data.append({'pc_normal': cur_data, 'uid': input_path.split('/')[-1].split('.')[0]})
    print(f"dataset total data samples: {len(self.data)}")
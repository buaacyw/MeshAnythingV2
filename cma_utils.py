import trimesh
# from mesh_to_pc import process_mesh_to_pc
import time
import torch
import numpy as np
from PIL import Image, ImageOps
from numpy import asarray
import os
from os.path import isfile, join, exists, dirname
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from datetime import datetime
import mesh2sdf.core
import numpy as np
import skimage.measure
import trimesh
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
# Utils

C0 = 0.28209479177387814
def SH2RGB(sh):
    return sh * C0 + 0.5

# Convert torch images to pillow image
def torch_imgs_to_pils(images):
    converted_images = []
    for image in images:
      i = 255. * image.cpu().numpy()
      img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
      converted_images.append(img)
    return converted_images

def prepare_torch_img(img, size_H, size_W, device="cuda", keep_shape=False):
    # [N, H, W, C] -> [N, C, H, W]
    img_new = img.permute(0, 3, 1, 2).to(device)
    img_new = F.interpolate(img_new, (size_H, size_W), mode="bilinear", align_corners=False).contiguous()
    if keep_shape:
        img_new = img_new.permute(0, 2, 3, 1)
    return img_new

def pils_to_torch_imgs(pil_images):
    images = []
    for pil_image in pil_images:
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
    return images

def parse_save_filename(save_path, output_directory, supported_extensions, class_name):

    folder_path, filename = os.path.split(save_path)
    filename, file_extension = os.path.splitext(filename)
    if file_extension.lower() in supported_extensions:
        if not os.path.isabs(save_path):
            folder_path = join(output_directory, folder_path)

        os.makedirs(folder_path, exist_ok=True)

        # replace time date format to current time
        now = datetime.now()  # current date and time
        all_date_format = ["%Y", "%m", "%d", "%H", "%M", "%S", "%f"]
        for date_format in all_date_format:
            if date_format in filename:
                filename = filename.replace(date_format, now.strftime(date_format))

        save_path = join(folder_path, filename) + file_extension
        print(f"[{class_name}] Saving model to {save_path}")
        return save_path
    else:
        print(
            f"[{class_name}] File name {filename} does not end with supported file extensions: {supported_extensions}"
        )

    return None

def normalize_vertices(vertices, scale=0.95):
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    return vertices, center, scale

def export_to_watertight(normalized_mesh, octree_depth: int = 7):
    """
        Convert the non-watertight mesh to watertight.

        Args:
            input_path (str): normlized path
            octree_depth (int):

        Returns:
            mesh(trimesh.Trimesh): watertight mesh

        """
    size = 2 ** octree_depth
    level = 2 / size
    scaled_vertices, to_orig_center, to_orig_scale = normalize_vertices(normalized_mesh.vertices)

    sdf = mesh2sdf.core.compute(scaled_vertices, normalized_mesh.faces, size=size)

    vertices, faces, normals, _ = skimage.measure.marching_cubes(np.abs(sdf), level)

    vertices = vertices / size * 2 - 1 # -1 to 1
    vertices = vertices / to_orig_scale + to_orig_center
    # vertices = vertices / to_orig_scale + to_orig_center
    mesh = trimesh.Trimesh(vertices, faces, normals=normals)

    return mesh

def process_mesh_to_pc(mesh_list, marching_cubes = False, sample_num = 8192, mc_level= 7):
    # mesh_list : list of trimesh
    pc_normal_list = []
    return_mesh_list = []
    for mesh in mesh_list:
        if marching_cubes:
            cur_time = time.time()
            mesh = export_to_watertight(mesh, octree_depth=mc_level)
            print("MC over! ", "mc_level: ", mc_level, "process_time:" , time.time() - cur_time)
        return_mesh_list.append(mesh)
        points, face_idx = mesh.sample(sample_num, return_index=True)
        normals = mesh.face_normals[face_idx]

        pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)
        pc_normal_list.append(pc_normal)
        print("process mesh success")
    return pc_normal_list, return_mesh_list

class Dataset:
    def __init__(self, input_type, input_list, mc=False, mc_level=7):
        super().__init__()
        self.data = []
        if input_type == "pc_normal":
            for input_path in input_list:
                # load npy
                # input_data = Image.open(input_path)
                # tdata = asarray(input_data)
                # cur_data = np.load(tdata)
                cur_data = np.load(input_path)
                # sample 4096
                assert (
                    cur_data.shape[0] >= 8192
                ), "input pc_normal should have at least 4096 points"
                idx = np.random.choice(cur_data.shape[0], 8192, replace=False)
                cur_data = cur_data[idx]
                self.data.append(
                    {
                        "pc_normal": cur_data,
                        "uid": input_path.split("/")[-1].split(".")[0],
                    }
                )

        elif input_type == "mesh":
            mesh_list = []
            for input_path in input_list:
                # load ply
                cur_data = trimesh.load(input_path)
                mesh_list.append(cur_data)
            if mc:
                print(
                    "First Marching Cubes and then sample point cloud, need several minutes..."
                )
            pc_list, _ = process_mesh_to_pc(
                mesh_list, marching_cubes=mc, mc_level=mc_level
            )
            for input_path, cur_data in zip(input_list, pc_list):
                self.data.append(
                    {
                        "pc_normal": cur_data,
                        "uid": input_path.split("/")[-1].split(".")[0],
                    }
                )
        print(f"dataset total data samples: {len(self.data)}")

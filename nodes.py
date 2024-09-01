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
from utils import Dataset, pils_to_torch_imgs, torch_imgs_to_pils, parse_save_filename
from mesh import Mesh

"""
The ComfyUI Meshanythingv2 Node simply takes an input image/text/3d Object and turns into a mesh, even smaller size.
"""

SUPPORTED_3D_EXTENSIONS = (
    ".obj",
    ".ply",
    ".glb",
)


class GrayScale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("greyed_image",)
    FUNCTION = "grayscale_image"

    CATEGORY = "CMA_V2"

    def grayscale_image(self, image):
        images = []
        for img in torch_imgs_to_pils(image):
            images.append(img.convert("L"))
        images = pils_to_torch_imgs(images)
        return (images,)


class SaveMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "save_path": (
                    "STRING",
                    {"default": "Mesh_%Y-%m-%d-%M-%S-%f.glb", "multiline": False},
                ),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_path",)
    FUNCTION = "save_mesh"
    CATEGORY = "CMA_V2"

    def save_mesh(self, mesh, save_path):
        save_path = parse_save_filename(
            save_path,
            folder_paths.output_directory,
            SUPPORTED_3D_EXTENSIONS,
            self.__class__.__name__,
        )

        if save_path is not None:
            mesh.write(save_path)

        return (save_path,)


class MeshImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "load_mesh"
    OUTPUT_NODE = True

    CATEGORY = "CMA_V2"

    def mesh_image(self, mesh):
        print(mesh)
        checkpoint_dir = os.path.join(folder_paths.output_directory, "meshanythingv2")
        os.makedirs(checkpoint_dir, exist_ok=True)
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            mixed_precision="fp16",
            project_dir=checkpoint_dir,
            kwargs_handlers=[kwargs]
        )

        model = MeshAnythingV2.from_pretrained("Yiwen-ntu/meshanythingv2")


class LoadMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mesh_path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("MESH",)

    FUNCTION = "load_mesh"

    CATEGORY = "CMA_V2"

    OUTPUT_NODE = True

    @classmethod
    def load_mesh(cls, mesh_path: str):
        if os.path.exists(mesh_path):
            folder, filename = os.path.split(mesh_path)
            if filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
                with torch.inference_mode(True):
                    mesh = Mesh.load(mesh_path)
            else:
                print(f"[LoadMesh] File name {filename} does not end with supported 3D file extensions: {SUPPORTED_3D_EXTENSIONS}")
        else:        
            print(f"[LoadMesh] File {mesh_path} does not exist")
        return (mesh, )


class LoadInputType:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_type": (
                    [
                        "pc_normal",
                        "mesh",
                    ],
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("input_type",)

    FUNCTION = "select_input"

    CATEGORY = "CMA_V2"

    OUTPUT_NODE = True

    @classmethod
    def select_input(cls, input_type):
        return (input_type,)


class ImageTo3DMeshNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)

    FUNCTION = "convert_image_to_mesh"
    CATEGORY = "CMA_V2"

    OUTPUT_NODE = True

    @classmethod
    def convert_image_to_mesh(cls, image):
        mesh = None
        image = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
        mesh = Mesh()
        mesh.v = torch.tensor(np.random.rand(100, 3), dtype=torch.float32)
        mesh.f = torch.tensor(np.random.randint(0, 100, (100, 3)), dtype=torch.int32)
        return (mesh,)


NODE_CLASS_MAPPINGS = {
    "CMA_MeshImage": MeshImage,
    "CMA_SaveMesh": SaveMesh,
    "CMA_GrayScale": GrayScale,
    "CMA_LoadMesh": LoadMesh,
    "CMA_LoadInputTYpe": LoadInputType,
    "CMA_ImageToMesh": ImageTo3DMeshNode,
}

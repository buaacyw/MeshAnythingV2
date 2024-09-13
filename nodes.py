from PIL import Image
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import datetime
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import trimesh
import time
import mesh2sdf.core
import skimage.measure
import os
from accelerate import Accelerator
import folder_paths
from os import listdir
from os.path import isfile, join, exists, dirname
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_model
from .MeshAnything.models.meshanything_v2 import MeshAnythingV2

SEED = 0
MC = False
MC_LEVEL = 7
SAMPLING = False
BATCHSIZE_PER_GPU = 1
WEIGHT_DTYPE = torch.float16
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

ROOT_PATH = os.path.join(
    folder_paths.get_folder_paths("custom_nodes")[0], "comfyui_meshanything_v2"
)
CKPT_ROOT_PATH = os.path.join(ROOT_PATH, "Checkpoints")
CKPT_DIFFUSERS_PATH = os.path.join(CKPT_ROOT_PATH, "Diffusers")
CONFIG_ROOT_PATH = os.path.join(ROOT_PATH, "Configs")

SUPPORTED_CHECKPOINTS_EXTENSIONS = (
    ".ckpt",
    ".bin",
    ".safetensors",
)

SUPPORTED_3D_EXTENSIONS = (
    ".obj",
    ".ply",
    ".glb",
)


class Dataset:
    def __init__(self, input_type, input_list, mc=False, mc_level=7):
        super().__init__()
        self.data = []
        if input_type == "pc_normal":
            for input_path in input_list:
                # load npy
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = {}
        data_dict["pc_normal"] = self.data[idx]["pc_normal"]
        # normalize pc coor
        pc_coor = data_dict["pc_normal"][:, :3]
        normals = data_dict["pc_normal"][:, 3:]
        bounds = np.array([pc_coor.min(axis=0), pc_coor.max(axis=0)])
        pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
        pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995
        assert (
            np.linalg.norm(normals, axis=-1) > 0.99
        ).all(), "normals should be unit vectors, something wrong"
        data_dict["pc_normal"] = np.concatenate(
            [pc_coor, normals], axis=-1, dtype=np.float16
        )
        data_dict["uid"] = self.data[idx]["uid"]

        return data_dict


def parse_save_filename(save_path, output_directory, supported_extensions, class_name):

    folder_path, filename = os.path.split(save_path)
    filename, file_extension = os.path.splitext(filename)
    if file_extension.lower() in supported_extensions:
        if not os.path.isabs(save_path):
            folder_path = join(output_directory, folder_path)

        os.makedirs(folder_path, exist_ok=True)

        # replace time date format to current time
        now = datetime.datetime.now()  # current date and time
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


def resume_or_download_model_from_hf(
    checkpoints_dir_abs, repo_id, model_name, class_name="", repo_type="model"
):

    ckpt_path = os.path.join(checkpoints_dir_abs, model_name)
    if not os.path.isfile(ckpt_path):
        print(
            f"[{class_name}] can't find checkpoint {ckpt_path}, will download it from repo {repo_id} instead"
        )

        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=repo_id,
            local_dir=checkpoints_dir_abs,
            filename=model_name,
            repo_type=repo_type,
        )

    return ckpt_path


def get_list_filenames(directory, extension_filter=None, recursive=False):
    """
    Recursively finds files with specified extensions in a directory and returns relative paths.

    Args:
        directory (str): The directory path to search.
        extension_filter (list): List of file extensions (e.g., ['.txt', '.csv']).

    Returns:
        list: List of relative file paths matching the specified extensions.
    """
    if exists(directory):
        if recursive:
            result = []
            for root, _, files in os.walk(directory):
                for item in files:
                    if (
                        extension_filter is None
                        or os.path.splitext(item)[1].lower() in extension_filter
                    ):
                        relative_path = os.path.relpath(
                            os.path.join(root, item), directory
                        )
                        result.append(relative_path)
            return result
        else:
            return [
                f
                for f in listdir(directory)
                if isfile(join(directory, f))
                and (extension_filter is None or f.lower().endswith(extension_filter))
            ]
    else:
        return []


def torch_imgs_to_pils(images, masks=None, alpha_min=0.1):
    """
    images (torch): [N, H, W, C] or [H, W, C]
    masks (torch): [N, H, W] or [H, W]
    """
    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    if masks is not None:
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)

        inv_mask_index = masks < alpha_min
        images[inv_mask_index] = 0.0

        masks = masks.unsqueeze(3)
        images = torch.cat((images, masks), dim=3)
        mode = "RGBA"
    else:
        mode = "RGB"

    pil_image_list = [
        Image.fromarray(
            (images[i].detach().cpu().numpy() * 255).astype(np.uint8), mode=mode
        )
        for i in range(images.shape[0])
    ]

    return pil_image_list


def pils_resize_foreground(
    pils: Union[Image.Image, List[Image.Image]],
    ratio: float,
) -> List[Image.Image]:
    if isinstance(pils, Image.Image):
        pils = [pils]

    new_pils = []
    for image in pils:
        image = np.array(image)
        assert image.shape[-1] == 4
        alpha = np.where(image[..., 3] > 0)
        y1, y2, x1, x2 = (
            alpha[0].min(),
            alpha[0].max(),
            alpha[1].min(),
            alpha[1].max(),
        )
        # crop the foreground
        fg = image[y1:y2, x1:x2]
        # pad to square
        size = max(fg.shape[0], fg.shape[1])
        ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
        ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
        new_image = np.pad(
            fg,
            ((ph0, ph1), (pw0, pw1), (0, 0)),
            mode="constant",
            constant_values=((0, 0), (0, 0), (0, 0)),
        )

        # compute padding according to the ratio
        new_size = int(new_image.shape[0] / ratio)
        # pad to size, double side
        ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
        ph1, pw1 = new_size - size - ph0, new_size - size - pw0
        new_image = np.pad(
            new_image,
            ((ph0, ph1), (pw0, pw1), (0, 0)),
            mode="constant",
            constant_values=((0, 0), (0, 0), (0, 0)),
        )
        new_image = Image.fromarray(new_image, mode="RGBA")
        new_pils.append(new_image)

    return new_pils


def pils_to_torch_imgs(
    pils: Union[Image.Image, List[Image.Image]], device="cuda", force_rgb=True
):
    if isinstance(pils, Image.Image):
        pils = [pils]

    images = []
    for pil in pils:
        if pil.mode == "RGBA" and force_rgb:
            pil = pil.convert("RGB")

        images.append(TF.to_tensor(pil).permute(1, 2, 0))

    images = torch.stack(images, dim=0).to(device)

    return images


def export_to_watertight(normalized_mesh, octree_depth: int = 7):
    """
    Convert the non-watertight mesh to watertight.

    Args:
        input_path (str): normlized path
        octree_depth (int):

    Returns:
        mesh(trimesh.Trimesh): watertight mesh

    """
    size = 2**octree_depth
    level = 2 / size
    scaled_vertices, to_orig_center, to_orig_scale = normalize_vertices(
        normalized_mesh.vertices
    )

    sdf = mesh2sdf.core.compute(scaled_vertices, normalized_mesh.faces, size=size)

    vertices, faces, normals, _ = skimage.measure.marching_cubes(np.abs(sdf), level)

    vertices = vertices / size * 2 - 1  # -1 to 1
    vertices = vertices / to_orig_scale + to_orig_center
    # vertices = vertices / to_orig_scale + to_orig_center
    mesh = trimesh.Trimesh(vertices, faces, normals=normals)

    return mesh


def normalize_vertices(vertices, scale=0.95):
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    return vertices, center, scale


def process_mesh_to_pc(mesh_list, marching_cubes=False, sample_num=8192, mc_level=7):
    # mesh_list : list of trimesh
    pc_normal_list = []
    return_mesh_list = []
    for mesh in mesh_list:
        if marching_cubes:
            cur_time = time.time()
            mesh = export_to_watertight(mesh, octree_depth=mc_level)
            print(
                "MC over! ",
                "mc_level: ",
                mc_level,
                "process_time:",
                time.time() - cur_time,
            )
        return_mesh_list.append(mesh)
        points, face_idx = mesh.sample(sample_num, return_index=True)
        normals = mesh.face_normals[face_idx]

        pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)
        pc_normal_list.append(pc_normal)
        print("process mesh success")
    return pc_normal_list, return_mesh_list


def switch_vector_axis(vector3s, target_axis):
    """
    Example:
        vector3s = torch.tensor([[1, 2, 3], [3, 2, 1], [2, 3, 1]])  # shape (N, 3)

        target_axis = (2, 0, 1) # or [2, 0, 1]
        vector3s[:, [0, 1, 2]] = vector3s[:, target_axis]

        # Result: tensor([[3, 1, 2], [1, 3, 2], [1, 2, 3]])
    """
    vector3s[:, [0, 1, 2]] = vector3s[:, target_axis]
    return vector3s


def switch_mesh_axis_and_scale(mesh, target_axis, target_scale, flip_normal=False):
    """
    Args:
        target_axis (array): shape (3)
        target_scale (array): shape (3)
    """
    target_scale = torch.tensor(target_scale).float().cuda()
    mesh.v = switch_vector_axis(mesh.v * target_scale, target_axis)
    mesh.vn = switch_vector_axis(mesh.vn * target_scale, target_axis)
    if flip_normal:
        mesh.vn *= -1
    return mesh


def get_target_axis_and_scale(axis_string, scale_value=1.0):
    """
    Coordinate system inverts when:
    1. Any of the axis inverts
    2. Two of the axises switch

    If coordinate system inverts twice in a row then it will not be inverted
    """
    axis_names = ["x", "y", "z"]

    target_axis, target_scale, coordinate_invert_count = [], [], 0
    axis_switch_count = 0
    for i in range(len(axis_names)):
        s = axis_string[i]
        if s[0] == "-":
            target_scale.append(-scale_value)
            coordinate_invert_count += 1
        else:
            target_scale.append(scale_value)

        new_axis_i = axis_names.index(s[1])
        if new_axis_i != i:
            axis_switch_count += 1
        target_axis.append(new_axis_i)

    if axis_switch_count == 2:
        coordinate_invert_count += 1

    return target_axis, target_scale, coordinate_invert_count


class Resize_Image_Foreground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "foreground_ratio": (
                    "FLOAT",
                    {"default": 0.85, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "images",
        "masks",
    )

    FUNCTION = "resize_img_foreground"
    CATEGORY = "CMA_3D/Preprocessor"

    def resize_img_foreground(self, images, masks, foreground_ratio):
        image_pils = torch_imgs_to_pils(images, masks)
        image_pils = pils_resize_foreground(image_pils, foreground_ratio)

        images = pils_to_torch_imgs(image_pils, images.device, force_rgb=False)
        images, masks = images[:, :, :, 0:-1], images[:, :, :, -1]
        return (
            images,
            masks,
        )


class MeshAnything3D:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # "reference_image": ("IMAGE",),
                # "reference_mask": ("MASK",),
                "input_path": ("STRING", {"default": '', "multiline": False}),
                "input_type": (["pc_normal", "mesh"], {"default": "pc"}),
                "batchsize_per_gpu": (
                    "INT",
                    {"default": BATCHSIZE_PER_GPU, "min": 1, "max": 5},
                ),
                "seed": (
                    "INT",
                    {"default": SEED, "min": 0, "max": 5},
                ),
                "mc_level": (
                    "INT",
                    {"default": MC_LEVEL, "min": 0, "max": 20},
                ),
                "mc": (
                    "BOOLEAN",
                    {"default": False, "label_on": "max", "label_off": "min"},
                ),
                "sampling": (
                    "BOOLEAN",
                    {"default": False, "label_on": "max", "label_off": "min"},
                ),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)

    FUNCTION = "run_Model"
    CATEGORY = "CMA_3D/Algorithm"

    @torch.no_grad()
    def run_Model(
        self,
        # reference_image,
        # reference_mask,
        input_path,
        input_type,
        batchsize_per_gpu,
        seed,
        mc_level,
        mc,
        sampling,
    ):
        # single_image = torch_imgs_to_pils(reference_image, reference_mask)[0]
        checkpoint_dir = os.path.join(folder_paths.output_directory, "cma")
        os.makedirs(checkpoint_dir, exist_ok=True)
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            mixed_precision="fp16", project_dir=checkpoint_dir, kwargs_handlers=[kwargs]
        )

        # Load model
        model = MeshAnythingV2.from_pretrained("Yiwen-ntu/meshanythingv2")

        # Convert single_image to .npy and get path
        # single_image = single_image.resize((64,64))
        # npImage = np.array(single_image)
        # npImage = np.delete(arr=npImage, obj=3, axis=2)
        # file_name = datetime.datetime.now().strftime("%d_%H-%M-%S") + ".npy"
        # npy_file_path = os.path.join(checkpoint_dir, file_name)
        npy_file_path = input_path
        # with open(npy_file_path, "wb") as npy_file:
        #     np.save(npy_file, npImage)

        # create dataset
        set_seed(seed)
        dataset = Dataset(input_type, [npy_file_path], mc, mc_level)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize_per_gpu,
            drop_last=False,
            shuffle=False,
        )

        if accelerator.state.num_processes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        dataloader, model = accelerator.prepare(dataloader, model)
        begin_time = time.time()
        print("Generation Start!!!")
        scene_mesh = None
        with accelerator.autocast():
            for curr_iter, batch_data_label in enumerate(dataloader):
                curr_time = time.time()
                outputs = model(batch_data_label["pc_normal"], sampling=sampling)
                batch_size = outputs.shape[0]
                device = outputs.device

                for batch_id in range(batch_size):
                    recon_mesh = outputs[batch_id]
                    valid_mask = torch.all(
                        ~torch.isnan(recon_mesh.reshape((-1, 9))), dim=1
                    )
                    recon_mesh = recon_mesh[valid_mask]  # nvalid_face x 3 x 3

                    vertices = recon_mesh.reshape(-1, 3).cpu()
                    vertices_index = np.arange(len(vertices))  # 0, 1, ..., 3 x face
                    triangles = vertices_index.reshape(-1, 3)

                    scene_mesh = trimesh.Trimesh(
                        vertices=vertices,
                        faces=triangles,
                        force="mesh",
                        merge_primitives=True,
                    )
                    scene_mesh.merge_vertices()
                    scene_mesh.update_faces(scene_mesh.nondegenerate_faces())
                    scene_mesh.update_faces(scene_mesh.unique_faces())
                    scene_mesh.remove_unreferenced_vertices()
                    scene_mesh.fix_normals()
                    save_path = os.path.join(
                        checkpoint_dir, f'{batch_data_label["uid"][batch_id]}_gen.obj'
                    )
                    num_faces = len(scene_mesh.faces)
                    brown_color = np.array([255, 165, 0, 255], dtype=np.uint8)
                    face_colors = np.tile(brown_color, (num_faces, 1))

                    scene_mesh.visual.face_colors = face_colors
                    scene_mesh.export(save_path)
                    print(f"{save_path} Over!!")
        end_time = time.time()
        print(f"Total time: {end_time - begin_time}")
        # mesh = trimesh.load_mesh(given_mesh=scene_mesh)

        return (scene_mesh,)


class Save_3D_Mesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "mesh_file_path": (
                    "STRING",
                    {"default": "Mesh_%Y-%m-%d-%M-%S-%f.glb", "multiline": False},
                ),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mesh_file_path",)
    FUNCTION = "save_mesh"
    CATEGORY = "CMA_3D/Import|Export"

    def save_mesh(self, mesh, mesh_file_path):
        mesh_file_path = parse_save_filename(
            mesh_file_path,
            folder_paths.output_directory,
            SUPPORTED_3D_EXTENSIONS,
            self.__class__.__name__,
        )

        if mesh_file_path is not None:
            mesh.export(mesh_file_path)

        return (mesh_file_path,)
    
class Preview_3DMesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
            },
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "preview_mesh"
    CATEGORY = "CMA_3D/Visualize"
    
    def preview_mesh(self, mesh_file_path):
        
        mesh_folder_path, filename = os.path.split(mesh_file_path)
        
        if not os.path.isabs(mesh_file_path):
            mesh_file_path = os.path.join(folder_paths.output_directory, mesh_folder_path)
        
        if not filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
            print(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3D file extensions: {SUPPORTED_3D_EXTENSIONS}")
            mesh_file_path = ""
        
        previews = [
            {
                "filepath": mesh_file_path,
            }
        ]
        return {"ui": {"previews": previews}, "result": ()}


NODE_CLASS_MAPPINGS = {
    "CMA_Resize_Image_Foreground": Resize_Image_Foreground,
    "CMA_MeshAnything3D": MeshAnything3D,
    "CMA_Save_3D_Mesh": Save_3D_Mesh,
    "CMA_Preview_Mesh": Preview_3DMesh
}

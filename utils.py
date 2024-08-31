import trimesh
from mesh_to_pc import process_mesh_to_pc
import torch
import numpy as np
from PIL import Image
from numpy import asarray
import os
from os.path import isfile, join, exists, dirname
import datetime

# Utils


def conv_pil_tensor(img):
    return (torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0),)


def conv_tensor_pil(tsr):
    return Image.fromarray(
        np.clip(255.0 * tsr.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


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

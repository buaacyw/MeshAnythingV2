import mesh2sdf.core
import numpy as np
import skimage.measure
import trimesh
import time
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


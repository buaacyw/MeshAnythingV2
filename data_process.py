import datetime
import numpy as np
import argparse
import os
import pickle
import time
import trimesh
from multiprocessing import Pool, Manager
import logging
import warnings
import objaverse
import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import random
import multiprocessing
from mesh_to_pc import export_to_watertight

CPU_COUNT = multiprocessing.cpu_count()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_obj(save_path, min_num, max_num):
    annotations = objaverse.load_annotations()
    filtered_uids = []
    for uid in tqdm.tqdm(list(annotations.keys())):
        face_num = annotations[uid]['faceCount']
        vertex_num = annotations[uid]['vertexCount']
        if face_num > min_num and face_num < max_num and vertex_num>300:
            filtered_uids.append(uid)
    # os.makedirs("demo", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(filtered_uids, f)

def merge_and_scale(cur_data, progress, start_time, total_tasks, out_dir, min_num, max_num):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task_start_time = time.time()
            # uid:
            uid = cur_data.split("/")[-1].split(".")[0]
            # write a temp file
            if os.path.exists(os.path.join(out_dir, uid + '.tmp')) or os.path.exists(os.path.join(out_dir, uid + '.npz')):
                return
            temp_path = os.path.join(out_dir, uid + '.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump([], f)

            mesh = trimesh.load(cur_data, force='mesh')

            npz_to_save = {}
            if hasattr(mesh,"faces") and mesh.faces.shape[0] >= min_num and mesh.faces.shape[0] <= max_num:
                mesh.merge_vertices()
                mesh.update_faces(mesh.nondegenerate_faces())
                mesh.update_faces(mesh.unique_faces())
                mesh.remove_unreferenced_vertices()

                # judge = True
                vertices = np.array(mesh.vertices.copy())
                bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])  # type: ignore
                vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
                vertices = vertices / (bounds[1] - bounds[0]).max()  # -0.5 to 0.5 length 1
                vertices = vertices.clip(-0.5, 0.5)

                cur_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy())

                min_length = cur_mesh.bounding_box_oriented.edges_unique_length.min()

                npz_to_save['vertices'] = mesh.vertices
                npz_to_save['faces'] = mesh.faces
                npz_to_save['min_length'] = min_length
                npz_to_save['uid'] = uid
                npz_to_save['vertices_num'] = mesh.vertices.shape[0]
                npz_to_save['faces_num'] = mesh.faces.shape[0]
            if w:
                for warn in w:
                    logging.warning(f" {uid} : {str(warn.message)}")
                    print("uid warning:", uid)
                return
            # save pc_normal
            np.savez(os.path.join(out_dir, uid + '.npz'), **npz_to_save)

            os.remove(temp_path)
            task_end_time = time.time()
            task_duration = task_end_time - task_start_time
            progress.value += 1
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / progress.value * total_tasks
            remaining_time = estimated_total_time - elapsed_time
            remaining_td = datetime.timedelta(seconds=int(remaining_time))
            logging.info(f"This task: {task_duration:.2f} s, Already:{elapsed_time}, progress: {progress.value}/{total_tasks}, remaining{remaining_td}")
    except Exception as e:
        logging.error(f"Error in {uid}: {e}")


def objaverse_process(out_dir, obj_base_dir, json_save_path, min_num, max_num):
    all_paths = []
    with open(json_save_path, "r") as f:
        filtered_uids = json.load(f)

    for cur_cat in tqdm.tqdm(sorted(os.listdir(obj_base_dir))):
        cur_cat_dir = os.path.join(obj_base_dir, cur_cat)
        cur_files = sorted(os.listdir(cur_cat_dir))
        cur_files = [os.path.join(cur_cat_dir, x) for x in cur_files if "stl" in x or "obj" in x or "ply" in x or "glb" in x or "gltf" in x]
        cur_files = [x for x in cur_files if x.split("/")[-1].split(".")[0] in filtered_uids]
        all_paths.extend(cur_files)
        print(len(all_paths))
    print(len(all_paths))
    os.makedirs(out_dir, exist_ok=True)
    cpu_count = os.cpu_count()
    print(f"CPU count: {cpu_count}")
    total_tasks = len(all_paths)
    manager = Manager()
    progress = manager.Value('i', 0)
    start_time = time.time()

    with Pool(processes=CPU_COUNT) as pool:
        pool.starmap_async(merge_and_scale, [(data, progress, start_time, total_tasks, out_dir, min_num, max_num) for data in all_paths])
        pool.close()
        pool.join()

def process_npz_file(npz_path, npz_list,empty_list):
    try:
        with np.load(npz_path) as data:
            data_dict = {key: data[key] for key in data}
            if data_dict and data['faces'].shape[0] >= 20:
                data_dict['faces_num'] = data['faces'].shape[0]
                data_dict['vertices_num'] = data['vertices'].shape[0]
                data_dict['uid'] = npz_path.split('/')[-1].split('.')[0]
                npz_list.append(data_dict)
            else:
                empty_list.append(npz_path)
            print("empty:", len(empty_list),"npz_list:", len(npz_list))
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")

def to_npz_files(input_dir, out_dir, test_length=100):
    os.makedirs(out_dir, exist_ok=True)
    npz_list = []
    empty_list = []

    npz_files = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.npz')]

    with ThreadPoolExecutor() as executor:
        futures = []
        for filename in tqdm(npz_files, desc="Processing files"):
            npz_path = os.path.join(input_dir, filename)
            futures.append(executor.submit(process_npz_file, npz_path, npz_list, empty_list))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing results"):
            future.result()  # Wait for each future to complete

    assert len(npz_list) > test_length
    test_water_indices = np.random.choice(len(npz_list), size=test_length, replace=False)
    test_water = [npz_list[i] for i in test_water_indices]
    train_water = [npz_list[i] for i in range(len(npz_list)) if i not in test_water_indices]

    print(f"Train water: {len(train_water)}, Test water: {len(test_water)}")

    np.savez(os.path.join(out_dir, "train.npz"), npz_list=train_water)
    np.savez(os.path.join(out_dir, "test.npz"), npz_list=test_water)

def process_mesh(mesh_data, idx, save_dir):
    tmp_file = os.path.join(save_dir, f"{idx}.tmp")
    npz_file = os.path.join(save_dir, f"{idx}.npz")
    if os.path.exists(tmp_file) or os.path.exists(npz_file):
        return
    open(tmp_file, 'w').close()

    vertices = mesh_data['vertices']
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    cur_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh_data['faces'], force='mesh', merge_primitives=True)

    error = False
    try:
        water_mesh = export_to_watertight(cur_mesh, 7)

        water_p, face_idx = water_mesh.sample(20000, return_index=True)
        water_n = water_mesh.face_normals[face_idx]
    except Exception as e:
        print(f"Error in {idx}: {e}")
        error = True

    if not error:
        npz_to_save = {}
        pc_normal = np.concatenate([water_p, water_n], axis=-1, dtype=np.float16)
        npz_to_save['pc_normal'] = pc_normal
        np.savez(npz_file, **npz_to_save)

        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    return

def extract_point_cloud(npz_dir, pc_save_dir):
    print(f"Using {CPU_COUNT} CPU cores")

    for file_name in ['train.npz', 'test.npz']:
        npz_file = os.path.join(npz_dir, file_name)
        npz_list = np.load(npz_file, allow_pickle=True)
        npz_list = npz_list['npz_list'].tolist()
        cur_pc_save_dir = os.path.join(pc_save_dir, file_name.split('.')[0])
        os.makedirs(cur_pc_save_dir, exist_ok=True)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_mesh, mesh_data, idx, cur_pc_save_dir) for idx, mesh_data in enumerate(npz_list)]
            for future in as_completed(futures):
                future.result()

def merge_npz_files(npz_dir, pc_save_dir, final_data_save_dir):
    os.makedirs(final_data_save_dir, exist_ok=True)

    for cur_mode in ["train", "test"]:
        npz_file = os.path.join(npz_dir, f"{cur_mode}.npz")
        data = np.load(npz_file, allow_pickle=True)
        npz_list = data['npz_list'].tolist()
        updated_npz_list = []

        cur_pc_save_dir = os.path.join(pc_save_dir, f"{cur_mode}")
        output_npz_file = os.path.join(final_data_save_dir, f"{cur_mode}.npz")

        for idx, mesh_data in tqdm(enumerate(npz_list), total=len(npz_list), desc="Processing files"):
            npz_file = os.path.join(cur_pc_save_dir, f"{idx}.npz")
            if os.path.exists(npz_file):
                with np.load(npz_file) as npz_data:
                    mesh_data['pc_normal'] = npz_data['pc_normal']
                    mesh_data['metrics'] = npz_data['metrics']
                updated_npz_list.append(mesh_data)

        np.savez(output_npz_file, npz_list=updated_npz_list)
        print(f"Final data saved to {output_npz_file}")
        print(f"Total number of data samples: {len(updated_npz_list)}")

        # Load and check the saved file for any issues
        try:
            loaded_data = np.load(output_npz_file, allow_pickle=True)
            loaded_npz_list = loaded_data['npz_list'].tolist()
            print(f"Loaded {len(loaded_npz_list)} data samples successfully.")
        except Exception as e:
            print(f"Error loading the saved npz file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Data", add_help=False)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--min_face_num", required=True, type=float)
    parser.add_argument("--max_face_num", required=True, type=float)
    parser.add_argument("--obj_base_dir", required=True, type=float, help="Path to objaverse like this: xxxx/objaverse/hf-objaverse-v1/glbs")
    parser.add_argument("--test_length", required=True, type=int)

    args = parser.parse_args()
    os.makedirs(args.out_dir)

    json_save_path = os.path.join(args.out_dir, "filtered.json")

    random.seed(0)
    np.random.seed(0)

    filter_obj(json_save_path, args.min_face_num, args.max_face_num)
    objaverse_process(args.out_dir, args.obj_base_dir, json_save_path, args.min_face_num, args.max_face_num)
    to_npz_files(args.out_dir, args.out_dir+"_npz", test_length=args.test_length)
    extract_point_cloud(npz_dir=args.out_dir+"_npz", pc_save_dir=args.out_dir+"_pc")
    merge_npz_files(npz_dir=args.out_dir+"_npz", pc_save_dir=args.out_dir+"_pc", final_data_save_dir=args.out_dir+"_final")
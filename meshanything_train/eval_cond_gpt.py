import os
import time
import torch
import numpy as np
from collections import defaultdict
from meshanything_train.misc import SmoothedValue

from plyfile import PlyData, PlyElement
import trimesh

def calc_chamfer_loss(vertices_gt, vertices_recon):
    dist1 = torch.cdist(vertices_gt, vertices_recon, p=2).min(dim=1)[0]
    dist2 = torch.cdist(vertices_recon, vertices_gt, p=2).min(dim=1)[0]
    chamfer_loss = dist1.mean() + dist2.mean()
    return chamfer_loss

def write_gt(vertices, triangles, save_path ):
    face_mask = triangles[:, 0] != -1
    triangles = triangles[face_mask].cpu()
    vertice_mask = ~(vertices == -1).all(dim=1)
    gt_mesh = vertices[vertice_mask].cpu()

    scene_mesh = trimesh.Trimesh(vertices=gt_mesh, faces=triangles, force="mesh", merge_primitives=True)
    scene_mesh.merge_vertices()
    scene_mesh.update_faces(scene_mesh.nondegenerate_faces())
    scene_mesh.update_faces(scene_mesh.unique_faces())
    scene_mesh.remove_unreferenced_vertices()
    scene_mesh.fix_normals()

    write_mesh_with_color(scene_mesh, save_path)

def write_mesh_with_color(mesh, save_path):
    num_faces = len(mesh.faces)
    brown_color = np.array([255, 165, 0, 255], dtype=np.uint8)
    face_colors = np.tile(brown_color, (num_faces, 1))
    mesh.visual.face_colors = face_colors
    mesh.export(save_path)

@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    dataset_loader,
    accelerator,
    logger,
    curr_train_iter=-1,
    test_only = False,
):
    do_generate = False
    num_batches = len(dataset_loader)
    logger.info(f"Start evaluating on {num_batches} batches, data samples: {len(dataset_loader.dataset)}")
    time_delta = SmoothedValue(window_size=10)
    before_eval_time = time.time()
    model.eval()

    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    sample_id = 0
    if curr_train_iter % args.generate_every_iteration == 0 or test_only:
        storage_dir = os.path.join(args.checkpoint_dir,  'visualization', f'iter_{curr_train_iter}')
        os.makedirs(storage_dir, exist_ok=True)

    cur_rank = accelerator.process_index
    loss_gather = defaultdict(list)
    iter_count = 0

    for curr_iter, batch_data_label in enumerate(dataset_loader):

        curr_time = time.time()
        if "vertices" in batch_data_label and args.data_n_max_triangles == args.gen_n_max_triangles:
            loss_outputs = model(batch_data_label)

            for key, value in loss_outputs.items():
                if 'loss' in key.lower():
                    gathered_value = accelerator.gather(value)
                    loss_gather["val_"+key].append(gathered_value.mean().item())
        if iter_count % args.generate_every_data == 0 and (curr_train_iter % args.generate_every_iteration == 0 or test_only):
            do_generate = True
            recon_faces = model(batch_data_label, is_eval=True)
            batch_size = recon_faces.shape[0]
            device = recon_faces.device

            for batch_id in range(batch_size):
                if 'uid' in batch_data_label:
                    uid = batch_data_label['uid'][batch_id]
                else:
                    uid = ""
                prefix = f"{cur_rank}_{sample_id}"

                write_gt(batch_data_label['vertices'][batch_id], batch_data_label['faces'][batch_id], os.path.join(storage_dir, f'{prefix}_gt_{uid}.ply'))
                write_gt(batch_data_label['gt_v'][batch_id], batch_data_label['gt_f'][batch_id], os.path.join(storage_dir, f'{prefix}_tgt_{uid}.ply'))

                pcd = batch_data_label['pc_normal'][batch_id].cpu().numpy()[:, :3]
                pcd = pcd / np.abs(pcd).max() * 0.50
                vertex = np.array([tuple(point) for point in pcd], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
                el = PlyElement.describe(vertex, 'vertex')
                PlyData([el], text=True).write(os.path.join(storage_dir, f'{prefix}_cond_{uid}.ply'))
                # store reconstructed faces
                try:
                    success = True
                    recon_mesh = recon_faces[batch_id]
                    if recon_mesh[-1, -1, -1] == 0:
                        success = False
                    valid_mask = torch.all(~torch.isnan(recon_mesh.reshape((-1,9))), dim=1)
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

                    if success:
                        cur_save_path = os.path.join(storage_dir, f'{prefix}_suc_generate_{uid}.ply')
                    else:
                        cur_save_path = os.path.join(storage_dir, f'{prefix}_fail_generate_{uid}.ply')

                    write_mesh_with_color(scene_mesh, cur_save_path)
                except Exception as e:
                    print("eval_cond_gpt calculate error", e)
                sample_id += 1
        iter_count += 1
        time_delta.update(time.time() - curr_time)

        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(
            f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; " +
            f"Evaluating on iter: {curr_train_iter}; "
            f"Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
        )
    loss_avg = {
        key: torch.tensor(loss_list, dtype=torch.float32).mean().item() \
            for key, loss_list in loss_gather.items()
    }
    # reduce res from all nodes
    if do_generate:

        file_list = sorted(os.listdir(storage_dir))
        all_gt = [name for name in file_list if "_gt_" in name]

        file_list = [name for name in file_list if "_generate_" in name]
        file_list_gt = [name.replace("_suc_generate_","_gt_").replace("_fail_generate_","_gt_") for name in file_list]
        all_num = len(all_gt)

        success_num = 0

        post_pc_chamfer_loss_list = []
        post_pc_large_chamfer_loss_list = []
        post_un_success_pc_chamfer_loss_list = []
        post_suc_face_ratio_list = []
        post_fail_face_ratio_list = []

        post_pc_tgt_chamfer_loss_list = []

        post_face_ratio_list = []
        if len(file_list) > 0:
            for file_name, file_name_gt in zip(file_list, file_list_gt):
                try:
                    mesh_1 = trimesh.load(os.path.join(storage_dir, file_name))
                    mesh_2 = trimesh.load(os.path.join(storage_dir, file_name_gt))
                    mesh_3 = trimesh.load(os.path.join(storage_dir, file_name_gt.replace("_gt_", "_tgt_")))

                    assert len(mesh_1.faces) > 0, f"Generated faces are empty"
                    gen_points = mesh_1.sample(10000)
                    gen_points = torch.tensor(gen_points, device=device, dtype=torch.float32)

                    gt_points = mesh_2.sample(10000)
                    gt_points = torch.tensor(gt_points, device=device, dtype=torch.float32)

                    tgt_points = mesh_3.sample(10000)
                    tgt_points = torch.tensor(tgt_points, device=device, dtype=torch.float32)

                    cur_chamfer_loss = calc_chamfer_loss(gt_points, gen_points).item()
                    cur_tgt_chamfer_loss = calc_chamfer_loss(tgt_points, gen_points).item()
                    post_pc_chamfer_loss_list.append(cur_chamfer_loss)
                    post_pc_tgt_chamfer_loss_list.append(cur_tgt_chamfer_loss)

                    if len(mesh_3.faces) > 800:
                        post_pc_large_chamfer_loss_list.append(cur_chamfer_loss)
                    face_ratio = len(mesh_1.faces) / len(mesh_2.faces)
                    post_face_ratio_list.append(face_ratio)
                    if "_suc_" in file_name:
                        success_num += 1
                        post_suc_face_ratio_list.append(face_ratio)
                    else:
                        post_un_success_pc_chamfer_loss_list.append(cur_chamfer_loss)
                        post_fail_face_ratio_list.append(face_ratio)
                except Exception as e:
                    print("post error:", e)
                    continue
        if len(post_pc_chamfer_loss_list) == 0:
            post_pc_chamfer_loss_list.append(-1)
            post_pc_tgt_chamfer_loss_list.append(-1)
            post_face_ratio_list.append(-1)
            post_un_success_pc_chamfer_loss_list.append(-1)
            post_fail_face_ratio_list.append(-1)
            post_suc_face_ratio_list.append(-1)

        if len(post_pc_large_chamfer_loss_list) == 0:
            post_pc_large_chamfer_loss_list.append(-1)
        post_pc_chamfer_loss_list = np.asarray(post_pc_chamfer_loss_list)
        loss_avg['post_pc_chamfer_distance'] = np.mean(post_pc_chamfer_loss_list)

        post_pc_large_chamfer_loss_list = np.asarray(post_pc_large_chamfer_loss_list)
        loss_avg['post_pc_large_chamfer_distance'] = np.mean(post_pc_large_chamfer_loss_list)

        post_un_success_pc_chamfer_loss_list = np.asarray(post_un_success_pc_chamfer_loss_list)
        loss_avg['post_un_success_pc_chamfer_distance'] = np.mean(post_un_success_pc_chamfer_loss_list)

        post_fail_face_ratio_list = np.asarray(post_fail_face_ratio_list)
        loss_avg['post_fail_face_ratio'] = np.mean(post_fail_face_ratio_list)

        post_suc_face_ratio_list = np.asarray(post_suc_face_ratio_list)
        loss_avg['post_suc_face_ratio'] = np.mean(post_suc_face_ratio_list)

        post_pc_tgt_chamfer_loss_list = np.asarray(post_pc_tgt_chamfer_loss_list)
        loss_avg['post_pc_tgt_chamfer_distance'] = np.mean(post_pc_tgt_chamfer_loss_list)

        post_face_ratio_list = np.asarray(post_face_ratio_list)
        loss_avg['post_face_ratio'] = np.mean(post_face_ratio_list)
        if all_num > 0:
            loss_avg['success_ratio'] = success_num/all_num
        else:
            print("ERROR: all_num is zero!!!!!!!!!!!!")
    after_eval_time = time.time()
    logger.info(f"Finished evaluating in {after_eval_time - before_eval_time:.2f} seconds")

    return {}, loss_avg
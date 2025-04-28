import os

import numpy as np
from meshanything_train.eval_cond_gpt import evaluate as evaluate_cond_gpt

import trimesh
import networkx as nx

def sample_surface_points(vertices, faces, sample_num=4096):
    scene_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, force="mesh", merge_primitives=True)

    # sample surface points/normals/colors
    points, face_idx = scene_mesh.sample(sample_num, return_index=True)
    normals = scene_mesh.face_normals[face_idx]

    points = points.astype(np.float16)
    points = np.clip(points, -0.9995, 0.9995)
    normals = normals.astype(np.float16)

    pc_normal = np.concatenate([points, normals], axis=-1)

    return pc_normal

def rotate_mesh(vertices):
    angle_y = np.random.uniform(0, 2 * np.pi)

    R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])

    rotated_vertices = np.dot(vertices, R_y.T)

    return rotated_vertices

class Dataset:
    def __init__(self, args, split_set="train"):
        super().__init__()
        self.num_tokens = args.n_discrete_size
        self.no_aug = args.no_aug
        self.input_pc_num = args.input_pc_num
        self.max_seq_ratio = args.max_seq_ratio
        self.split_set = split_set
        if split_set == "test":
            self.no_aug = True
        self.shift_scale = args.shift_scale
        self.data = np.load(os.path.join(args.data_dir, split_set + ".npz"), allow_pickle=True)['npz_list'].tolist()

        self.max_triangles = args.data_n_max_triangles
        self.max_ratio = 0.70
        self.max_vertices = int(self.max_triangles * self.max_ratio)

        self.min_triangles = args.n_min_triangles
        self.min_ratio = 0.40

        self.max_token_length = int(self.max_triangles * 9 * self.max_seq_ratio)


        self.data = [cur_data for cur_data in self.data
                     if cur_data['faces_num'] <= self.max_triangles
                     and cur_data['faces_num'] >= self.min_triangles]

        print(f"{split_set} dataset total data samples: {len(self.data)}")

        self.data = [cur_data for cur_data in self.data
                     if cur_data['vertices_num'] / cur_data['faces_num'] <= self.max_ratio
                     and cur_data['vertices_num'] / cur_data['faces_num'] >= self.min_ratio]
        print(f"{split_set} dataset total data samples after filter: {len(self.data)}")

        # check whether args has llm
        self.eval_func = evaluate_cond_gpt


    def __len__(self):
        return len(self.data)

    def tokenize(self, mesh):
        naive_v_length = mesh.faces.shape[0] * 9

        graph = mesh.vertex_adjacency_graph

        unvisited_faces = mesh.faces.copy()
        dis_vertices = np.asarray((mesh.vertices.copy() + 0.5) * self.num_tokens)

        sequence = []
        while unvisited_faces.shape[0] > 0:
            # find the face with the smallest index
            if len(sequence) == 0 or sequence[-1] == -1:
                cur_face = unvisited_faces[0]
                unvisited_faces = unvisited_faces[1:]
                sequence.extend(cur_face.tolist())
            else:
                cur_cache = sequence[-2:]
                commons = sorted(list(nx.common_neighbors(graph, cur_cache[0], cur_cache[1])))
                next_token = None
                for common in commons:
                    common_face = sorted(np.array(cur_cache + [common]))
                    # find index of common face
                    equals = np.where((unvisited_faces == common_face).all(axis=1))[0]
                    assert len(equals) == 1 or len(equals) == 0
                    if len(equals) == 1:
                        next_token = common
                        next_face_index = equals[0]
                        break
                if next_token is not None:
                    unvisited_faces = np.delete(unvisited_faces, next_face_index, axis=0)
                    sequence.append(int(next_token))
                else:
                    sequence.append(-1)

        final_sequence = []
        id_sequence = []
        split_flag = 3
        for token_id in sequence:
            if token_id == -1:
                final_sequence.append(self.num_tokens)
                id_sequence.append(3)
                split_flag = 3
            else:
                final_sequence.extend(dis_vertices[token_id].tolist())
                if split_flag == 0:
                    id_sequence.extend([7,8,9])
                else:
                    split_flag -= 1
                    id_sequence.extend([4,5,6])

        assert len(final_sequence) == len(id_sequence)
        cur_ratio = len(final_sequence) / naive_v_length
        if cur_ratio >= self.max_seq_ratio:
            # print(f"token sequence too long: {cur_ratio}")
            return None, None
        else:
            return final_sequence, id_sequence

    def sort_vertices_and_faces(self, vertices_, faces_):
        assert (vertices_ <= 0.5).all() and (vertices_ >= -0.5).all() # [-0.5, 0.5]
        vertices = (vertices_+0.5) * self.num_tokens # [0, num_tokens]
        vertices -= 0.5 # for evenly distributed, [-0.5, num_tokens -0.5] will be round to 0 or num_tokens (-1)
        vertices_quantized_ = np.clip(vertices.round(), 0, self.num_tokens-1).astype(int)  # [0, num_tokens -1]
        origin_face_num = len(faces_)

        cur_mesh = trimesh.Trimesh(vertices=vertices_quantized_, faces=faces_)

        cur_mesh.merge_vertices()
        cur_mesh.update_faces(cur_mesh.nondegenerate_faces())
        cur_mesh.update_faces(cur_mesh.unique_faces())
        cur_mesh.remove_unreferenced_vertices()

        if len(cur_mesh.faces) < self.min_triangles/3*2 or len(cur_mesh.faces) < origin_face_num*0.2:
            return None, None

        sort_inds = np.lexsort(cur_mesh.vertices.T)
        vertices = cur_mesh.vertices[sort_inds]
        faces = [np.argsort(sort_inds)[f] for f in cur_mesh.faces]

        faces = [sorted(sub_arr) for sub_arr in faces]

        def sort_faces(face):
            return face[0], face[1], face[2]

        faces = sorted(faces, key=sort_faces)

        vertices = vertices / self.num_tokens - 0.5  # [0, num_tokens -1] to [-0.5, 0.5)  for computing

        return vertices, faces

    def __getitem__(self, idx):
        data = self.data[idx]
        # objaverse and shapenet mesh isn't normalized but pc is normalized to -0.5 and 0.5 (roughly)
        vertices = data['vertices']
        faces = data['faces']
        assert vertices.shape[1] == 3 and faces.shape[1] ==3

        assert self.min_triangles <= len(faces) <= self.max_triangles
        assert self.min_ratio <= len(vertices) / len(faces) <= self.max_ratio
        data_dict = {}
        if 'category' in data:
            data_dict['uid'] = data['category']+'_'+data['uid']
        else:
            data_dict['uid'] = data['uid']
        # scale x, y, z
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()
        # aligned from now on

        if not self.no_aug:
            x_lims = (0.75, 1.25)
            y_lims = (0.75, 1.25)
            z_lims = (0.75, 1.25)

            x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
            y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
            z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
            vertices = np.stack([vertices[:, 0] * x, vertices[:, 1] * y, vertices[:, 2] * z], axis=-1)
        # normalize normal

        # normalize x, y, z
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        if not self.no_aug:
            vertices = rotate_mesh(vertices)

        # scale to -0.5 to 0.5
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()

        # shift x, y, z
        if not self.no_aug:
            x_lims=(-self.shift_scale, self.shift_scale)
            y_lims=(-self.shift_scale, self.shift_scale)
            z_lims=(-self.shift_scale, self.shift_scale)
            x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
            y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
            z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
            x = max(min(x, 0.5 - vertices[:, 0].max()) , -0.5 - vertices[:, 0].min())
            y = max(min(y, 0.5 - vertices[:, 1].max()) , -0.5 - vertices[:, 1].min())
            z = max(min(z, 0.5 - vertices[:, 2].max()) , -0.5 - vertices[:, 2].min())
            vertices = np.stack([vertices[:, 0] + x, vertices[:, 1] + y, vertices[:, 2] + z], axis=-1)

        vertices = vertices.clip(-0.5, 0.5)
        assert vertices.min() >= -0.5 and vertices.max() <= 0.5

        gt_vertices, gt_faces = vertices.copy(), faces.copy()
        vertices, faces = self.sort_vertices_and_faces(vertices, faces)
        if vertices is None:
            # sample another data
            # print("discrete reroll!")
            return self.__getitem__(np.random.randint(0, len(self.data)))
        else:
            dis_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            sequence, id_sequence = self.tokenize(dis_mesh)
            if sequence is None:
                return self.__getitem__(np.random.randint(0, len(self.data)))
            assert len(sequence) == len(id_sequence)
            pad_sequence = np.ones(self.max_token_length) * -1
            pad_sequence[:len(sequence)] = sequence

            pad_id_sequence = np.ones(self.max_token_length) * -1
            pad_id_sequence[:len(id_sequence)] = id_sequence

            pad_vertices = np.ones((self.max_vertices, 3)) * -1
            pad_faces = np.ones((self.max_triangles, 3)) * -1

            num_vertices, num_faces = len(vertices), len(faces)
            pad_vertices[:num_vertices] = vertices
            pad_faces[:num_faces] = faces

            pad_gt_vertices = np.ones((self.max_vertices, 3)) * -1
            pad_gt_faces = np.ones((self.max_triangles, 3)) * -1

            num_gt_vertices, num_gt_faces = len(gt_vertices), len(gt_faces)
            pad_gt_vertices[:num_gt_vertices] = gt_vertices
            pad_gt_faces[:num_gt_faces] = gt_faces

            data_dict['vertices'] = np.asarray(pad_vertices).astype(np.float16)
            data_dict['faces'] = np.asarray(pad_faces).astype(np.int64)
            data_dict['sequence'] = np.asarray(pad_sequence).astype(np.int64)
            data_dict['id_sequence'] = np.asarray(pad_id_sequence).astype(np.int64)
            data_dict['gt_v'] = np.asarray(pad_gt_vertices).astype(np.float16)
            data_dict['gt_f'] = np.asarray(pad_gt_faces).astype(np.int64)
            # pc_coor max to 0.9995

            vertices *= (2 * 0.9995)
            assert (-0.9995 <= vertices).all() and (vertices <= 0.9995).all()
            data_dict['pc_normal'] = sample_surface_points(vertices, faces, self.input_pc_num)

            return data_dict



import numpy as np
import trimesh
import networkx as nx
import os

def mesh_sort(vertices_, faces_):
    assert (vertices_ <= 0.5).all() and (vertices_ >= -0.5).all()  # [-0.5, 0.5]
    vertices = (vertices_ + 0.5) * 128  # [0, num_tokens]
    vertices -= 0.5  # for evenly distributed, [-0.5, num_tokens -0.5] will be round to 0 or num_tokens (-1)
    vertices_quantized_ = np.clip(vertices.round(), 0, 128 - 1).astype(int)  # [0, num_tokens -1]

    cur_mesh = trimesh.Trimesh(vertices=vertices_quantized_, faces=faces_)

    cur_mesh.merge_vertices()
    cur_mesh.update_faces(cur_mesh.nondegenerate_faces())
    cur_mesh.update_faces(cur_mesh.unique_faces())
    cur_mesh.remove_unreferenced_vertices()

    sort_inds = np.lexsort(cur_mesh.vertices.T)
    vertices = cur_mesh.vertices[sort_inds]
    faces = [np.argsort(sort_inds)[f] for f in cur_mesh.faces]

    faces = [sorted(sub_arr) for sub_arr in faces]

    def sort_faces(face):
        return face[0], face[1], face[2]

    faces = sorted(faces, key=sort_faces)

    vertices = vertices / 128 - 0.5  # [0, num_tokens -1] to [-0.5, 0.5)  for computing

    return vertices, faces

def adjacent_mesh_tokenization(mesh):
    naive_v_length = mesh.faces.shape[0] * 9

    graph = mesh.vertex_adjacency_graph

    unvisited_faces = mesh.faces.copy()
    dis_vertices = np.asarray((mesh.vertices.copy() + 0.5) * 128)

    sequence = []
    while unvisited_faces.shape[0] > 0:
        # find the face with the smallest index
        if len(sequence) == 0 or sequence[-1] == -1:
            cur_face = unvisited_faces[0]
            unvisited_faces = unvisited_faces[1:]
            sequence.extend(cur_face.tolist())
        else:
            last_vertices = sequence[-2:]
            # find common neighbors
            commons = sorted(list(nx.common_neighbors(graph, last_vertices[0], last_vertices[1])))
            next_token = None
            for common in commons:
                common_face = sorted(np.array(last_vertices + [common]))
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
    for token_id in sequence:
        if token_id == -1:
            final_sequence.append(128)
        else:
            final_sequence.extend(dis_vertices[token_id].tolist())

    cur_ratio = len(final_sequence) / naive_v_length

    return cur_ratio


if __name__ == "__main__":
    # read_ply
    data_dir = 'gt_examples'
    data_list = sorted(os.listdir(data_dir))
    data_list = [os.path.join(data_dir, x) for x in data_list if x.endswith('.ply') or x.endswith('.obj')]
    ratio_list = []
    for idx, cur_data in enumerate(data_list):
        cur_mesh = trimesh.load(cur_data)

        vertices = cur_mesh.vertices
        faces = cur_mesh.faces
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()
        vertices = vertices.clip(-0.5, 0.5)

        vertices, faces = mesh_sort(vertices, faces)
        dis_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        try:
            cur_ratio = adjacent_mesh_tokenization(dis_mesh)

            ratio_list.append(cur_ratio)
        except Exception as e:
            print(e)

        # print mean and variance of ratio:
        print(f"mean ratio: {np.mean(ratio_list)}, variance ratio: {np.var(ratio_list)}")
import numpy as np
import collections
import trimesh
from trimesh.remesh import grouping
from scipy.spatial.transform import Rotation as R

def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()
# https://github.com/ratcave/wavefront_reader
def read_mtlfile(fname):
    materials = {}
    with open(fname) as f:
        lines = f.read().splitlines()

    for line in lines:
        if line:
            split_line = line.strip().split(' ', 1)
            if len(split_line) < 2:
                continue

            prefix, data = split_line[0], split_line[1]
            if 'newmtl' in prefix:
                material = {}
                materials[data] = material
            elif materials:
                if data:
                    split_data = data.strip().split(' ')

                    # assume texture maps are in the same level
                    # WARNING: do not include space in your filename!!
                    if 'map' in prefix:
                        material[prefix] = split_data[-1].split('\\')[-1]
                    elif len(split_data) > 1:
                        material[prefix] = tuple(float(d) for d in split_data)
                    else:
                        try:
                            material[prefix] = int(data)
                        except ValueError:
                            material[prefix] = float(data)

    return materials

def rotate_mesh(mesh, angle, axis='x'):
    mesh2 = mesh.copy()
    mesh2.visual = mesh.visual
    vertices = mesh.vertices
    vertices_re = (np.zeros_like(vertices))
    if axis == 'y':  # pitch
        rotation_axis = np.array([1, 0, 0])
    elif axis == 'x':  # yaw
        rotation_axis = np.array([0, 1, 0])
    elif axis == 'z':  # roll
        rotation_axis = np.array([0, 0, 1])
    else:  # default is x (yaw)
        rotation_axis = np.array([0, 1, 0])

    for i in range(vertices.shape[0]):
        vec = vertices[i, :]
        rotation_degrees = angle
        rotation_radians = np.radians(rotation_degrees)

        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)
        rotated_vec = rotation.apply(vec)
        vertices_re[i, :] = rotated_vec
    mesh2.vertices = vertices_re
    return mesh2

def load_obj_mesh_mtl(mesh_file):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    # face per material
    face_data_mat = {}
    face_norm_data_mat = {}
    face_uv_data_mat = {}

    # current material name
    mtl_data = None
    cur_mat = None

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)
        elif values[0] == 'mtllib':
            mtl_data = read_mtlfile(mesh_file.replace(mesh_file.split('/')[-1],values[1]))
        elif values[0] == 'usemtl':
            cur_mat = values[1]
        elif values[0] == 'f':
            # local triangle data
            l_face_data = []
            l_face_uv_data = []
            l_face_norm_data = []

            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]) if int(x.split('/')[0]) < 0 else int(x.split('/')[0])-1, values[1:4]))
                l_face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]) if int(x.split('/')[0]) < 0 else int(x.split('/')[0])-1, [values[3], values[4], values[1]]))
                l_face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]) if int(x.split('/')[0]) < 0 else int(x.split('/')[0])-1, values[1:4]))
                l_face_data.append(f)
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]) if int(x.split('/')[1]) < 0 else int(x.split('/')[1])-1, values[1:4]))
                    l_face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]) if int(x.split('/')[1]) < 0 else int(x.split('/')[1])-1, [values[3], values[4], values[1]]))
                    l_face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]) if int(x.split('/')[1]) < 0 else int(x.split('/')[1])-1, values[1:4]))
                    l_face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]) if int(x.split('/')[2]) < 0 else int(x.split('/')[2])-1, values[1:4]))
                    l_face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]) if int(x.split('/')[2]) < 0 else int(x.split('/')[2])-1, [values[3], values[4], values[1]]))
                    l_face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]) if int(x.split('/')[2]) < 0 else int(x.split('/')[2])-1, values[1:4]))
                    l_face_norm_data.append(f)

            face_data += l_face_data
            face_uv_data += l_face_uv_data
            face_norm_data += l_face_norm_data

            if cur_mat is not None:
                if cur_mat not in face_data_mat.keys():
                    face_data_mat[cur_mat] = []
                if cur_mat not in face_uv_data_mat.keys():
                    face_uv_data_mat[cur_mat] = []
                if cur_mat not in face_norm_data_mat.keys():
                    face_norm_data_mat[cur_mat] = []
                face_data_mat[cur_mat] += l_face_data
                face_uv_data_mat[cur_mat] += l_face_uv_data
                face_norm_data_mat[cur_mat] += l_face_norm_data

    vertices = np.array(vertex_data)
    faces = np.array(face_data)

    norms = np.array(norm_data)
    norms = normalize_v3(norms)
    face_normals = np.array(face_norm_data)

    uvs = np.array(uv_data)
    face_uvs = np.array(face_uv_data)

    out_tuple = (vertices, faces, norms, face_normals, uvs, face_uvs)

    if cur_mat is not None and mtl_data is not None:
        for key in face_data_mat:
            face_data_mat[key] = np.array(face_data_mat[key])
            face_uv_data_mat[key] = np.array(face_uv_data_mat[key])
            face_norm_data_mat[key] = np.array(face_norm_data_mat[key])

        out_tuple += (face_data_mat, face_norm_data_mat, face_uv_data_mat, mtl_data)

    return out_tuple
def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr

def subdivide_keti(vertices, uv, faces, face_uvs, face_index=None):
    def faces_to_edges(faces, return_index=False):
        faces = np.asanyarray(faces)

        # each face has three edges
        edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))

        if return_index:
            # edges are in order of faces due to reshape
            face_index = np.tile(np.arange(len(faces)), (3, 1)).T.reshape(-1)
            return edges, face_index
        return edges

    if face_index is None:
        face_index = np.arange(len(faces))
    else:
        face_index = np.asanyarray(face_index)

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_index]

    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    uv_edges = np.sort(faces_to_edges(face_uvs), axis=1)

    dict = collections.defaultdict(tuple)
    for k in range(len(edges)):
        dict[tuple(edges[k])] = uv_edges[k]

    unique, inverse = grouping.unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

    mid_uvs = [dict[tuple(i)] for i in edges[unique]]
    mid_uvs = [[(uv[i[0]][0] + uv[i[1]][0]) / 2, (uv[i[0]][1] + uv[i[1]][1]) / 2] for i in mid_uvs]

    # the new faces_subset with correct winding
    f = np.column_stack([faces_subset[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces_subset[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces_subset[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))
    # add the 3 new faces_subset per old face
    new_faces = np.vstack((faces, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]

    new_vertices = np.vstack((vertices, mid))
    # new_uv = np.vstack((uv, mid_uvs))

    return new_vertices, new_faces, mid_uvs
def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm

def load_file2info(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, list):
        mesh_file = mesh_file[0]

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)
        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces

def load_file2mesh(mesh_file, texture, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []
    dict = collections.defaultdict(int)

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file

    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
            # vt = list(map(float, values[4:6]))
            # uv_data.append(vt)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)
        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f_c = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f_c)
                    f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                    dict[f[0] - 1] = f_c[0] - 1
                    dict[f[1] - 1] = f_c[1] - 1
                    dict[f[2] - 1] = f_c[2] - 1
                else:
                    face_uv_data.append([1, 1, 1])

            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertex_colors = []
    for k in range(len(vertex_data)):
        if k in dict:
            vertex_colors.append(uv_data[dict[k]])
        else:
            vertex_colors.append([0.0, 0.0])

    w, h = texture.shape[0], texture.shape[1]
    vertices = np.array(vertex_data)
    visuals = np.array(vertex_colors)
    faces = np.array(face_data) - 1

    vertex_colors = visuals
    vertex_colors = [[int(item[0] * w), int(item[1] * h)] for item in vertex_colors]

    for cnt, item in enumerate(vertex_colors):
        u = int(item[0])
        v = int(item[1])
        u = w - 1 if u >= w else u
        v = h - 1 if v >= h else v
        # if u == texture.shape[0]:
        #     u -= 1
        # if v == texture.shape[1]:
        #     v -= 1
        vertex_colors[cnt] = texture[u, v, :]
        # cnt += 1

    visuals = np.array(vertex_colors)
    mesh = trimesh.Trimesh(vertices=vertices,
                           faces=faces,
                           vertex_colors=visuals,
                           process=False)

    return mesh


def compute_tangent_from_normals(normals):
    c1 = np.cross(normals, np.array([0, 1, 0.0]))
    tan = c1
    normalize_v3(tan)
    btan = np.cross(normals, tan)
    return tan, btan

def compute_tangent(vertices, faces, normals, uvs, faceuvs):
    # NOTE: this could be numerically unstable around [0,0,1]
    # but other current solutions are pretty freaky somehow
    c1 = np.cross(normals, np.array([0, 1, 0.0]))
    tan = c1
    normalize_v3(tan)
    btan = np.cross(normals, tan)

    # NOTE: traditional version is below

    # pts_tris = vertices[faces]
    # uv_tris = uvs[faceuvs]

    # W = np.stack([pts_tris[::, 1] - pts_tris[::, 0], pts_tris[::, 2] - pts_tris[::, 0]],2)
    # UV = np.stack([uv_tris[::, 1] - uv_tris[::, 0], uv_tris[::, 2] - uv_tris[::, 0]], 1)

    # for i in range(W.shape[0]):
    #     W[i,::] = W[i,::].dot(np.linalg.inv(UV[i,::]))

    # tan = np.zeros(vertices.shape, dtype=vertices.dtype)
    # tan[faces[:,0]] += W[:,:,0]
    # tan[faces[:,1]] += W[:,:,0]
    # tan[faces[:,2]] += W[:,:,0]

    # btan = np.zeros(vertices.shape, dtype=vertices.dtype)
    # btan[faces[:,0]] += W[:,:,1]
    # btan[faces[:,1]] += W[:,:,1]
    # btan[faces[:,2]] += W[:,:,1]

    # normalize_v3(tan)

    # ndott = np.sum(normals*tan, 1, keepdims=True)
    # tan = tan - ndott * normals

    # normalize_v3(btan)
    # normalize_v3(tan)

    # tan[np.sum(np.cross(normals, tan) * btan, 1) < 0,:] *= -1.0

    return tan, btan

# https://github.com/classner/up/blob/master/up_tools/camera.py
def rotateY(theta):
    """Rotate all points in a 2D array around the y axis."""
    return np.array([
        [np.cos(theta / 180 * np.pi), 0.0, np.sin(theta / 180 * np.pi), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-np.sin(theta / 180 * np.pi), 0.0, np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotateX( points, angle ):
    """Rotate all points in a 2D array around the x axis."""
    rx = np.array([
        [1.,    0.,                 0.           ],
        [0.,    np.cos(angle),     -np.sin(angle)],
        [0.,    np.sin(angle),     np.cos(angle) ]
    ])
    return np.dot(points, rx)

def rotateZ( points, angle ):
    """Rotate all points in a 2D array around the z axis."""
    rz = np.array([
        [np.cos(angle),     -np.sin(angle),     0. ],
        [np.sin(angle),     np.cos(angle),      0. ],
        [0.,                0.,                 1. ]
    ])
    return np.dot(points, rz)

if __name__ == '__main__':
    pts, tri, nml, trin, uvs, triuv = load_file2info('/home/ICT2000/ssaito/Documents/Body/tmp/Baseball_Pitching/0012.obj', True, True)
    compute_tangent(pts, tri, uvs, triuv)
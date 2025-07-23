import torch

def to_torch(data, device):
    return data.to(device, dtype=torch.float32) if torch.is_tensor(data) else torch.tensor(data, dtype=torch.float32, device=device)

class TexturedMesh:
    """ Triangle mesh defined by an indexed vertex buffer.

    Args:
        vertices (tensor): Vertex buffer (Vx3)
        indices (tensor): Index buffer (Fx3)
        uv_vts (tensor) : UV coordinate for vertices (Vx2)
        uv_faces (tensor) : indices of uv_vts w.r.t. faces (Fx3)
        tex (tensor) : texture map (HxWx3)
        device (torch.device): Device where the mesh buffers are stored
    """

    def __init__(self, vertices, indices, uv_vts, uv_faces, tex, device='cpu'):
        self.device = device

        self.vertices = to_torch(vertices, device)
        self.indices = to_torch(indices, device)
        self.uv_vts = to_torch(uv_vts, device)
        self.uv_faces = to_torch(uv_faces, device)
        self.tex = to_torch(tex, device)

        # additional information that are calculated internally
        self.face_normals = None
        self.vertex_normals = None
        self._edges = None
        self._connected_faces = None
        self._laplacian = None

        if self.indices is not None:
            self.compute_normals()

    def to(self, device):
        mesh = TexturedMesh(self.vertices.to(device),
                            self.indices.to(device),
                            self.uv_vts.to(device),
                            self.uv_faces.to(device),
                            self.tex.to(device),
                            device=device)
        mesh._edges = self._edges.to(device) if self._edges is not None else None
        mesh._connected_faces = self._connected_faces.to(device) if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.to(device) if self._laplacian is not None else None
        return mesh

    def detach(self):
        mesh = TexturedMesh(self.vertices.detach(),
                            self.indices.detach(),
                            self.uv_vts.detach(),
                            self.uv_faces.detach(),
                            self.tex.detach(),
                            device=self.device)
        mesh.face_normals = self.face_normals.detach()
        mesh.vertex_normals = self.vertex_normals.detach()
        mesh._edges = self._edges.detach() if self._edges is not None else None
        mesh._connected_faces = self._connected_faces.detach() if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.detach() if self._laplacian is not None else None
        return mesh

    def with_vertices(self, vertices):
        """ Create a mesh with the same connectivity but with different vertex positions

        Args:
            vertices (tensor): New vertex positions (Vx3)
        """

        assert len(vertices) == len(self.vertices)

        mesh_new = TexturedMesh(self.vertices,
                                self.indices,
                                self.uv_vts,
                                self.uv_faces,
                                self.tex,
                                device=self.device)

        mesh_new._edges = self._edges
        mesh_new._connected_faces = self._connected_faces
        mesh_new._laplacian = self._laplacian
        return mesh_new

    def with_colors(self, colors):
        """ Create a mesh with the same connectivity but with different texture map

        Args:
            colors (tensor): New color values (HxWx3)
        """

        # assert len(colors) == len(self.colors)
        mesh_new = TexturedMesh(self.vertices,
                                self.indices,
                                self.uv_vts,
                                self.uv_faces,
                                colors,
                                device=self.device)

        # mesh_new = TexturedMesh(self.vertices, self.indices, colors, self.device)
        mesh_new._edges = self._edges
        mesh_new._connected_faces = self._connected_faces
        mesh_new._laplacian = self._laplacian
        return mesh_new

    @property
    def edges(self):
        if self._edges is None:
            from diff_renderer.normal_nds.nds.utils.geometry import find_edges
            self._edges = find_edges(self.indices)
        return self._edges

    @property
    def connected_faces(self):
        if self._connected_faces is None:
            from diff_renderer.normal_nds.nds.utils.geometry import find_connected_faces
            self._connected_faces = find_connected_faces(self.indices)
        return self._connected_faces

    @property
    def laplacian(self):
        if self._laplacian is None:
            from diff_renderer.normal_nds.nds.utils.geometry import compute_laplacian_uniform
            self._laplacian = compute_laplacian_uniform(self)
        return self._laplacian

    def compute_connectivity(self):
        self._edges = self.edges
        self._connected_faces = self.connected_faces
        self._laplacian = self.laplacian

    def compute_normals(self):
        # Compute the face normals
        a = self.vertices[self.indices][:, 0, :]
        b = self.vertices[self.indices][:, 1, :]
        c = self.vertices[self.indices][:, 2, :]
        self.face_normals = torch.nn.functional.normalize(torch.cross(b - a, c - a), p=2, dim=-1)

        # Compute the vertex normals
        vertex_normals = torch.zeros_like(self.vertices)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 0], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 1], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 2], self.face_normals)
        self.vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1)

    # @profile
    def connected_faces_with_mask(self, mask):
        masked_vertex_indices = mask.nonzero(as_tuple=True)[0]
        num_faces = self.indices.shape[0]
        is_vertex_masked = torch.isin(self.indices.reshape(-1), masked_vertex_indices).reshape([num_faces, -1])
        self.faces_with_mask = self.indices[is_vertex_masked.prod(axis=1).nonzero(as_tuple=True)[0]]
        from nds.utils.geometry import find_connected_faces
        connected_faces_with_mask_ = find_connected_faces(self.faces_with_mask)
        return connected_faces_with_mask_

    def compute_normals_with_mask(self):
        # Compute the face normals
        vertices = self.vertices[self.faces_with_mask]
        a = vertices[:, 0, :]
        b = vertices[:, 1, :]
        c = vertices[:, 2, :]
        return torch.nn.functional.normalize(torch.cross(b - a, c - a), p=2, dim=-1)
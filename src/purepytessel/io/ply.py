import numpy as np
from purepytessel.meshing import Mesh


def write_ply(filename: str, mesh: Mesh) -> None:
    """
    Write a triangle mesh to a binary little-endian PLY file.

    Parameters
    ----------
    filename : str
        Output filename (.ply)
    mesh : Mesh
        Mesh with vertices (M,3), normals (M,3), indices (N,3)
    """
    vertices = mesh.vertices
    normals = mesh.normals
    faces = mesh.indices

    assert vertices.shape == normals.shape
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3

    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]

    with open(filename, "wb") as f:
        # ------------------------------------------------------------------
        # Header
        # ------------------------------------------------------------------
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n_vertices}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            f"element face {n_faces}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))

        # ------------------------------------------------------------------
        # Vertex data
        # ------------------------------------------------------------------
        vertex_data = np.empty(
            n_vertices,
            dtype=[
                ("x", "f4"), ("y", "f4"), ("z", "f4"),
                ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ],
        )

        vertex_data["x"] = vertices[:, 0]
        vertex_data["y"] = vertices[:, 1]
        vertex_data["z"] = vertices[:, 2]
        vertex_data["nx"] = normals[:, 0]
        vertex_data["ny"] = normals[:, 1]
        vertex_data["nz"] = normals[:, 2]

        vertex_data.tofile(f)

        # ------------------------------------------------------------------
        # Face data
        # ------------------------------------------------------------------
        face_data = np.empty(
            n_faces,
            dtype=[
                ("n", "u1"),      # number of vertices (always 3)
                ("v0", "i4"),
                ("v1", "i4"),
                ("v2", "i4"),
            ],
        )

        face_data["n"] = 3
        face_data["v0"] = faces[:, 0]
        face_data["v1"] = faces[:, 1]
        face_data["v2"] = faces[:, 2]

        face_data.tofile(f)
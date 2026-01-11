import numpy as np

from purepytessel.containers.mesh import Mesh


# -----------------------------------------------------------------------------
# Coordinate transforms
# -----------------------------------------------------------------------------

def grid_to_fractional(
    points: np.ndarray,
    dims: tuple[int, int, int],
) -> np.ndarray:
    """
    Convert grid-space coordinates to fractional coordinates.

    points : (..., 3)   grid coordinates (x, y, z)
    dims   : (nz, ny, nx)

    returns: (..., 3)   fractional coordinates in [0, 1]
    """
    nz, ny, nx = dims

    scale = np.array(
        [1.0 / nx, 1.0 / ny, 1.0 / nz],
        dtype=np.float32,
    )

    return points * scale


def fractional_to_world(
    points_frac: np.ndarray,
    unitcell: np.ndarray,
) -> np.ndarray:
    """
    Convert fractional coordinates to world coordinates.

    points_frac : (..., 3)
    unitcell    : (3, 3), rows are basis vectors

    returns     : (..., 3)
    """
    return points_frac @ unitcell


def normals_grid_to_world(
    normals_grid: np.ndarray,
    unitcell: np.ndarray,
) -> np.ndarray:
    """
    Transform normals from grid space to world space.

    Uses inverse-transpose of the unit cell.

    normals_grid : (..., 3)
    unitcell     : (3, 3)
    """
    invT = np.linalg.inv(unitcell).T
    normals_w = normals_grid @ invT

    # renormalize
    n = np.linalg.norm(normals_w, axis=-1, keepdims=True)
    n[n == 0.0] = 1.0

    return normals_w / n

def deduplicate_vertices(
    vertices: np.ndarray,
    normals: np.ndarray,
    indices: np.ndarray,
    tol: float = 1e-6,
):
    """
    Deduplicate vertices with tolerance-based hashing.

    vertices : (M,3)
    normals  : (M,3)
    indices  : (N,3)

    Returns:
        vertices_u : (K,3)
        normals_u  : (K,3)
        indices_u  : (N,3)
    """
    assert vertices.shape == normals.shape

    # Quantize vertices
    scale = 1.0 / tol
    verts_q = np.round(vertices * scale).astype(np.int64)

    vertex_map = {}        # quantized tuple -> new index
    new_vertices = []
    new_normals = []
    new_indices = np.empty_like(indices)

    for tri_idx, tri in enumerate(indices):
        for j, old_idx in enumerate(tri):
            key = tuple(verts_q[old_idx])

            if key in vertex_map:
                new_idx = vertex_map[key]
                new_normals[new_idx] += normals[old_idx]
            else:
                new_idx = len(new_vertices)
                vertex_map[key] = new_idx
                new_vertices.append(vertices[old_idx])
                new_normals.append(normals[old_idx].copy())

            new_indices[tri_idx, j] = new_idx

    # Normalize averaged normals
    new_vertices = np.asarray(new_vertices, dtype=np.float32)
    new_normals = np.asarray(new_normals, dtype=np.float32)

    n = np.linalg.norm(new_normals, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    new_normals /= n

    return new_vertices, new_normals, new_indices

# -----------------------------------------------------------------------------
# Mesh assembly
# -----------------------------------------------------------------------------

def triangles_to_mesh(
    field,
    triangles_grid: np.ndarray,
    normals_grid: np.ndarray,
) -> Mesh:
    """
    Assemble a Mesh from grid-space triangles and vertex normals.

    triangles_grid : (N, 3, 3)  grid-space vertex positions
    normals_grid   : (N, 3, 3)  grid-space vertex normals
    """
    assert triangles_grid.shape == normals_grid.shape
    assert triangles_grid.ndim == 3

    unitcell = field.unitcell
    dims = field.dimensions  # (nz, ny, nx)

    # ------------------------------------------------------------------
    # Vertices
    # ------------------------------------------------------------------
    verts_grid = triangles_grid.reshape(-1, 3)

    verts_frac = grid_to_fractional(verts_grid, dims)
    verts_frac -= np.array([0.5, 0.5, 0.5], dtype=np.float32)
    verts_world = fractional_to_world(verts_frac, unitcell)

    # ------------------------------------------------------------------
    # Normals
    # ------------------------------------------------------------------
    norms_grid = normals_grid.reshape(-1, 3)
    norms_world = normals_grid_to_world(norms_grid, unitcell)

    # ------------------------------------------------------------------
    # Indices
    # ------------------------------------------------------------------
    n_tris = triangles_grid.shape[0]
    indices = np.arange(3 * n_tris, dtype=np.int32).reshape(-1, 3)

    return Mesh(
        vertices=verts_world.astype(np.float32),
        normals=norms_world.astype(np.float32),
        indices=indices,
    )

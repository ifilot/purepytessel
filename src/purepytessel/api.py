import numpy as np

from purepytessel.containers.scalar_field import ScalarField
from purepytessel.algos.marching_cubes import marching_cubes
from purepytessel.meshing import triangles_to_mesh
from purepytessel.io.ply import write_ply


def isosurface_to_ply(
    values: np.ndarray,
    unitcell: np.ndarray,
    isovalue: float,
    filename: str,
) -> None:
    """
    Extract an isosurface from a 3D scalar field and write it to a PLY file.

    Parameters
    ----------
    values : ndarray (nz, ny, nx)
        Scalar field values in (z, y, x) order.
    unitcell : ndarray (3,3)
        Unit cell matrix (rows = basis vectors).
    isovalue : float
        Isosurface value.
    filename : str
        Output .ply filename.
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if values.ndim != 3:
        raise ValueError("values must be a 3D numpy array (nz, ny, nx)")

    unitcell = np.asarray(unitcell, dtype=np.float32)
    if unitcell.shape != (3, 3):
        raise ValueError("unitcell must be shape (3,3)")

    values = np.asarray(values, dtype=np.float32)

    nz, ny, nx = values.shape

    # ------------------------------------------------------------------
    # Build scalar field container
    # ------------------------------------------------------------------
    field = ScalarField(
        values=values,
        dimensions=(nz, ny, nx),
        unitcell=unitcell,
    )

    # ------------------------------------------------------------------
    # Marching cubes (grid space)
    # ------------------------------------------------------------------
    triangles_grid, normals_grid = marching_cubes(field, isovalue)

    if triangles_grid.shape[0] == 0:
        raise RuntimeError("No triangles generated (isosurface empty?)")

    # ------------------------------------------------------------------
    # Mesh assembly (world space)
    # ------------------------------------------------------------------
    mesh = triangles_to_mesh(field, triangles_grid, normals_grid)

    # ------------------------------------------------------------------
    # Write PLY
    # ------------------------------------------------------------------
    write_ply(filename, mesh)

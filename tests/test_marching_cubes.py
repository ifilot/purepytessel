import numpy as np

from purepytessel.algos.marching_cubes import marching_cubes
from purepytessel.containers.scalar_field import ScalarField
from purepytessel.meshing import triangles_to_mesh


def test_marching_cubes_plane_x():
    """
    Test marching cubes on a simple scalar field f(x,y,z) = x.
    The isosurface at x = 1.5 should be a plane (in grid coords).
    Data storage is (z, y, x).
    """
    nx, ny, nz = 4, 4, 4

    # Build values in z,y,x order so reshape to (nz,ny,nx) is consistent.
    values = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                values.append(float(i))  # f = x

    field = ScalarField(
        values=values,
        dimensions=(nz, ny, nx),   # IMPORTANT for (z,y,x)
        unitcell=np.identity(3)
    )

    isovalue = 1.5
    tris, norms = marching_cubes(field, isovalue)
    mesh = triangles_to_mesh(field, tris, norms)

    # geometry
    assert np.allclose(mesh.vertices[:,0], 1.5)

    # normals
    expected = np.array([1.0, 0.0, 0.0])
    assert np.allclose(mesh.normals, expected, atol=1e-6)
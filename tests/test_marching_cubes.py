import numpy as np

from purepytessel.algos.marching_cubes import marching_cubes
from purepytessel.containers.scalar_field import ScalarField
from purepytessel.meshing import triangles_to_mesh


def test_marching_cubes_plane_x_periodic():
    """
    Periodic boundary condition test.

    Scalar field: f(x, y, z) = x on a 4×4×4 grid.
    The isosurface at x = 1.5 should form a plane at

        x_world = (1.5 / nx) - 0.5 = -0.125
    """
    nx = ny = nz = 4
    isovalue = 1.5

    # Build scalar field values (flattened, z-major order)
    values = [float(i) for k in range(nz) for j in range(ny) for i in range(nx)]

    field = ScalarField(
        values=values,
        dimensions=(nz, ny, nx),
        unitcell=np.identity(3, dtype=np.float32),
    )

    triangles, normals = marching_cubes(field, isovalue)
    mesh = triangles_to_mesh(field, triangles, normals)

    # --------------------------------------------------
    # Geometry: plane position
    # --------------------------------------------------
    x_expected = (isovalue / nx) - 0.5
    xs = mesh.vertices[:, 0]

    assert np.allclose(xs, x_expected, atol=1e-6)

    # --------------------------------------------------
    # Geometry: plane flatness
    # --------------------------------------------------
    assert np.ptp(xs) < 1e-6  # peak-to-peak range

    # --------------------------------------------------
    # Normals: ±x direction
    # --------------------------------------------------
    nxv, nyv, nzv = mesh.normals.T

    assert np.allclose(np.abs(nxv), 1.0, atol=1e-6)
    assert np.allclose(nyv, 0.0, atol=1e-6)
    assert np.allclose(nzv, 0.0, atol=1e-6)


def test_anisotropic_unitcell():
    """
    Verify that anisotropic unit cell scaling is respected
    in the resulting mesh geometry.
    """
    nx = ny = nz = 16
    values = np.random.rand(nz, ny, nx).astype(np.float32)

    unitcell = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )

    field = ScalarField(values, (nz, ny, nx), unitcell)
    triangles, normals = marching_cubes(field, 0.5)
    mesh = triangles_to_mesh(field, triangles, normals)

    # Bounding box extents should reflect unitcell anisotropy
    bbox = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)

    assert bbox[0] > bbox[1] > bbox[2]


def test_sphere_radius():
    """
    Isosurface of a quadratic radial field should approximate
    a sphere of radius r.
    """
    nx = ny = nz = 32
    center = np.array([nx / 2, ny / 2, nz / 2], dtype=np.float32)
    radius = 8.0

    # Squared distance field
    z, y, x = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    values = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
    values = values.astype(np.float32)

    field = ScalarField(values, (nz, ny, nx), np.eye(3, dtype=np.float32))
    triangles, _ = marching_cubes(field, radius**2)

    points = triangles.reshape(-1, 3)
    distances = np.linalg.norm(points - center, axis=1)

    assert np.allclose(distances, radius, atol=0.75)


def test_plane_x():
    """
    Non-periodic plane test: f(x, y, z) = x.

    The isosurface should be a plane at x = iso
    with normals aligned to the ±x direction.
    """
    nx = ny = nz = 24
    iso = 11.5

    z, y, x = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    values = x.astype(np.float32)

    field = ScalarField(values, (nz, ny, nx), np.eye(3, dtype=np.float32))
    triangles, normals = marching_cubes(field, iso)

    # Geometry: all vertices lie on x = iso
    assert np.allclose(triangles[..., 0], iso, atol=1e-6)

    # Normals: ±x direction
    nxv, nyv, nzv = normals.T

    assert np.allclose(np.abs(nxv), 1.0, atol=1e-5)
    assert np.allclose(nyv, 0.0, atol=1e-6)
    assert np.allclose(nzv, 0.0, atol=1e-6)
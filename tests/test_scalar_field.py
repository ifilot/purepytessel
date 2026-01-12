import numpy as np

from purepytessel.containers.scalar_field import ScalarField


def test_scalarfield_shape_and_layout():
    """
    Verify that ScalarField reshapes a flat values array into
    (z, y, x) order with correct row-major (C-order) layout.
    """
    nx, ny, nz = 4, 3, 2
    values = np.arange(nx * ny * nz, dtype=np.float32)

    field = ScalarField(
        values=values,
        dimensions=(nz, ny, nx),
        unitcell=np.eye(3, dtype=np.float32),
    )

    arr = field.values

    # Shape must match (z, y, x)
    assert arr.shape == (nz, ny, nx)

    # x varies fastest
    assert arr[0, 0, 0] == 0
    assert arr[0, 0, 1] == 1

    # y increments by nx
    assert arr[0, 1, 0] == nx

    # z increments by nx * ny
    assert arr[1, 0, 0] == nx * ny

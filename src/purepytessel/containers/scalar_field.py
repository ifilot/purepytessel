import numpy as np


class ScalarField:
    """
    Scalar field on a periodic 3D grid.

    Storage order:
        values[k, j, i] == f(x=i, y=j, z=k)

    dimensions = (nz, ny, nx)
    """

    def __init__(self, values, dimensions, unitcell):
        self.dimensions = tuple(dimensions)  # (nz, ny, nx)
        nz, ny, nx = self.dimensions

        self.values = np.asarray(values, dtype=np.float32).reshape((nz, ny, nx))

        self.unitcell = np.asarray(unitcell, dtype=np.float32).reshape(3, 3)
        self.unitcell_inv = np.linalg.inv(self.unitcell)

    # ------------------------------------------------------------------
    # Grid ↔ fractional ↔ world
    # ------------------------------------------------------------------

    def grid_to_fractional(self, i, j, k):
        """
        Grid (x,y,z) → fractional coordinates, periodic.
        """
        nz, ny, nx = self.dimensions

        return np.array(
            [
                (i % nx) / nx,
                (j % ny) / ny,
                (k % nz) / nz,
            ],
            dtype=np.float32,
        )

    def fractional_to_grid(self, fx, fy, fz):
        """
        Fractional → grid coordinates (continuous).
        """
        nz, ny, nx = self.dimensions

        return np.array(
            [
                fx * nx,
                fy * ny,
                fz * nz,
            ],
            dtype=np.float32,
        )

    def grid_to_world(self, i, j, k):
        """
        Grid → world coordinates.
        """
        frac = self.grid_to_fractional(i, j, k)
        return frac @ self.unitcell

    def world_to_grid(self, x, y, z):
        """
        World → grid coordinates (continuous).
        """
        frac = self.unitcell_inv @ np.array([x, y, z], dtype=np.float32)
        return self.fractional_to_grid(frac[0], frac[1], frac[2])

    # ------------------------------------------------------------------
    # Periodic trilinear interpolation
    # ------------------------------------------------------------------

    def get_value_interp(self, x, y, z):
        """
        Trilinear interpolation in world space (periodic).
        """
        gx, gy, gz = self.world_to_grid(x, y, z)

        nz, ny, nx = self.dimensions

        i0 = int(np.floor(gx)) % nx
        j0 = int(np.floor(gy)) % ny
        k0 = int(np.floor(gz)) % nz

        i1 = (i0 + 1) % nx
        j1 = (j0 + 1) % ny
        k1 = (k0 + 1) % nz

        xd = gx - np.floor(gx)
        yd = gy - np.floor(gy)
        zd = gz - np.floor(gz)

        v = self.values

        c000 = v[k0, j0, i0]
        c100 = v[k0, j0, i1]
        c010 = v[k0, j1, i0]
        c110 = v[k0, j1, i1]
        c001 = v[k1, j0, i0]
        c101 = v[k1, j0, i1]
        c011 = v[k1, j1, i0]
        c111 = v[k1, j1, i1]

        return (
            c000 * (1 - xd) * (1 - yd) * (1 - zd) +
            c100 * xd       * (1 - yd) * (1 - zd) +
            c010 * (1 - xd) * yd       * (1 - zd) +
            c110 * xd       * yd       * (1 - zd) +
            c001 * (1 - xd) * (1 - yd) * zd +
            c101 * xd       * (1 - yd) * zd +
            c011 * (1 - xd) * yd       * zd +
            c111 * xd       * yd       * zd
        )

    # ------------------------------------------------------------------
    # Periodic gradients
    # ------------------------------------------------------------------

    def compute_gradients(self) -> np.ndarray:
        """
        Compute periodic central-difference gradients.

        Returns:
            grads : (nz, ny, nx, 3)  [dx, dy, dz]
        """
        v = self.values
        nz, ny, nx = self.dimensions

        # Periodic central differences
        dx = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) * 0.5
        dy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) * 0.5
        dz = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) * 0.5

        return np.stack((dx, dy, dz), axis=-1)
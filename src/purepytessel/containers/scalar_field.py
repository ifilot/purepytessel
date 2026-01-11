import numpy as np

class ScalarField:
    def __init__(self, values, dimensions, unitcell):
        self.values = np.asarray(values, dtype=float).reshape(dimensions)
        self.dimensions = dimensions
        self.unitcell = np.asarray(unitcell, dtype=float).reshape(3, 3)
        self.unitcell_inv = np.linalg.inv(self.unitcell)

    def grid_to_realspace(self, i, j, k):
        frac = np.array([i/self.dim[0], j/self.dim[1], k/self.dim[2]])
        r = self.unitcell.T @ frac
        return Vec3(*r)

    def realspace_to_grid(self, x, y, z):
        frac = self.unitcell_inv @ np.array([x, y, z])
        return Vec3(
            frac[0]*self.dim[0],
            frac[1]*self.dim[1],
            frac[2]*self.dim[2]
        )

    def get_value_interp(self, x, y, z):
        g = self.realspace_to_grid(x, y, z)
        i0, j0, k0 = np.floor([g.x, g.y, g.z]).astype(int)
        i1, j1, k1 = (i0+1) % self.dim[0], (j0+1) % self.dim[1], (k0+1) % self.dim[2]

        xd, yd, zd = g.x - i0, g.y - j0, g.z - k0

        c000 = self.values[i0, j0, k0]
        c100 = self.values[i1, j0, k0]
        c010 = self.values[i0, j1, k0]
        c110 = self.values[i1, j1, k0]
        c001 = self.values[i0, j0, k1]
        c101 = self.values[i1, j0, k1]
        c011 = self.values[i0, j1, k1]
        c111 = self.values[i1, j1, k1]

        return (
            c000*(1-xd)*(1-yd)*(1-zd) +
            c100*xd*(1-yd)*(1-zd) +
            c010*(1-xd)*yd*(1-zd) +
            c001*(1-xd)*(1-yd)*zd +
            c101*xd*(1-yd)*zd +
            c011*(1-xd)*yd*zd +
            c110*xd*yd*(1-zd) +
            c111*xd*yd*zd
        )

    def compute_gradients(self) -> np.ndarray:
        """
        Compute scalar-field gradients on the grid.

        vals : (nz, ny, nx)

        returns : (nz, ny, nx, 3)   [dx, dy, dz]
        """
        dz, dy, dx = np.gradient(self.values)
        return np.stack((dx, dy, dz), axis=-1)
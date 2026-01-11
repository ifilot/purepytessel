import numpy as np

from purepytessel.api import isosurface_to_ply

def gaussian_3d(nx, ny, nz, center, sigma):
    """
    Create a 3D Gaussian scalar field on a regular grid.

    Returns values shaped (nz, ny, nx).
    """
    cx, cy, cz = center

    x = np.arange(nx, dtype=np.float32)
    y = np.arange(ny, dtype=np.float32)
    z = np.arange(nz, dtype=np.float32)

    # meshgrid in (x, y, z), then transpose to (z, y, x)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    X = X.transpose(2, 1, 0)
    Y = Y.transpose(2, 1, 0)
    Z = Z.transpose(2, 1, 0)

    r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    return np.exp(-r2 / (2.0 * sigma ** 2))


def main():
    # ------------------------------------------------------------------
    # Grid parameters
    # ------------------------------------------------------------------
    nx = ny = nz = 64
    center = (nx / 2, ny / 2, nz / 2)
    sigma = 10.0

    # ------------------------------------------------------------------
    # Scalar field
    # ------------------------------------------------------------------
    values = gaussian_3d(nx, ny, nz, center, sigma)

    # ------------------------------------------------------------------
    # Isosurface extraction + export
    # ------------------------------------------------------------------
    isovalue = 0.1  # controls radius of the sphere

    isosurface_to_ply(
        values=values,
        unitcell=np.eye(3, dtype=np.float32),
        isovalue=isovalue,
        filename="gaussian_isosurface.ply",
    )

    print("Wrote gaussian_isosurface.ply")


if __name__ == "__main__":
    main()

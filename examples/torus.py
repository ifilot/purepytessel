import numpy as np
import time

from purepytessel.api import isosurface_to_ply


def torus_field(nx, ny, nz, R, r, noise=0.0):
    """
    Implicit torus scalar field.

    f(x,y,z) = (sqrt(x^2 + y^2) - R)^2 + z^2
    Isosurface at f = r^2

    Returns values shaped (nz, ny, nx).
    """
    x = np.linspace(-1.5, 1.5, nx, dtype=np.float32)
    y = np.linspace(-1.5, 1.5, ny, dtype=np.float32)
    z = np.linspace(-1.0, 1.0, nz, dtype=np.float32)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    X = X.transpose(2, 1, 0)
    Y = Y.transpose(2, 1, 0)
    Z = Z.transpose(2, 1, 0)

    rho = np.sqrt(X**2 + Y**2)
    f = (rho - R) ** 2 + Z ** 2

    if noise > 0.0:
        f += noise * np.random.randn(*f.shape).astype(np.float32)

    return f


def main():
    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    nx, ny, nz = 256, 256, 256

    # Torus parameters
    R = 0.8    # major radius
    r = 0.3    # minor radius

    # Scalar field
    values = torus_field(nx, ny, nz, R, r, noise=0.01)

    # ------------------------------------------------------------------
    # Anisotropic + non-orthogonal unit cell
    # ------------------------------------------------------------------
    unitcell = np.array(
        [
            [0.04, 0.00, 0.00],
            [0.01, 0.05, 0.00],
            [0.00, 0.00, 0.06],
        ],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------
    # Isosurface extraction + export
    # ------------------------------------------------------------------
    st = time.perf_counter()
    isosurface_to_ply(
        values=values,
        unitcell=unitcell,
        isovalue=r ** 2,
        filename="torus_anisotropic_noisy.ply",
    )

    print("Wrote torus_anisotropic_noisy.ply")
    print("Time: %.2f s" % (time.perf_counter() - st))


if __name__ == "__main__":
    main()

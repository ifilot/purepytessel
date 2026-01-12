import numpy as np
from purepytessel.containers.scalar_field import ScalarField

def test_gradient_linear_x():
    nx = ny = nz = 16
    vals = np.zeros((nz, ny, nx), dtype=np.float32)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                vals[k, j, i] = i

    field = ScalarField(vals, (nz, ny, nx), np.eye(3))
    grads = field.compute_gradients()

    gx = grads[..., 0]
    gy = grads[..., 1]
    gz = grads[..., 2]

    # interior points
    assert np.allclose(gx[1:-1, 1:-1, 1:-1], 1.0)
    assert np.allclose(gy, 0.0)
    assert np.allclose(gz, 0.0)

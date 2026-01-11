from purepytessel.containers.scalar_field import ScalarField
from purepytessel.algos.tables import (
    EDGE_TABLE,
    TRIANGLE_TABLE,
    VERTEX_OFFSETS,
    EDGE_VERTICES,
)

import numpy as np
from numba import njit, prange

PRECISION = 1e-8


# -----------------------------------------------------------------------------
# Phase 1: count triangles per cube
# -----------------------------------------------------------------------------

@njit(cache=True)
def _count_triangles(vals, isovalue):
    nz, ny, nx = vals.shape
    counts = np.zeros((nz - 1, ny - 1, nx - 1), dtype=np.int32)

    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):

                cubidx = (
                    (vals[k, j, i]       < isovalue) << 0 |
                    (vals[k, j + 1, i]   < isovalue) << 1 |
                    (vals[k, j + 1, i + 1] < isovalue) << 2 |
                    (vals[k, j, i + 1]   < isovalue) << 3 |
                    (vals[k + 1, j, i]   < isovalue) << 4 |
                    (vals[k + 1, j + 1, i] < isovalue) << 5 |
                    (vals[k + 1, j + 1, i + 1] < isovalue) << 6 |
                    (vals[k + 1, j, i + 1] < isovalue) << 7
                )

                if cubidx == 0 or cubidx == 255:
                    continue

                tri_edges = TRIANGLE_TABLE[cubidx]
                ntri = 0
                for t in range(0, 16, 3):
                    if tri_edges[t] == -1:
                        break
                    ntri += 1

                counts[k, j, i] = ntri

    return counts


def _prefix_sum(counts):
    flat = counts.ravel()
    offsets = np.empty_like(flat)
    total = 0
    for i in range(flat.size):
        offsets[i] = total
        total += flat[i]
    return offsets, total


# -----------------------------------------------------------------------------
# Phase 2: generate triangles + normals
# -----------------------------------------------------------------------------

@njit(parallel=True, cache=True, fastmath=True)
def _generate_triangles(
    vals,
    grads,
    isovalue,
    offsets,
    triangles,
    normals,
):
    nz, ny, nx = vals.shape
    ny1 = ny - 1
    nx1 = nx - 1

    for idx in prange(offsets.size):
        base = offsets[idx]

        # decode flat index → (k, j, i)
        k = idx // (ny1 * nx1)
        rem = idx % (ny1 * nx1)
        j = rem // nx1
        i = rem % nx1

        cubidx = (
            (vals[k, j, i]       < isovalue) << 0 |
            (vals[k, j + 1, i]   < isovalue) << 1 |
            (vals[k, j + 1, i + 1] < isovalue) << 2 |
            (vals[k, j, i + 1]   < isovalue) << 3 |
            (vals[k + 1, j, i]   < isovalue) << 4 |
            (vals[k + 1, j + 1, i] < isovalue) << 5 |
            (vals[k + 1, j + 1, i + 1] < isovalue) << 6 |
            (vals[k + 1, j, i + 1] < isovalue) << 7
        )

        if cubidx == 0 or cubidx == 255:
            continue

        edge_mask = EDGE_TABLE[cubidx]

        edge_pos = np.zeros((12, 3), dtype=np.float32)
        edge_nrm = np.zeros((12, 3), dtype=np.float32)

        # interpolate edges
        for e in range(12):
            if not (edge_mask & (1 << e)):
                continue

            v0, v1 = EDGE_VERTICES[e]
            dx0, dy0, dz0 = VERTEX_OFFSETS[v0]
            dx1, dy1, dz1 = VERTEX_OFFSETS[v1]

            val0 = vals[k + dz0, j + dy0, i + dx0]
            val1 = vals[k + dz1, j + dy1, i + dx1]

            p0x = i + dx0
            p0y = j + dy0
            p0z = k + dz0

            p1x = i + dx1
            p1y = j + dy1
            p1z = k + dz1

            g0 = grads[k + dz0, j + dy0, i + dx0]
            g1 = grads[k + dz1, j + dy1, i + dx1]

            if abs(val1 - val0) < PRECISION:
                mu = 0.0
            else:
                mu = (isovalue - val0) / (val1 - val0)

            # position
            px = p0x + mu * (p1x - p0x)
            py = p0y + mu * (p1y - p0y)
            pz = p0z + mu * (p1z - p0z)

            edge_pos[e, 0] = px
            edge_pos[e, 1] = py
            edge_pos[e, 2] = pz

            # normal
            nx = g0[0] + mu * (g1[0] - g0[0])
            ny = g0[1] + mu * (g1[1] - g0[1])
            nz_ = g0[2] + mu * (g1[2] - g0[2])

            nrm = np.sqrt(nx * nx + ny * ny + nz_ * nz_)
            if nrm > 0.0:
                nx /= nrm
                ny /= nrm
                nz_ /= nrm

            edge_nrm[e, 0] = nx
            edge_nrm[e, 1] = ny
            edge_nrm[e, 2] = nz_

        tri_edges = TRIANGLE_TABLE[cubidx]
        tcount = 0

        for t in range(0, 16, 3):
            if tri_edges[t] == -1:
                break

            out = base + tcount

            e0 = tri_edges[t]
            e1 = tri_edges[t + 1]
            e2 = tri_edges[t + 2]

            triangles[out, 0] = edge_pos[e0]
            triangles[out, 1] = edge_pos[e1]
            triangles[out, 2] = edge_pos[e2]

            normals[out, 0] = edge_nrm[e0]
            normals[out, 1] = edge_nrm[e1]
            normals[out, 2] = edge_nrm[e2]

            tcount += 1


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def marching_cubes(
    field: ScalarField,
    isovalue: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated marching cubes.

    Returns:
        triangles_grid : (N, 3, 3)
        normals_grid   : (N, 3, 3)
    """
    vals = field.values
    grads = field.compute_gradients()

    counts = _count_triangles(vals, isovalue)
    offsets, total = _prefix_sum(counts)

    if total == 0:
        empty = np.empty((0, 3, 3), dtype=np.float32)
        return empty, empty

    triangles = np.empty((total, 3, 3), dtype=np.float32)
    normals = np.empty((total, 3, 3), dtype=np.float32)

    _generate_triangles(
        vals,
        grads,
        isovalue,
        offsets,
        triangles,
        normals,
    )

    return triangles, normals

from dataclasses import dataclass
import numpy as np

@dataclass
class Mesh:
    vertices: np.ndarray  # (M,3)
    normals: np.ndarray   # (M,3)
    indices: np.ndarray   # (N,3)

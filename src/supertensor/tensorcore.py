from numba.experimental import jitclass
from numba import types
import numpy as np


spec = [
    ("data", types.float64[:]),
    ("grad", types.float64[:]),
]


@jitclass(spec)
class TensorCore:

    data: np.ndarray
    grad: np.ndarray

    def __init__(self, data: np.ndarray | float | int):
        if isinstance(data, (float, int)):
            data = np.array([data], dtype=np.float64)
        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float64)

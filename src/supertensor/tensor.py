import logging
from typing import Union
import numpy as np

from .mathfuncs import add, do_math, multiply

from .oprecords import AdditionOpRecord, OpRecord, ScalarMulOpRecord
from .tensorcore import TensorCore

TensorOrNumeric = Union[float, int, "Tensor"]


class Tensor:
    _tensor_data: TensorCore
    _history: list[OpRecord]
    track_grad: bool
    gpu_enabled: bool

    def __init__(self, data, track_grad: bool = False, enable_gpu: bool = False):
        self._tensor_data = TensorCore(data)
        self._history = []
        self.track_grad = track_grad
        self.gpu_enabled = enable_gpu

    def backward(self, prior_gradient: np.ndarray | None = None):
        gradient = prior_gradient
        for op in reversed(self._history):
            gradient = op.backward(self._tensor_data, gradient)

    def _perform_op(
        self,
        x: TensorOrNumeric,
        op_type: OpRecord,
        math_func: callable,
        op_kwargs: dict = dict(),
    ):
        if isinstance(x, Tensor):
            x = x.data
        new_tensor = Tensor(
            do_math(math_func, use_gpu=self.gpu_enabled)(
                self._tensor_data.data, second=x
            ),
            track_grad=self.track_grad,
            enable_gpu=self.gpu_enabled,
        )
        if self.track_grad:
            op = op_type(self._tensor_data, **op_kwargs)
            new_tensor._history.append(op)
        return new_tensor

    def __add__(self, x: TensorOrNumeric) -> "Tensor":
        return self._perform_op(x, AdditionOpRecord, add)

    def __mul__(self, x: TensorOrNumeric) -> "Tensor":
        return self._perform_op(x, ScalarMulOpRecord, multiply, {"scalar": x})

    @property
    def data(self):
        return self._tensor_data.data

    @property
    def grad(self):
        return self._tensor_data.grad

    def reset_grad(self):
        self._tensor_data.grad = np.zeros_like(self._tensor_data.data)

    def __str__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

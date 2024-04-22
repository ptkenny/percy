import logging
import numpy as np

from .tensorcore import TensorCore


def requires_prior_gradient(func: callable):
    def wrapper(self, tensor: TensorCore, prior_gradient: np.ndarray | None = None):
        if prior_gradient is None:
            if self.input_tensor.data.ndim != 0:
                prior_gradient = np.ones_like(self.input_tensor.data)
            else:
                prior_gradient = np.array([1.0])
        return func(self, tensor, prior_gradient)

    return wrapper


class OpRecord:
    input_tensor: TensorCore

    def __init__(self):
        raise NotImplementedError

    def backward(self, tensor: TensorCore, gradient_weights: np.ndarray | None = None):
        """
        A mutating function that computes the gradient of the input tensor with respect to the
        operation that was performed on it. This function should be implemented by each operation
        record.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


class AdditionOpRecord(OpRecord):

    def __init__(self, input_tensor: TensorCore):
        self.input_tensor = input_tensor

    @requires_prior_gradient
    def backward(
        self, tensor: TensorCore, prior_gradient: np.ndarray | None = None
    ) -> np.ndarray:
        tensor.grad += prior_gradient
        return tensor.grad

    def __str__(self) -> str:
        return f"AddOpRecord(input_tensor={self.input_tensor})"


class ScalarMulOpRecord(OpRecord):
    scalar: float

    def __init__(self, input_tensor: TensorCore, scalar: float):
        self.input_tensor = input_tensor
        self.scalar = scalar

    @requires_prior_gradient
    def backward(
        self, tensor: TensorCore, prior_gradient: np.ndarray | None = None
    ) -> np.ndarray:
        tensor.grad += prior_gradient * self.scalar
        return tensor.grad

    def __str__(self) -> str:
        return (
            f"ScalarMulOpRecord(input_tensor={self.input_tensor}, scalar={self.scalar})"
        )

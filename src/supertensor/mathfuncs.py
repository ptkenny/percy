"""
Container for all JIT compilable math functions. All functions in this module
must be JIT compilable, and will run either on the GPU or CPU.
"""

from dataclasses import dataclass
import logging
import os
import numpy as np

from numba import cuda, jit

_func_cache = dict()
_blocks_per_grid = os.environ.get("BLOCKS_PER_GRID", 128)
_threads_per_block = os.environ.get("THREADS_PER_BLOCK", 128)


@dataclass
class MathFunc:
    gpu: callable
    cpu: callable


def _get_func(func) -> callable:
    # Might need global _func_cache?
    if func not in _func_cache:
        _func_cache[func] = MathFunc(
            gpu=cuda.jit(func),
            cpu=jit(func),
        )
    return _func_cache[func]


def do_math(func: callable, use_gpu: bool = False) -> np.ndarray:
    func = _get_func(func)
    if use_gpu:
        return func.gpu[_blocks_per_grid, _threads_per_block]()
    return func.cpu


def add(first: np.ndarray, second: np.ndarray | None = None) -> np.ndarray:
    return first + second


def multiply(first: np.ndarray, second: np.ndarray | None = None) -> np.ndarray:
    return first * second

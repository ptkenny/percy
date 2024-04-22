import logging
import time

import numpy as np
import pytest
from ..supertensor.tensor import Tensor


@pytest.fixture
def logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger()


def test_tensor(logger):
    t1 = Tensor(np.array([1.0, 2.0]), track_grad=True)
    t2 = t1 + 2.0
    assert t2.data.tolist() == [3.0, 4.0], "scalar addition failed"
    t2.backward()
    assert t2.grad.tolist() == [1.0, 1.0], "gradient calculation failed"
    t2.reset_grad()
    assert t2.grad.tolist() == [0.0, 0.0], "gradient reset failed"
    t3 = t2 + t1
    assert t3.data.tolist() == [4.0, 6.0], "tensor addition failed"
    t3.backward()
    assert t3.grad.tolist() == [1.0, 1.0], "gradient calculation failed"
    t4 = t3 * 2.0
    assert t4.data.tolist() == [8.0, 12.0], "scalar multiplication failed"
    t4.backward()
    assert t4.grad.tolist() == [2.0, 2.0], "gradient calculation failed"


def test_stress(logger):
    total_ops = 1000000
    operands = np.random.rand(total_ops)
    tensors = [Tensor(operand) for operand in operands]
    st = time.time()
    # for i, operand in enumerate(operands):
    #     if i % 2 == 0:
    #         tensors[i] = tensors[i] + operand
    #     else:
    #         tensors[i] = tensors[i] * operand
    logger.info(f"time taken for {total_ops} ops(no gpu): {time.time() - st:.4f}s")

    tensors = [Tensor(operand, enable_gpu=True) for operand in operands]
    gpu_st = time.time()
    for i, operand in enumerate(operands):
        if i % 2 == 0:
            tensors[i] = tensors[i] + operand
        else:
            tensors[i] = tensors[i] * operand
    logger.info(f"time taken for {total_ops} ops(gpu): {time.time() - gpu_st:.4f}s")

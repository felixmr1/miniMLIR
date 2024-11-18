import time
from mlir.ir import (
    Context,
    Location,
)
import cupy as cp
from utils.utils import calculate_grid_size, extract_gpu_code
from mlir_modules.matmul import create_mlir_matmul_module
from mlir_passmanager.pipeline import lower
import numpy as np


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    with Context() as ctx:
        with Location.unknown(ctx):
            a_rows, a_cols = A.shape
            b_rows, b_cols = B.shape

            # Verify matrix dimensions
            assert a_cols == b_rows, "Invalid matrix dimensions for multiplication"

            blocks_size = (16, 16)
            grids_size = calculate_grid_size((a_rows, b_cols), blocks_size)

            print(f"Matrix dimensions: {a_rows}x{a_cols}")
            print(f"Using grid={grids_size}, block={blocks_size}")

            # Create result matrix
            C = np.zeros((a_rows, b_cols), dtype=np.float32)

            # Create and load module
            module = create_mlir_matmul_module(a_rows, b_cols, blocks_size, grids_size)
            lowered_module = lower(module)
            ptx = extract_gpu_code(lowered_module)

            # Load kernel
            mod = cp.cuda.Module()
            mod.load(ptx)
            kernel = mod.get_function("matmul_kernel")

            # Transfer data to GPU
            a_gpu = cp.asarray(A)
            b_gpu = cp.asarray(B)
            c_gpu = cp.asarray(C)

            try:
                # Launch kernel with calculated dimensions
                kernel(
                    grids_size,
                    blocks_size,
                    (a_gpu.data.ptr, b_gpu.data.ptr, c_gpu.data.ptr),
                )
                cp.cuda.Stream.null.synchronize()

                # Get result
                result = cp.asnumpy(c_gpu)

                # Verify a sample calculation
                sample_i, sample_j = 0, 0
                expected_val = np.dot(A[sample_i, :], B[:, sample_j])
                actual_val = result[sample_i, sample_j]
                print(f"Sample verification at ({sample_i},{sample_j}):")
                print(f"Expected: {expected_val}, Got: {actual_val}")

                return result

            finally:
                # Cleanup
                del a_gpu, b_gpu, c_gpu
                cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    # Test with different sizes
    for N in [10, 32, 64]:
        print(f"\nTesting {N}x{N} matrix multiplication:")
        A = np.random.randn(N, N).astype(np.float32)
        B = np.random.randn(N, N).astype(np.float32)

        # CPU reference
        start = time.time()
        expected = np.matmul(A, B)
        print(f"CPU TIME: {time.time() - start}")

        # GPU computation
        start = time.time()
        result = matmul(A, B)
        print(f"GPU TIME: {time.time() - start}")

        # Compare results
        if result is not None:
            max_diff = np.max(np.abs(result - expected))
            print(f"Maximum difference between CPU and GPU: {max_diff}")

            # Print more detailed error analysis if difference is large
            if max_diff > 1e-5:
                print("Large error detected. Additional diagnostics:")
                print(f"Mean absolute error: {np.mean(np.abs(result - expected))}")
                print(f"Relative error: {np.max(np.abs((result - expected)/expected))}")

                # Find location of maximum difference
                max_idx = np.unravel_index(
                    np.argmax(np.abs(result - expected)), result.shape
                )
                print(f"Maximum difference at position: {max_idx}")
                print(f"Expected value: {expected[max_idx]}")
                print(f"Got value: {result[max_idx]}")

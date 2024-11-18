from mlir.ir import Context, Module as MlirModule, Operation
from mlir.dialects.gpu import BinaryOp, ObjectAttr
from typing import Optional


def extract_gpu_code(module: MlirModule) -> Optional[bytes]:
    """Extract compiled GPU code (PTX/CUBIN) from MLIR module.

    Args:
        module: MLIR module containing gpu.binary operation

    Returns:
        Compiled GPU code as bytes, or None if no binary found
    """
    binary_op = find_gpu_binary(module)
    if not binary_op:
        return None

    # Get the last object from the binary (typically the compiled code)
    objects = [ObjectAttr(obj).object for obj in binary_op.objects]
    return objects[-1] if objects else None


def find_gpu_binary(module: MlirModule) -> Optional[Operation]:
    """Find the first gpu.binary operation in the module."""

    def is_gpu_binary(op: Operation) -> bool:
        return isinstance(op, BinaryOp)

    def walk_operations(op: Operation) -> Optional[Operation]:
        # Check if current op is a gpu.binary
        if is_gpu_binary(op):
            return op

        # Recursively check all nested operations
        for region in op.regions:
            for block in region.blocks:
                for nested_op in block.operations:
                    result = walk_operations(nested_op)
                    if result:
                        return result
        return None

    return walk_operations(module.operation)


def calculate_gflops(n, time_seconds):
    """
    For matrix multiplication, the number of operations is:
    - n*n multiplications and n*n additions for each of the n rows
    - Total = 2 * n^3 operations
    """
    operations = 2 * (n**3)  # Multiple-add for each element
    return (operations / 1e9) / time_seconds  # Convert to GFLOPS


def ptx_from_mlir_model(fp: str) -> Optional[bytes]:
    with open(fp, "r") as f:
        module = f.read()

    with Context() as context:
        parsed_module = MlirModule.parse(module)
        return extract_gpu_code(parsed_module)


def calculate_grid_size(matrix_size: tuple, block_size: tuple) -> tuple:
    grid_size_x = (matrix_size[0] + block_size[0] - 1) / block_size[0]
    grid_size_y = (matrix_size[1] + block_size[1] - 1) / block_size[1]
    return (int(grid_size_x), int(grid_size_y))

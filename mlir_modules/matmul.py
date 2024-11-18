from numpy import block
from mlir.ir import (
    Module,
    UnitAttr,
    TypeAttr,
    DictAttr,
    InsertionPoint,
    Block,
    StringAttr,
    F32Type,
    MemRefType,
    FunctionType,
    ArrayAttr,
    DenseI32ArrayAttr,
    IndexType,
    SymbolRefAttr,
    AffineMap,
    AffineMapAttr,
    AffineDimExpr,
    IntegerAttr,
    IntegerType,
)
from mlir.dialects import func, gpu, arith, linalg
from utils.utils import calculate_grid_size


# TODO: Can we keep boiler plate GPU static and just change the operation?
def create_mlir_matmul_module(
    rows: int,
    cols: int,
    block_size: tuple,
    grid_size: tuple,
) -> Module:
    module = Module.create()
    # Container module attribute
    module.operation.attributes["gpu.container_module"] = UnitAttr.get()

    # Create GPU module for device code
    with InsertionPoint(module.body):
        gpu_module = gpu.GPUModuleOp(sym_name="kernels")
        gpu_module_block = Block.create_at_start(gpu_module.regions[0], [])

        # Create kernel function inside GPU module
        with InsertionPoint(gpu_module_block):
            # TODO: Dynamic data types.
            f32 = F32Type.get()
            memref_t = MemRefType.get([rows, cols], f32)
            func_t = FunctionType.get([memref_t, memref_t, memref_t], [])
            type_attr = TypeAttr.get(func_t)

            kernel_func = gpu.GPUFuncOp(
                function_type=type_attr,
                arg_attrs=ArrayAttr.get([DictAttr.get({})] * 3),
            )

            kernel_func.attributes["sym_name"] = StringAttr.get("matmul_kernel")
            kernel_func.attributes["gpu.kernel"] = UnitAttr.get()

            # TODO: Dynamic block sizes?
            kernel_func.attributes["gpu.known_block_size"] = DenseI32ArrayAttr.get(
                [block_size[0], block_size[1], 1]
            )

            # Add shared memory allocation attributes
            shared_mem_size = (
                2 * block_size[0] * block_size[0] * 4
            )  # 2 tiles, 4 bytes per float
            kernel_func.attributes["gpu.shared_memory_size"] = IntegerAttr.get(
                IntegerType.get_unsigned(32), shared_mem_size
            )

            # Create kernel block with arguments
            kernel_func_block = Block.create_at_start(
                kernel_func.regions[0], [memref_t, memref_t, memref_t]
            )

            # Create kernel block
            with InsertionPoint(kernel_func_block):
                affine_map_a = AffineMap.get(
                    3, 0, [AffineDimExpr.get(0), AffineDimExpr.get(2)]
                )
                affine_map_b = AffineMap.get(
                    3, 0, [AffineDimExpr.get(2), AffineDimExpr.get(1)]
                )
                affine_map_c = AffineMap.get(
                    3, 0, [AffineDimExpr.get(0), AffineDimExpr.get(1)]
                )

                indexing_maps = ArrayAttr.get(
                    [
                        AffineMapAttr.get(affine_map_a),
                        AffineMapAttr.get(affine_map_b),
                        AffineMapAttr.get(affine_map_c),
                    ]
                )

                matmul_op = linalg.MatmulOp(
                    result_tensors=[],  # Empty since we're using memrefs directly
                    inputs=[
                        kernel_func_block.arguments[0],  # A matrix
                        kernel_func_block.arguments[1],  # B matrix
                    ],
                    outputs=[kernel_func_block.arguments[2]],  # C matrix
                    indexing_maps=indexing_maps,
                )
                # Create block for matmul computation
                matmul_block = Block.create_at_start(
                    matmul_op.regions[0], [f32, f32, f32]
                )

                # Add computation in the block
                with InsertionPoint(matmul_block):
                    # a * b
                    mul = arith.MulFOp(
                        matmul_block.arguments[0], matmul_block.arguments[1]
                    )
                    # c + (a * b)
                    add = arith.AddFOp(matmul_block.arguments[2], mul.result)
                    # Yield the result
                    linalg.YieldOp([add.result])

                gpu.ReturnOp([])

    # Create host function
    with InsertionPoint(module.body):
        host_func = func.FuncOp("launcher", func_t)

        # Create host block with arguments
        host_block = Block.create_at_start(
            host_func.regions[0], [memref_t, memref_t, memref_t]
        )

        grid_size = calculate_grid_size(matrix_size=(rows, cols), block_size=block_size)
        with InsertionPoint(host_block):
            c1 = arith.ConstantOp(IndexType.get(), 1)
            grid_size_x = arith.ConstantOp(IndexType.get(), grid_size[0])
            grid_size_y = arith.ConstantOp(IndexType.get(), grid_size[1])
            block_size_x = arith.ConstantOp(IndexType.get(), grid_size[0])
            block_size_y = arith.ConstantOp(IndexType.get(), grid_size[1])

            # Create a nested symbol reference
            kernel_ref = SymbolRefAttr.get(
                symbols=[
                    "kernels",  # Module name
                    "matmul_kernel",  # Kernel name
                ]
            )

            # Launch kernel with correct symbol reference
            gpu.LaunchFuncOp(
                kernel=kernel_ref,
                gridSizeX=grid_size_x.result,
                gridSizeY=grid_size_y.result,
                gridSizeZ=c1.result,
                blockSizeX=block_size_x.result,
                blockSizeY=block_size_y.result,
                blockSizeZ=c1.result,
                kernelOperands=host_func.body.blocks[0].arguments,
                asyncToken=None,
                asyncDependencies=[],
            )

            func.ReturnOp([])
    return module

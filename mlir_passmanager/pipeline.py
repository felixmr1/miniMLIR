from mlir.passmanager import PassManager
from mlir.ir import Module


# TODO: Make passes dynamic? Or keep opinionated?
def lower(module: Module) -> Module:
    # Create pass manager
    pm = PassManager()
    pm.add("convert-linalg-to-parallel-loops")
    pm.add("convert-scf-to-cf")
    pm.add("convert-cf-to-llvm")
    # TODO: Extract GPU features automatically
    pm.add(
        "gpu-lower-to-nvvm-pipeline{cubin-chip=sm_89 cubin-features=+ptx70 cubin-format=isa kernel-bare-ptr-calling-convention=1 opt-level=2}"
    )

    # Get the operation from the module
    module_op = module.operation
    pm.run(module_op)
    return module

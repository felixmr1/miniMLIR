# miniMLIR

MLIR-based code generation using Python.

## Overview

Python
      ↓
High-level MLIR
      ↓
GPU dialect
      ↓
NVVM dialect
      ↓
LLVM IR
      ↓
PTX/CUDA (current implementation)
      ↓
GPU Kernels

## Requirements

- Python 3.8+
- LLVM/MLIR (with Python bindings)
- CUDA Toolkit
- CuPy
- NumPy

## Installation

### Install Python dependencies
```bash
pip install numpy cupy-cuda12x  # Use appropriate CUDA version
```

### Build llvm-project from scratch (to get python bindnings)
```bash
git clone git@github.com:llvm/llvm-project.git
cd llvm-project
mkdir build && cd build
cmake -G Ninja \
	-DLLVM_ENABLE_PROJECTS="mlir;clang" \
	-DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
	-DCMAKE_BUILD_TYPE=Release \
	-DMLIR_ENABLE_CUDA_RUNNER=ON \
	-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc \
	-DCMAKE_CUDA_ARCHITECTURES=86 \
	-DLLVM_ENABLE_ASSERTIONS=ON \
	-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
	-DPython3_EXECUTABLE=$(which python) \
	-DLLVM_ENABLE_RTTI=ON \
	../llvm
```

## Usage

The main entry point is matmul.py:

```python
import numpy as np
from matmul import matmul
```

### Create sample matrices
a = np.random.randn(1024, 1024).astype(np.float32)
b = np.random.randn(1024, 1024).astype(np.float32)

### Perform GPU matrix multiplication
result = matmul(a, b)

## Implementation Details

### Block and Grid Configuration

The implementation uses fixed block sizes of 16x16 for simplicity. Grid sizes are calculated based on matrix dimensions:

def calculate_grid_size(matrix_size: tuple, block_size: tuple) -> tuple:
    grid_size_x = (matrix_size[0] + block_size[0] - 1) // block_size[0]
    grid_size_y = (matrix_size[1] + block_size[1] - 1) // block_size[1]
    return (int(grid_size_x), int(grid_size_y))

### MLIR Generation

The project uses MLIR's Python bindings to generate code progressively through multiple dialects:

1. High-level matrix multiplication using linalg dialect
2. GPU kernel generation with proper block/thread configuration
3. Lowering to NVVM/PTX for CUDA execution

### Performance Considerations

- Uses fixed 16x16 thread blocks (for now)
- Grid size adapts to matrix dimensions
- Direct global memory access (future improvement: add shared memory)
- Bounds checking for non-standard matrix sizes

## Examples

Simple matrix multiplication:

```python
from matmul import matmul
import numpy as np

# Create test matrices
a = np.random.randn(32, 32).astype(np.float32)
b = np.random.randn(32, 32).astype(np.float32)

# CPU reference result
expected = np.matmul(a, b)

# GPU computation
result = matmul(a, b)

# Verify results
max_diff = np.max(np.abs(result - expected))
print(f"Maximum difference between CPU and GPU: {max_diff}")
```

## Future Improvements

- [ ] More Ops
- [ ] Add shared memory usage for better performance
- [ ] Support for different data types
- [ ] Dynamic block size selection based on matrix size
- [ ] Memory coalescing optimizations
- [ ] Better error handling and bounds checking
- [ ] Support for non-square matrices
- [ ] Device support CPU/CUDA/ROCm/etc..

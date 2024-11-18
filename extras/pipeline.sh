#!/bin/bash

# Check if the argument is provided
if [ -z "$1" ]; then
	echo "Error: No target specified."
	echo "Usage: $0 <cpu|cuda>"
	exit 1
fi

# Get the first argument
target="$1"

# Run commands based on the argument
if [ "$target" == "cuda" ]; then
	mlir-opt matmul_gpu.mlir \
		--mlir-print-ir-before-all \
		--mlir-print-op-generic \
		--convert-linalg-to-parallel-loops \
		--gpu-lower-to-nvvm-pipeline="cubin-chip=sm_89 cubin-features=+ptx70 cubin-format=isa kernel-bare-ptr-calling-convention=1 opt-level=2" \
		--llvm-legalize-for-export
elif [ "$target" == "cpu" ]; then
	mlir-opt matmul_cpu.mlir \
		--convert-linalg-to-loops \
		--convert-scf-to-cf \
		--convert-arith-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts
else
	echo "Error: Invalid target '$target'. Please specify 'cpu' or 'cuda'."
	echo "Usage: $0 <cpu|cuda>"
	exit 1
fi

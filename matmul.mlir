module attributes {gpu.container_module} {
  // Device code module
  gpu.module @kernels {
    gpu.func @matmul_kernel(%A: memref<1024x1024xf32>,
                           %B: memref<1024x1024xf32>,
                           %C: memref<1024x1024xf32>)
        kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>} {
      linalg.matmul ins(%A, %B: memref<1024x1024xf32>, memref<1024x1024xf32>)
                    outs(%C: memref<1024x1024xf32>)
      gpu.return
    }
  }

  // Host code
  func.func @matmul(%A: memref<1024x1024xf32>,
                    %B: memref<1024x1024xf32>,
                    %C: memref<1024x1024xf32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    gpu.launch_func @kernels::@matmul_kernel
      blocks in (%c32, %c32, %c1)
      threads in (%c32, %c32, %c1)
      args(%A : memref<1024x1024xf32>,
           %B : memref<1024x1024xf32>,
           %C : memref<1024x1024xf32>)
    return
  }
}

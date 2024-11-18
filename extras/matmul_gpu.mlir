module attributes {gpu.container_module} {
  // device code
  gpu.module @kernels {
    gpu.func @matmul_kernel(%a: memref<1024x1024xf32>,
                           %b: memref<1024x1024xf32>,
                           %c: memref<1024x1024xf32>)
        kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>} {
      // get the thread indices
      %tx = gpu.thread_id x
      %ty = gpu.thread_id y

      // load a[tx,ty] and store to c[tx,ty]
      %val = memref.load %a[%tx, %ty] : memref<1024x1024xf32>
      memref.store %val, %c[%tx, %ty] : memref<1024x1024xf32>
      gpu.return
    }
  }

  // host code
  func.func @matmul(%a: memref<1024x1024xf32>,
                    %b: memref<1024x1024xf32>,
                    %c: memref<1024x1024xf32>) -> memref<1024x1024xf32> {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    gpu.launch_func @kernels::@matmul_kernel
      blocks in (%c32, %c32, %c1)
      threads in (%c32, %c32, %c1)
      args(%a : memref<1024x1024xf32>,
           %b : memref<1024x1024xf32>,
           %c : memref<1024x1024xf32>)

    return %c : memref<1024x1024xf32>
  }
}

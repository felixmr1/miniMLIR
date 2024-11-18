module {
  func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>)
    attributes {llvm.emit_c_interface} {
    linalg.matmul
      ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
      outs(%C: memref<?x?xf32>)
    return
  }
}

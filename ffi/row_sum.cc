// ffi/row_sum.cc

#include <cstdint>
#include <utility>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Compute Row Sum
void ComputeRowSum(int64_t num_rows, int64_t num_cols, const float* x, float* y) {
  for (int64_t i = 0; i < num_rows; ++i) {
    float sum = 0.0f;
    for (int64_t j = 0; j < num_cols; ++j) {
      sum += x[i * num_cols + j];
    }
    y[i] = sum;
  }
}

// Helper function to extract dimensions
template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T>& buffer) {
  auto dims = buffer.dimensions();
  if (dims.size() != 2) {
    return {0, 0}; // Expecting a 2D matrix
  }
  return {dims[0], dims[1]}; // num_rows, num_cols
}

// FFI Handler Implementation
ffi::Error RowSumImpl(ffi::Buffer<ffi::F32> x,
                      ffi::ResultBuffer<ffi::F32> y) {
  auto [num_rows, num_cols] = GetDims(x);
  if (num_rows == 0 || num_cols == 0) {
    return ffi::Error::InvalidArgument("RowSum input must be a 2D array with non-zero dimensions");
  }

  const float* input_data = x.typed_data();
  float* output_data = y->typed_data();

  ComputeRowSum(num_rows, num_cols, input_data, output_data);

  return ffi::Error::Success();
}

// Register the FFI Handler
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RowSum, RowSumImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()  // x (2D matrix)
        .Ret<ffi::Buffer<ffi::F32>>()  // y (1D vector)
);


// ffi/rms_norm.cc

#include <cmath>
#include <cstdint>
#include <utility>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Compute RMS Normalization
float ComputeRmsNorm(float eps, int64_t size, const float* x, float* y) {
  float sm = 0.0f;
  for (int64_t n = 0; n < size; ++n) {
    sm += x[n] * x[n];
  }
  float scale = 1.0f / std::sqrt(sm / static_cast<float>(size) + eps);
  for (int64_t n = 0; n < size; ++n) {
    y[n] = x[n] * scale;
  }
  return scale;
}

// Helper function to extract dimensions
template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T>& buffer) {
  auto dims = buffer.dimensions();
  if (dims.size() == 0) {
    return {0, 0};
  }
  return {buffer.element_count(), dims.back()};
}

// FFI Handler Implementation
ffi::Error RmsNormImpl(float eps, ffi::Buffer<ffi::F32> x,
                       ffi::ResultBuffer<ffi::F32> y) {
  auto [totalSize, lastDim] = GetDims(x);
  if (lastDim == 0) {
    return ffi::Error::InvalidArgument("RmsNorm input must be an array");
  }
  for (int64_t n = 0; n < totalSize; n += lastDim) {
    ComputeRmsNorm(eps, lastDim, &(x.typed_data()[n]), &(y->typed_data()[n]));
  }
  return ffi::Error::Success();
}

// Register the FFI Handler
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RmsNorm, RmsNormImpl,
    ffi::Ffi::Bind()
        .Attr<float>("eps")
        .Arg<ffi::Buffer<ffi::F32>>()  // x
        .Ret<ffi::Buffer<ffi::F32>>()  // y
);


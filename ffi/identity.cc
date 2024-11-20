// ffi/identity.cc

#include <algorithm>
#include <cstdint>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Fixed buffer size
constexpr int64_t FIXED_SIZE = 3;

// Modified Identity function: multiplies each element by 2
ffi::Error IdentityImpl(ffi::Buffer<ffi::F32> x, ffi::ResultBuffer<ffi::F32> y) {
  // Optional: Assert the buffer size at compile-time or handle mismatches
  // Here, we assume it's always FIXED_SIZE

  const float* x_data = x.typed_data();      // Pointer to input data
  float* y_data = y->typed_data();           // Pointer to output buffer

  // Multiply each element by 2 and store in y_data
  for (int64_t i = 0; i < FIXED_SIZE; ++i) {
    y_data[i] = x_data[i] * 2.0f;
  }

  return ffi::Error::Success();              // Indicate successful execution
}

// Register the FFI handler
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Identity, IdentityImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()      // Input buffer x
        .Ret<ffi::Buffer<ffi::F32>>()      // Output buffer y
);


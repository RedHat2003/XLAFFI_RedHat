// ffi/sumrows.cc

#include <cstdint>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Define fixed dimensions for the matrix
constexpr int64_t ROWS = 4;  // Number of rows
constexpr int64_t COLS = 3;  // Number of columns

ffi::Error SumRows(ffi::Buffer<ffi::F32> input, ffi::ResultBuffer<ffi::F32> output) {
    const float* input_data = input.typed_data();
    float* output_data = output->typed_data();

    // Sum each row
    for (int64_t row = 0; row < ROWS; ++row) {
        float sum = 0.0f;
        for (int64_t col = 0; col < COLS; ++col) {
            sum += input_data[row * COLS + col];
        }
        output_data[row] = sum;
    }

    return ffi::Error::Success();
}

// Register the SumRows FFI handler with XLA
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SumRows,      // Handler name as referenced in Python
    SumRows,      // C++ function implementing the handler
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()      // Input buffer (matrix)
        .Ret<ffi::Buffer<ffi::F32>>()      // Output buffer (vector)
);


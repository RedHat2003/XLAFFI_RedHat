// sumrows.cc

#include <cstdint>
#include "xla/ffi/ffi.h"

namespace ffi = xla::ffi;

// Define fixed dimensions for the matrix
constexpr int64_t ROWS = 4;  // Number of rows
constexpr int64_t COLS = 3;  // Number of columns

ffi::Error SumRowsImpl(ffi::OpaqueBuffer* input, ffi::OpaqueBuffer* output) {
    const float* input_data = static_cast<const float*>(input->data);
    float* output_data = static_cast<float*>(output->data);
    int64_t input_size = input->length / sizeof(float);
    int64_t output_size = output->length / sizeof(float);

    // Check that the input size matches ROWS * COLS
    if (input_size != ROWS * COLS) {
        return ffi::Error::InvalidArgument("Input size does not match expected matrix dimensions.");
    }

    // Check that the output size matches ROWS
    if (output_size != ROWS) {
        return ffi::Error::InvalidArgument("Output size does not match expected vector size.");
    }

    // Sum each row
    for (int64_t row = 0; row < ROWS; ++row) {
        float sum = 0.0f;
        for (int64_t col = 0; col < COLS; ++col) {
            sum += input_data[row * COLS + col];
        }
        output_data[row] = sum;
    }

    return ffi::Error::Ok();
}

XLA_FFI_REGISTER_RAW_HANDLER("SumRows", SumRowsImpl);


# python/row_sum.py

import ctypes
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.extend as jex
import numpy as np

# Locate the compiled shared library
ffi_build_dir = Path(__file__).parent.parent / "ffi" / "_build"
lib_path = next(ffi_build_dir.glob("row_sum.so"), None)

if lib_path is None:
    raise FileNotFoundError("Could not find 'row_sum.so' in the build directory.")

# Load the shared library using ctypes
row_sum_lib = ctypes.cdll.LoadLibrary(str(lib_path))

# Register the FFI target with JAX
jex.ffi.register_ffi_target(
    "row_sum",
    jex.ffi.pycapsule(row_sum_lib.RowSum),
    platform="cpu"
)

def row_sum_ref(x):
    """Reference implementation using JAX."""
    return jnp.sum(x, axis=1)

def row_sum(x):
    """Row Sum using FFI."""
    if x.dtype != jnp.float32:
        raise ValueError("Only the float32 dtype is implemented by row_sum")
    
    call = jex.ffi.ffi_call(
        "row_sum",
        jax.ShapeDtypeStruct((x.shape[0],), x.dtype),
        vmap_method="broadcast_all",
    )
    
    return call(x)

# Testing the implementation
if __name__ == "__main__":
    # Create a sample 3x5 matrix
    x = jnp.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0]
    ], dtype=jnp.float32)
    print("Input Matrix:\n", x)
    
    # Compute row sums using FFI
    y_ffi = row_sum(x)
    print("Row Sum (FFI):\n", y_ffi)
    
    # Compute row sums using reference implementation
    y_ref = row_sum_ref(x)
    print("Row Sum (Reference):\n", y_ref)
    
    # Verify correctness
    np.testing.assert_allclose(y_ffi, y_ref, rtol=1e-5)
    print("Row sum via FFI matches the reference implementation.")


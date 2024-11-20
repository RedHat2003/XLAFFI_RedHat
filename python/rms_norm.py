# python/rms_norm.py

import ctypes
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.extend as jex
import numpy as np

# Locate the compiled shared library
ffi_build_dir = Path(__file__).parent.parent / "ffi" / "_build"
lib_path = next(ffi_build_dir.glob("rms_norm.so"), None)

if lib_path is None:
    raise FileNotFoundError("Could not find 'rms_norm.so' in the build directory.")

# Load the shared library using ctypes
rms_norm_lib = ctypes.cdll.LoadLibrary(str(lib_path))

# Register the FFI target with JAX
jex.ffi.register_ffi_target(
    "rms_norm",
    jex.ffi.pycapsule(rms_norm_lib.RmsNorm),
    platform="cpu"
)

def rms_norm_ref(x, eps=1e-5):
    """Reference implementation using JAX."""
    scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x / scale

def rms_norm(x, eps=1e-5):
    """RMS Normalization using FFI."""
    if x.dtype != jnp.float32:
        raise ValueError("Only the float32 dtype is implemented by rms_norm")
    
    call = jex.ffi.ffi_call(
        "rms_norm",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )
    
    return call(x, eps=np.float32(eps))

# Testing the implementation
if __name__ == "__main__":
    x = jnp.linspace(-0.5, 0.5, 15).reshape((3, 5)).astype(jnp.float32)
    print("Input:\n", x)
    
    # Compute using FFI
    y_ffi = rms_norm(x)
    print("RMS Norm (FFI):\n", y_ffi)
    
    # Compute using reference
    y_ref = rms_norm_ref(x)
    print("RMS Norm (Reference):\n", y_ref)
    
    # Verify correctness
    np.testing.assert_allclose(y_ffi, y_ref, rtol=1e-5)
    print("RMS normalization via FFI matches the reference implementation.")


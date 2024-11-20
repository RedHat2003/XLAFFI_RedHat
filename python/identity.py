# python/identity.py

import ctypes
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.extend as jex
import numpy as np

# Locate the compiled shared library
ffi_build_dir = Path(__file__).parent.parent / "ffi" / "_build"
lib_path = next(ffi_build_dir.glob("identity.so"), None)

if lib_path is None:
    raise FileNotFoundError("Could not find 'identity.so' in the build directory.")

# Load the shared library using ctypes
identity_lib = ctypes.cdll.LoadLibrary(str(lib_path))

# Register the FFI target with JAX
jex.ffi.register_ffi_target(
    "identity",
    jex.ffi.pycapsule(identity_lib.Identity),
    platform="cpu"
)

def identity(x):
    if x.dtype != jnp.float32:
        raise ValueError("Only float32 dtype is implemented by identity")
    call = jex.ffi.ffi_call(
        "identity",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",  # Corrected from 'broadcast'
    )
    return call(x)

# Testing the implementation
if __name__ == "__main__":
    x = jnp.ones(shape = (4, ), dtype=jnp.float32)

    # Use the identity function via FFI
    y = identity(x)
    print("Output y:", y)



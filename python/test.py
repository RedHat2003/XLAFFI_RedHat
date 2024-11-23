import ctypes 
from pathlib import Path 
import jax 
import jax.numpy as jnp 
import jax.extend as jex 
import numpy as np 

ffi_build_dir = Path (__file__).parent.parent /"ffi" / "_build" 
lib_path = next (ffi_build_dir.glob("sayhi.so") , None ) 

sayhi_lib = ctypes.cdll.LoadLibrary (lib_path) 

jex.ffi.register_ffi_target(
    "sayhi",
    jex.ffi.pycapsule(sayhi_lib.Print),
    platform="cpu"
)

def sayhi(x):
    if x.dtype != jnp.float32:
        raise ValueError("Only float32 dtype is implemented by identity")
    call = jex.ffi.ffi_call(
        "sayhi",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",  # Corrected from 'broadcast'
    )
    return call(x)

# Testing the implementation
if __name__ == "__main__":
    x = jnp.ones(shape = (4, ), dtype=jnp.float32)

    # Use the identity function via FFI
    y = sayhi(x)
    print("Output y:", y)




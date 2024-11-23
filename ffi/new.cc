#include <cstdio> 
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/c_api.h"
namespace ffi = xla::ffi ; 

ffi::Error SayHi(ffi::BufferR1 <ffi::F32> x , ffi::Result<ffi::BufferR1<ffi::F32>> y ){
	const float* data_x = x.typed_data() ; 
	float* data_y = y->typed_data() ; 
	printf ("u are finally there !");
	for (int i = 0 ; i < 4 ; i ++) {
	
		data_y[i]  = data_x [i] ; 
	}

	return ffi::Error::Success();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Print, SayHi,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Ret<ffi::BufferR1<ffi::F32>>());  // Ensure closing parenthesis


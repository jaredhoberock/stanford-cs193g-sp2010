# Prerequisites

To get started programming with CUDA, download and install the [CUDA Toolkit and developer driver](https://developer.nvidia.com/cuda-downloads). The toolkit includes `nvcc`, the NVIDIA CUDA Compiler, and other software necessary to develop CUDA applications. The driver ensures that GPU programs run correctly on CUDA-capable hardware, which you'll also need.

You can confirm that the CUDA Toolkit is correctly installed on your machine by running `nvcc --version` from a command line. For example, on a Linux machine,

```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_18:49:52_PDT_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
```

outputs the compiler information. If the previous command was not successful, then the CUDA Toolkit is likely not installed, or the path to nvcc (`C:\CUDA\bin` on Windows machines, `/usr/local/cuda/bin` elsewhere) is not part of your `PATH` environment variable.

Additionally, you'll also need a host compiler which works with `nvcc` to compile and build CUDA programs. On Windows, this is `cl.exe`, the Microsoft compiler, which ships with Microsoft Visual Studio. On POSIX OSes, this is `gcc` or `g++`. The official [CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) can tell you which compiler versions are supported on your particular platform.

To make sure everything is set up correctly, let's compile and run a trivial CUDA program to ensure all the tools work together correctly.

```
#include <stdio.h>

__global__ void foo()
{
}

int main()
{
  foo<<<1,1>>>();

  printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));

  return 0;
}
```

To compile this program, copy it to a file called `test.cu` and compile it from the command line. For example, on a Linux system, the following should work:

```
$ nvcc test.cu -o test
$ ./test
CUDA error: no error
```

If the program succeeds without error, then let's start coding!


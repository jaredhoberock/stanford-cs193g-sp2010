# Prerequisites #

To get started programming with CUDA, download and install the [CUDA Toolkit and developer driver](http://www.nvidia.com/getcuda).  The toolkit includes `nvcc`, the NVIDIA CUDA Compiler, and other software necessary to develop CUDA applications. The driver ensures that GPU programs run correctly on [CUDA-capable hardware](http://www.nvidia.com/object/cuda_gpus.htm), which you'll also need.

You can confirm that the CUDA Toolkit is correctly installed on your machine by running `nvcc --version` from a command line.  For example, on a Mac OSX machine,

```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2009 NVIDIA Corporation
Built on Thu_Mar_11_12:08:00_PST_2010
Cuda compilation tools, release 3.0, V0.2.1221
```

outputs the compiler information. If the previous command was not successful, then the CUDA Toolkit is likely not installed, or the path to `nvcc` (`C:\CUDA\bin` on Windows machines, `/usr/local/cuda/bin` on POSIX OSes) is not part of your `PATH` environment variable.

Additionally, you'll also need a _host compiler_ which works with `nvcc` to compile and build CUDA programs.  On Windows, this is `cl.exe`, the Microsoft compiler, which ships with Microsoft Visual Studio. On POSIX OSes, this is `gcc` or `g++`.  The official CUDA [Getting Started Guides](http://www.nvidia.com/getcuda) can tell you which compiler versions are supported on your particular platform.

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

To compile this program, copy it to a file called `test.cu` and compile it from the command line.  For example, on a Mac OSX system, the following should work:

```
$ nvcc test.cu -o test
$ ./test 
CUDA error: no error
```

If the program succeeds without error, then let's [start coding](TutorialAHeterogeneousProgrammingModel.md)!
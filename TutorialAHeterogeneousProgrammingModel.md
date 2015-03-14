# A Heterogeneous Programming Model #

One of the most important ways in which CUDA programs differ from typical C programs is that the programmer is presented with a _heterogeneous_ view of processor and memory resources.  On the one hand, the CPU, a serial _host_ processor, accesses memory and executes function calls similar to a traditional C program.  On the other hand, the GPU, referred to as the _device_, executes parallel _kernels_ and has special access to a separate memory space.

The following code listing introduces CUDA's heterogeneous memory model by demonstrating the difference between the host and device memory spaces.  The program, which is executed by the serial CPU, or host, allocates and interacts with host memory using functions which should be familiar to C programmers.  Similarly, the host uses analogous system calls, such as `cudaMalloc` and `cudaFree` to manage device space memory.

```
#include <stdlib.h>
#include <stdio.h>

int main(void)
{
  int num_elements = 16;
  int num_bytes = num_elements * sizeof(int);

  int *device_array = 0;
  int *host_array = 0;

  // malloc host memory
  host_array = (int*)malloc(num_bytes);

  // cudaMalloc device memory
  cudaMalloc((void**)&device_array, num_bytes);

  // zero out the device array with cudaMemset
  cudaMemset(device_array, 0, num_bytes);

  // copy the contents of the device array to the host
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  // print out the result element by element
  for(int i = 0; i < num_elements; ++i)
    printf("%d ", host_array[i]);

  // use free to deallocate the host array
  free(host_array);

  // use cudaFree to deallocate the device array
  cudaFree(device_array);

  return 0;
}
```

To compile this program, copy it to a file named `cuda_memory_model.cu` and invoke `nvcc` from the command line.  If you're using a Windows machine, the following commands should work:

```
C:\Program Files\Microsoft Visual Studio 8\VC>nvcc cuda_memory_model.cu -o cuda_memory_model.exe

C:\Program Files\Microsoft Visual Studio 8\VC>cuda_memory_model
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

The result is a sequence of zeros, as expected.  Note that in order to access the device space result, the host was required to explicitly copy the array to host memory.  Even though the `cudaMalloc` system call returns what appears to be a normal pointer, the memory it references is unavailable to the host. Attempts by the host to dereference it will likely yield a segmentation fault.  Similarly, attempts by the device to dereference a host pointer will also meet with disaster.  The full source code of this example, with additional commentary, is available in this site's [code repository](http://code.google.com/p/stanford-cs193g-sp2010/source/browse/trunk/tutorials/cuda_memory_model.cu).

Now that we know the difference between host and device space memory, let's write [our first kernel](TutorialHelloWorld.md)!
# `__device__` Functions #

In the last couple sections, we learned how to use the `__global__` keyword to mark a function as code that the host can call to cause the invocation of a parallel kernel on the device. Within a `__global__` function, each CUDA thread so spawned follows its own path of execution _serially_.  In CUDA, kernels consist of mostly arbitrary C/C++ code, so they can get sophisticated quickly.  As productive parallel programmers, we'll will want to [abstract](http://en.wikipedia.org/wiki/Abstraction_(computer_science)) and [encapsulate](http://en.wikipedia.org/wiki/Information_hiding) our kernel code into functions.

The `__device__` keyword lets us mark functions as callable from threads executing on the device.  The syntax is like the `__global__` keyword: we just prepend it to the function signature:

```
__device__ float my_device_function(float x)
{
  return x + 1;
}
```

Though `__device__` functions are similar to `__global__` functions in that they are executed by device threads, they actually behave more like normal C functions.  Unlike `__global__` functions, `__device__` functions cannot be configured (no `<<<B,T>>>` needed) and aren't subject to any special restrictions on the types of their parameters or results.  Host code isn't allowed to call `__device__` functions directly -- if we want access to the functionality in a `__device__` function, we need to write a `__global__` function to call it for us!

As you might expect, `__device__` functions can call other functions decorated with `__device__`:

```
__device__ float my_second_device_function(float y)
{
  return my_device_function(y) / 2;
}
```

As long as they don't call themselves:

```
__device__ int my_illegal_recursive_device_function(int x)
{
  if(x == 0) return 1;
  return x * my_illegal_recursive_device_function(x-1);
}
```

This code produces the following compiler error:

```
$nvcc factorial.cu
factorial.cu
./factorial.cu(4): Error: Recursive function call is not supported yet: _Z36my_illegal_recursive_device_functioni
```

The following code listing shows how we might use `__device__` functions to package up various bits of code when developing a CUDA kernel.

```
#include <stdlib.h>
#include <stdio.h>

__device__ int get_global_index(void)
{
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int get_constant(void)
{
  return 7;
}

__global__ void kernel1(int *array)
{
  int index = get_global_index();
  array[index] = get_constant();
}

__global__ void kernel2(int *array)
{
  int index = get_global_index();
  array[index] = get_global_index();
}

int main(void)
{
  int num_elements = 256;
  int num_bytes = num_elements * sizeof(int);

  int *device_array = 0;
  int *host_array = 0;

  // allocate memory
  host_array = (int*)malloc(num_bytes);
  cudaMalloc((void**)&device_array, num_bytes);

  int block_size = 128;
  int grid_size = num_elements / block_size;

  // launch kernel1 and inspect its results
  kernel1<<<grid_size,block_size>>>(device_array);
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  printf("kernel1 results:\n");
  for(int i = 0; i < num_elements; ++i)
  {
    printf("%d ", host_array[i]);
  }
  printf("\n\n");

  // launch kernel2 and inspect its results
  kernel2<<<grid_size,block_size>>>(device_array);
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  printf("kernel2 results:\n");
  for(int i = 0; i < num_elements; ++i)
  {
    printf("%d ", host_array[i]);
  }
  printf("\n\n");

  // deallocate memory
  free(host_array);
  cudaFree(device_array);
  return 0;
}
```

This is the first time we've included more than one `__global__` function in our example.  This is fine -- they're just like C functions in this way.  Note also that both `kernel1` and `kernel2` can call any of the `__device__` functions they can "see".  The usual C/C++ [scoping](http://en.wikipedia.org/wiki/Scope_(programming)) rules apply.

For `kernel1`, we've essentially taken our `__global__` function from a previous section and [refactored](http://en.wikipedia.org/wiki/Refactoring) it into `__device__` functions `get_global_index` and `get_constant`.  `get_constant` isn't so useful, but `get_global_index` encapsulates the tedious calculation of each CUDA thread's global index in the grid.  Rather than repeating ourself, both kernels can simply call the `__device__` function.  Note that `get_global_index` uses the built-in variables `blockIdx`, `blockDim`, and `threadIdx`, which are available to `__device__` functions as they are to `__global__` functions.

An expanded version of this example is [available](http://code.google.com/p/stanford-cs193g-sp2010/source/browse/trunk/tutorials/device_functions.cu) in the site's code repository.

In the [next section](TutorialWhenSomethingGoesWrong.md), we'll learn how to detect CUDA errors, such as crashing kernels.
# `__global__` Functions #

In the [last section](TutorialHelloWorld.md), we learned that `__global__` functions serve as the point of entry into a kernel which executes in parallel on a GPU device.  In fact, `__global__` functions expose a particular form of parallel computation called [data parallelism](http://en.wikipedia.org/wiki/Data_parallelism).  The basic idea of data parallelism is to distribute a large task composed of many similar but independent pieces across a set of computational resources.  In CUDA, the task to be performed is described by a `__global__` function, and the computational resources are CUDA threads.  Individual threads work on individual subtasks, and since they are independent, these subtasks can be performed in parallel.

Consider the data parallel task of filling an array with a particular value.  We have almost all the tools we need to implement this operation with a `__global__` function in CUDA: we know [how to allocate device memory](TutorialAHeterogeneousProgrammingModel.md), and we know [how to write \_\_global\_\_ functions and instantiate CUDA threads](TutorialHelloWorld.md) which execute them.  The missing piece is the mapping of subtasks (setting a particular array element to the given value) to particular CUDA threads.

In order to map each particular CUDA thread to a particular subtask, CUDA provides built-in variables which uniquely identify each thread in its _grid_ of thread blocks.  To see how to use them, let's look at some code which fills a device array with the value `7`.

```
#include <stdlib.h>
#include <stdio.h>

__global__ void kernel(int *array)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  array[index] = 7;
}

int main(void)
{
  int num_elements = 256;

  int num_bytes = num_elements * sizeof(int);

  // pointers to host & device arrays
  int *device_array = 0;
  int *host_array = 0;

  // malloc a host array
  host_array = (int*)malloc(num_bytes);

  // cudaMalloc a device array
  cudaMalloc((void**)&device_array, num_bytes);

  int block_size = 128;
  int grid_size = num_elements / block_size;

  kernel<<<grid_size,block_size>>>(device_array);

  // download and inspect the result on the host:
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  // print out the result element by element
  for(int i=0; i < num_elements; ++i)
  {
    printf("%d ", host_array[i]);
  }
 
  // deallocate memory
  free(host_array);
  cudaFree(device_array);
}
```

The variables of interest are used inside the `__global__` function, `kernel`.  `threadIdx.x` uniquely identifies each CUDA thread within its thread block.  Similarly, `blockIdx.x` uniquely identifies each thread block within the grid at large.  These variables are combined with `blockDim.x`, which encodes the number of threads in each thread block, to yield an index which uniquely identifies each CUDA thread within the entire grid:

```
  int index = blockIdx.x * blockDim.x + threadIdx.x;
```

With this index, it's easy to ensure each CUDA thread has a 1-to-1 mapping to a unique array element if we've been careful to launch exactly as many CUDA threads as we have array locations to write.  In fact, when expressed this way, it's easy to recognize the relationship between the body of a `__global__` function and the body of a traditional C for loop:

```
  for(int index = 0; index < num_elements; ++index)
  {
    array[index] = 7;
  }
```

The crucial difference is that each for loop iteration executes _sequentially_ while a `__global__` function executes _in parallel_.

You might have noticed that in this example, we passed an argument to our `__global__` function, specifically, a pointer to the device array we wished to fill.  `__global__` functions are much like regular C functions in this regard -- they may accept parameters, as long as they are not [C++ references](http://en.wikipedia.org/wiki/Reference_(C%2B%2B)).  On the other hand, one important way that `__global__` functions differ from normal C functions is that they must return `void`.  The rationale is clear:  of the potentially hundreds of thousands of threads executing a particular `__global__` function, whose result should be returned to the host? `[1]`

The full source code for this example along with additional commentary is [available](http://code.google.com/p/stanford-cs193g-sp2010/source/browse/trunk/tutorials/global_functions.cu) in this site's code repository.  Additionally, we also recommend working through a [closely-related example](http://code.google.com/p/stanford-cs193g-sp2010/source/browse/trunk/tutorials/vector_addition.cu) which performs parallel vector addition.  The vector addition example addresses an additional wrinkle not examined here: how to robustly handle workload sizes which are not evenly divided by `blockDim.x`.

Keen observers will note this example's final curiosity: the `.x` member of each of the built-in variables `threadIdx`, `blockIdx`, and `blockDim`.  In fact, these variables are vector-typed, implying multi-dimensional kernel launch, which is the subject of the [next section](TutorialMultidimensionalKernelLaunch.md).

<br><font size='1'><code>[1]</code> In fact, this rationale doesn't tell the whole story.  <code>__global__</code> function launches are actually <a href='http://en.wikipedia.org/wiki/Asynchronous_I/O'>asynchronous</a>: control immediately returns to the calling function before the kernel executes.  Even supposing a particular kernel's unique result was unambiguous, there wouldn't yet be a result to return!</font>
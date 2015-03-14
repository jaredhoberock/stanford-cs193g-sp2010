# Multidimensional Kernel Launch #

The [last section](TutorialGlobalFunctions.md) hinted that `__global__` functions might be launched using grid configurations that weren't strictly one-dimensional.  In fact, if it is convenient for describing our data parallel problem, CUDA allows us to create thread blocks in 1-, 2-, or 3D. `[1]`  Many problems, such as the previous section's array processing tasks, are most naturally described in a flat, linear style mimicking our mental model of C's memory layout. Other tasks, particularly those often encountered in the computational sciences, are naturally embedded in two or three dimensions.  For example, [image processing](http://www.youtube.com/results?search_query=image+processing+cuda&page=&utm_source=opensearch) tasks typically impose a regular 2D raster (an image) over the problem domain.  [Computational fluid dynamics](http://www.youtube.com/results?search_query=cuda+cfd&search_type=&aq=f) tasks might be most naturally expressed by partitioning a volume over a 3D grid.

The following code listing demonstrates an example of a 2D kernel launch, and shows how to map two dimensional thread and block indices to the physical one dimensional layout of device memory.

```
#include <stdlib.h>
#include <stdio.h>

__global__ void kernel(int *array)
{
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  // map the two 2D indices to a single linear, 1D index
  int grid_width = gridDim.x * blockDim.x;
  int index = index_y * grid_width + index_x;

  // map the two 2D block indices to a single linear, 1D block index
  int result = blockIdx.y * gridDim.x + blockIdx.x;

  // write out the result
  array[index] = result;
}

int main(void)
{
  int num_elements_x = 16;
  int num_elements_y = 16;

  int num_bytes = num_elements_x * num_elements_y * sizeof(int);

  int *device_array = 0;
  int *host_array = 0;

  // allocate memory in either space
  host_array = (int*)malloc(num_bytes);
  cudaMalloc((void**)&device_array, num_bytes);

  // create two dimensional 4x4 thread blocks
  dim3 block_size;
  block_size.x = 4;
  block_size.y = 4;

  // configure a two dimensional grid as well
  dim3 grid_size;
  grid_size.x = num_elements_x / block_size.x;
  grid_size.y = num_elements_y / block_size.y;

  // grid_size & block_size are passed as arguments to the triple chevrons as usual
  kernel<<<grid_size,block_size>>>(device_array);

  // download and inspect the result on the host:
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  // print out the result element by element
  for(int row = 0; row < num_elements_y; ++row)
  {
    for(int col = 0; col < num_elements_x; ++col)
    {
      printf("%2d ", host_array[row * num_elements_x + col]);
    }
    printf("\n");
  }
  printf("\n");

  // deallocate memory
  free(host_array);
  cudaFree(device_array);
}
```

The host code of this example uses a type we haven't seen before: `dim3`.  In order to configure a two-dimensional grid, we pass values of this type to the triple chevrons instead of the plain integer arguments we've used before for one-dimensional kernels.  `dim3` is a simple vector type native to CUDA, and you can assume its definition looks something like:

```
  struct dim3
  {
    unsigned int x, y, z;
    ...
  };
```

You'll note that our example didn't set either of `block_size.z` or `grid_size.z`.  By default, they are initialized to `1`.  Though the dimensionality of our problem is 2D, notice we configured the launch using arithmetic analogous to  our previous 1D examples.  First, we described the total size of the data parallel task at hand:

```
  int num_elements_x = 16;
  int num_elements_y = 16;
```

Next, we decided how to partition the task into blocks of threads:

```
  dim3 block_size;
  block_size.x = 4;
  block_size.y = 4;
```

And finally, we decided how many blocks of threads were needed to completely cover the size of the problem:

```
  dim3 grid_size;
  grid_size.x = num_elements_x / block_size.x;
  grid_size.y = num_elements_y / block_size.y;
```

Our `__global__` function, `kernel` similarly performs arithmetic analogous to our previous examples to uniquely identify each thread involved in the computation at large.  The built-in variables `threadIdx`, `blockIdx`, and `blockDim` are munged together to provide a unique, global index.  This time, we do it in 2D:

```
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;
```

This time, we used a new built-in variable we haven't seen before, `gridDim`, which encodes the number of thread blocks in each dimension.  We used it to map the 2D global indices we computed, `index_x` and `index_y`, to a single, linear 1D index:

```
  int grid_width = gridDim.x * blockDim.x;
  int index = index_y * grid_width + index_x;
```

This was necessary in order to assign each individual CUDA thread to a unique element of `array`.

The actual work our kernel does is trivial.  Similarly to how we assigned a unique 1D index to each CUDA thread, we used `blockIdx` and `gridDim` to assign each thread block a unique index, which we wrote out to `array`.

To test the program, copy it to a file named `two_dimensional_kernel_launch.cu`.  You can also [find it](http://code.google.com/p/stanford-cs193g-sp2010/source/browse/trunk/tutorials/two_dimensional_kernel_launch.cu) in the site's code repository with additional comments. If you compile and run it, and you'll visualize the 2D structure of our threads:

```
$ nvcc two_dimensional_kernel_launch.cu -o two_dimensional_kernel_launch
$ ./two_dimensional_kernel_launch 
 0  0  0  0  1  1  1  1  2  2  2  2  3  3  3  3 
 0  0  0  0  1  1  1  1  2  2  2  2  3  3  3  3 
 0  0  0  0  1  1  1  1  2  2  2  2  3  3  3  3 
 0  0  0  0  1  1  1  1  2  2  2  2  3  3  3  3 
 4  4  4  4  5  5  5  5  6  6  6  6  7  7  7  7 
 4  4  4  4  5  5  5  5  6  6  6  6  7  7  7  7 
 4  4  4  4  5  5  5  5  6  6  6  6  7  7  7  7 
 4  4  4  4  5  5  5  5  6  6  6  6  7  7  7  7 
 8  8  8  8  9  9  9  9 10 10 10 10 11 11 11 11 
 8  8  8  8  9  9  9  9 10 10 10 10 11 11 11 11 
 8  8  8  8  9  9  9  9 10 10 10 10 11 11 11 11 
 8  8  8  8  9  9  9  9 10 10 10 10 11 11 11 11 
12 12 12 12 13 13 13 13 14 14 14 14 15 15 15 15 
12 12 12 12 13 13 13 13 14 14 14 14 15 15 15 15 
12 12 12 12 13 13 13 13 14 14 14 14 15 15 15 15 
12 12 12 12 13 13 13 13 14 14 14 14 15 15 15 15 
```

From this output, the 2D structure of both our thread blocks and grid is clear!

Now that we've got kernel launch down, the [next section](TutorialDeviceFunctions.md) introduces `__device__` functions.

<br><font size='1'><code>[1]</code> While thread blocks can be 1-, 2-, or 3D, grids of thread blocks are limited to 1- or 2D.</font>
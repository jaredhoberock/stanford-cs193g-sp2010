# When something goes wrong #

In the past few sections, we've learned how to [allocate and manage device memory](TutorialAHeterogeneousProgrammingModel.md), [write kernels](TutorialGlobalFunctions.md) which execute in parallel and in [multiple dimensions](TutorialMultidimensionalKernelLaunch.md), and [decompose our device code into functions](TutorialDeviceFunctions.md) so that our kernels may grow ever more sophisticated. By now, we have all the basic tools needed to start thinking about building non-trivial parallel applications. On the other hand, these tools we've collected also enable us to write buggy programs. With hundreds of thousands of threads come hundreds of thousands of things which could go wrong. This section discusses how to detect and deal with errors as soon as they occur.

To get started, let's write a buggy program that's obviously incorrect just to see what happens.

```
__global__ void foo(int *ptr)
{
  *ptr = 7;
}

int main(void)
{
  foo<<<1,1>>>(0);
  return 0;
}
```

This program doesn't even bother to allocate device memory -- it just passes a null pointer to a kernel which immediately writes to it. This is a rather blatant instance of a common programming error -- dereferencing a null pointer. Any sane program would sooner crash than attempt something so heinous. However, when I compile and run this program on my system, my screen flashes, but otherwise there's no indication that something went wrong:

```
$ nvcc crash.cu -o crash
$ ./crash
```

Most C and C++ programmers would probably be used to some sort of immediate feedback that something funny is afoot.  A debugging window might open, or a error message might be automatically written to the terminal indicating a [segmentation fault](http://en.wikipedia.org/wiki/Segmentation_fault). However, in a CUDA program, if we suspect an error has occurred during a kernel launch, then we must explicitly check for it after the kernel has executed.

The _CUDA Runtime_ is happy to field questions about abnormal program execution, but tends to be tight-lipped when it comes to offering unsolicited information. To find out if an error has occurred inside a kernel, the host needs to actively call the CUDA Runtime function `cudaGetLastError`, which returns a value encoding the kind of the last error it has encountered:

```
cudaError_t cudaGetLastError(void);
```

To make things slightly more complicated, we need to be sure to check for the error only after we're sure a kernel has finished executing. Because kernel launches are _asynchronous_, the host thread doesn't wait on the device to finish its work before continuing on with its own business, which might include launching more kernels, copying memory, enjoying a Mai Tai, etc. So in order to correctly check for an error, the host needs to _block_ so it can _synchronize_ with the device.  We can do this with the `cudaThreadSynchronize` function.

To see how all the pieces fit together, let's check for errors in our last example:

```
#include <stdio.h>
#include <stdlib.h>

__global__ void foo(int *ptr)
{
  *ptr = 7;
}

int main(void)
{
  foo<<<1,1>>>(0);

  // make the host block until the device is finished with foo
  cudaThreadSynchronize();

  // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  return 0;
}
```

This example, when compiled, produces the following output:

```
$ nvcc check_for_error.cu -o check_for_error
$ ./check_for_error
CUDA error: unspecified launch failure
```

Note that we've used the function `cudaGetErrorString` to conveniently turn the enumerated variable, `error`, into human-readable form [1](1.md). This saves us the hassle of looking up the error description in a table somewhere.

In this example, we've explicitly synchronized with the device in order to check its error status, but many CUDA Runtime functions are already implicitly synchronous and return the current error status as a result.  For example, consider the next code listing which demands an unreasonable sum of memory from `cudaMalloc`:

```
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
  int *ptr = 0;

  // gimme!
  cudaError_t error = cudaMalloc((void**)&ptr, UINT_MAX);
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  return 0;
}
```

This program outputs a different error message:

```
$ nvcc big_malloc.cu -o big_malloc
$ ./big_malloc
CUDA error: out of memory
```

Though it's important to routinely check for errors during the debugging process, you shouldn't liberally sprinkle `cudaThreadSynchronize` and `cudaGetLastError` through your code. Explicit synchronization may come with a significant performance penalty. A common idiom is to wrap synchronous error checking calls with a function which can be enabled in debug mode and elided upon release:

```
#include <stdio.h>
#include <stdlib.h>

inline void check_cuda_errors(const char *filename, const int line_number)
{
#ifdef DEBUG
  cudaThreadSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
    exit(-1);
  }
#endif
}

__global__ void foo(int *ptr)
{
  *ptr = 7;
}

int main(void)
{
  foo<<<1,1>>>(0);
  check_cuda_errors(__FILE__, __LINE__);

  return 0;
}
```

During debugging, we can compile with a macro, `DEBUG`, defined to enable our synchronous error-checking code:

```
$ nvcc -DDEBUG debug_errors.cu -o debug_errors
$ ./debug_errors
CUDA error at debug_errors.cu:25: unspecified launch failure
```

Here, we've used the built-in preprocessor variables, `__FILE__` and `__LINE__`, to easily identify where in the code the error occurred.

Expanded source code for this example is [available](http://code.google.com/p/stanford-cs193g-sp2010/source/browse/trunk/tutorials/cuda_errors.cu) in the site's code repository.

<br><font size='1'><code>[1]</code> Human-readable, but still somewhat obtuse: "unspecified launch error" usually means a bad pointer has been dereferenced, or in other words, a crash.</font>
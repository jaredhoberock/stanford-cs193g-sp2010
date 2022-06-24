# Hello, world!

In the previous section, we saw that CUDA presents a view of memory that is
partitioned into separate host and device memory spaces. Similarly, CUDA
requires the programmer to segregate code into functions which execute on
either the host or device (or both). The first kind of these functions is
called a `__global__` function. `__global__` functions, or kernels, are like the
main function from C. They are the first point of entry into a program which is
executed by the GPU device. `__global__` functions are qualified with the special
CUDA keyword, `__global__`, which is prepended to their function signature.

One important way in which `__global__` functions differ from normal C functions
is that they are executed in parallel. When we make a function call in C, a
single thread of control enters that function, does some work, and then returns
control to the calling function. Instead, `__global__` functions execute in
parallel over hundreds, thousands, or even hundreds of thousands of CUDA
threads.

To see CUDA threads in action, let's compile the following Hello, world! program.

```c++
include "util/cuPrintf.cu"
include <stdio.h>

__global__ void device_greetings(void)
{
  cuPrintf("Hello, world from the device!\n");
}

int main(void)
{
  // greet from the host
  printf("Hello, world from the host!\n");

  // initialize cuPrintf
  cudaPrintfInit();

  // launch a kernel with a single thread to greet from the device
  device_greetings<<<1,1>>>();

  // display the device's greeting
  cudaPrintfDisplay();

  // clean up after cuPrintf
  cudaPrintfEnd();

  return 0;
}
```

This program requires the non-standard files `cuPrintf.cu` and `cuPrintf.cuh`,
which are [available in this site's code repository](/util).

Let's compile this program by copying it to a file called `hello_world.cu`.
Additionally, place the files `cuPrintf.cu` and `cuPrintf.cuh` in a directory
called `util` at the same level as `hello_world.cu`. We don't need to specify a
special include directory to `nvcc` since it searches the current directory by
default. Even though our program looks like mostly normal C code, it's
important that we use `nvcc` to compile it, as it contains special CUDA
keywords and syntax that regular compilers won't recognize.

Compiling and running the program on a Linux system yields

```
$ nvcc hello_world.cu -o hello_world
$ ./hello_world
Hello, world from the host! Hello, world from the device!
```

It works, but a single greeting doesn't seem very parallel. You'll notice from
the comment in the above program that we launched a single CUDA thread using
some strange looking function call syntax. The triple chevron `<<<B,N>>>`
notation configures a kernel's launch by specifying the number `B` of thread
groups, or blocks, and the number `N` of threads per block. Let's see what
happens when change the kernel launch's configuration.

Try changing the line

```c++
device_greetings<<<1,1>>>();
```

to

```c++
device_greetings<<<10,64>>>();
```

and recompiling. You'll be greeted by `10 * 64 = 640` threads!

```
$ ./hello_world
Hello, world from the host!
Hello, world from the device!
Hello, world from the device!
Hello, world from the device!
Hello, world from the device!
Hello, world from the device!
Hello, world from the device!
Hello, world from the device!
Hello, world from the device!
Hello, world from the device!
Hello, world from the device!
Hello, world from the device!
Hello, world from the device!
Hello, world from the device!
...
```

As always, you can view the [full source code of this example](hello_world.cu)
along with additional commentary in our code repository. We'll learn more
about `__global__` functions in the next section.


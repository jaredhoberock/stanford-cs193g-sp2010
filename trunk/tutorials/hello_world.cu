#include "util/cuPrintf.cu"
#include <stdio.h>


__global__ void device_greetings(void)
{
  cuPrintf("Hello, world from the device!\n");
}


int main(void)
{
  printf("Hello, world from the host!\n");

  // initialize cuPrintf
  cudaPrintfInit();

  // launch a kernel to say hello
  device_greetings<<<1,1>>>();

  // display the device's greeting
  cudaPrintfDisplay();
  
  // clean up after cuPrintf
  cudaPrintfEnd();

  return 0;
}


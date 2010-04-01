struct event_pair
{
  cudaEvent_t start;
  cudaEvent_t end;
};


inline void check_launch(char * kernel_name)
{
  cudaThreadSynchronize();
  if(cudaGetLastError() == cudaSuccess)
  {
    printf("done with %s kernel\n",kernel_name);
  }
  else
  {
    printf("error on %s kernel\n",kernel_name);
    exit(1);
  }
}


inline void start_timer(event_pair * p)
{
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}


inline void stop_timer(event_pair * p, char * kernel_name)
{
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);
  
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  printf("%s took %.1f ms\n",kernel_name, elapsed_time);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
}


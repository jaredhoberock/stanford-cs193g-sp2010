// This is machine problem 1, part 3, page ranking
// The problem is to compute the rank of a set of webpages
// given a link graph, aka a graph where each node is a webpage,
// and each edge is a link from one page to another.
// We're going to use the Pagerank algorithm (http://en.wikipedia.org/wiki/Pagerank),
// specifically the iterative algorithm for calculating the rank of a page
// We're going to run 20 iterations of the propage step.
// Implement the corresponding code in CUDA.

/* SUBMISSION GUIDELINES:
 * You should copy your entire device_graph_iterate fuction and the 
 * supporting kernal into a file called mp1-part3-solution.cu and submit
 * that file. The fuction needs to have the exact same interface as the 
 * device_graph_iterate function we provided. The kernel is internal 
 * to your code and can look any way you want.
 */


#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <ctime>
#include <limits>

#include "mp1-util.h"

event_pair timer;

// amount of floating point numbers between answer and computed value 
// for the answer to be taken correctly. 2's complement magick.
const int maxUlps = 1000;
  
void host_graph_propagate(unsigned int *graph_indices, unsigned int *graph_edges, float *graph_nodes_in, float *graph_nodes_out, float * inv_edges_per_node, int array_length)
{
  for(int i=0; i < array_length; i++)
  {
    float sum = 0.f; 
    for(int j = graph_indices[i]; j < graph_indices[i+1]; j++)
    {
      sum += graph_nodes_in[graph_edges[j]]*inv_edges_per_node[graph_edges[j]];
    }
    graph_nodes_out[i] = 0.5f/(float)array_length + 0.5f*sum;
  }
}


void host_graph_iterate(unsigned int *graph_indices, unsigned int *graph_edges, float *graph_nodes_A, float *graph_nodes_B, float * inv_edges_per_node, int nr_iterations, int array_length)
{
  assert((nr_iterations % 2) == 0);
  for(int iter = 0; iter < nr_iterations; iter+=2)
  {
    host_graph_propagate(graph_indices, graph_edges, graph_nodes_A, graph_nodes_B, inv_edges_per_node, array_length);
    host_graph_propagate(graph_indices, graph_edges, graph_nodes_B, graph_nodes_A, inv_edges_per_node, array_length);
  }
}


// TODO your kernel code here



void device_graph_iterate(unsigned int *h_graph_indices,
                          unsigned int *h_graph_edges,
                          float *h_graph_nodes_input,
                          float *h_graph_nodes_result,
                          float *h_inv_edges_per_node,
                          int nr_iterations,
                          int num_elements,
                          int avg_edges)
{
  // TODO all of your gpu memory allocation and copying to gpu memory has to be in this function

  start_timer(&timer);

  // TODO your device code should go here, so you can measure how long it takes
  
  check_launch("gpu graph propagate");
  stop_timer(&timer,"gpu graph propagate");

  // TODO your final result should end up in h_graph_nodes_result, which is a *host* pointer
}


int main(void)
{
  // create arrays of 2M elements
  int num_elements = 1 << 21;
  int avg_edges = 8;
  int iterations = 20;
  
  // pointers to host & device arrays
  unsigned int *h_graph_indices = 0;
  float *h_inv_edges_per_node = 0;
  unsigned int *h_graph_edges = 0;
  float *h_graph_nodes_input = 0;
  float *h_graph_nodes_result = 0;
  float *h_graph_nodes_checker_A = 0;
  float *h_graph_nodes_checker_B = 0;
  
  
  // malloc host array
  // index array has to be n+1 so that the last thread can 
  // still look at its neighbor for a stopping point
  h_graph_indices = (unsigned int*)malloc((num_elements+1) * sizeof(unsigned int));
  h_inv_edges_per_node = (float*)malloc((num_elements) * sizeof(float));
  h_graph_edges = (unsigned int*)malloc(num_elements * avg_edges * sizeof(unsigned int));
  h_graph_nodes_input = (float*)malloc(num_elements * sizeof(float));
  h_graph_nodes_result = (float*)malloc(num_elements * sizeof(float));
  h_graph_nodes_checker_A = (float*)malloc(num_elements * sizeof(float));
  h_graph_nodes_checker_B = (float*)malloc(num_elements * sizeof(float));
  
  // if any memory allocation failed, report an error message
  if(h_graph_indices == 0 || h_graph_edges == 0 || h_graph_nodes_input == 0 || h_graph_nodes_result == 0 || 
	 h_inv_edges_per_node == 0 || h_graph_nodes_checker_A == 0 || h_graph_nodes_checker_B == 0)
  {
    printf("couldn't allocate memory\n");
    exit(1);
  }

  // generate random input
  // initialize
  srand(time(NULL));
   
  h_graph_indices[0] = 0;
  for(int i=0;i< num_elements;i++)
  {
    int nr_edges = (i % 15) + 1;
    h_inv_edges_per_node[i] = 1.f/(float)nr_edges;
    h_graph_indices[i+1] = h_graph_indices[i] + nr_edges;
    if(h_graph_indices[i+1] >= (num_elements * avg_edges))
    {
      printf("more edges than we have space for\n");
      exit(1);
    }
    for(int j=h_graph_indices[i];j<h_graph_indices[i+1];j++)
    {
      h_graph_edges[j] = rand() % num_elements;
    }
    
    h_graph_nodes_input[i] =  1.f/(float)num_elements;
    h_graph_nodes_checker_A[i] =  h_graph_nodes_input[i];
    h_graph_nodes_result[i] = std::numeric_limits<float>::infinity();
  }
  
  device_graph_iterate(h_graph_indices, h_graph_edges, h_graph_nodes_input, h_graph_nodes_result, h_inv_edges_per_node, iterations, num_elements, avg_edges);
  
  start_timer(&timer);
  // generate reference output
  host_graph_iterate(h_graph_indices, h_graph_edges, h_graph_nodes_checker_A, h_graph_nodes_checker_B, h_inv_edges_per_node, iterations, num_elements);
  
  check_launch("host graph propagate");
  stop_timer(&timer,"host graph propagate");
  
  // check CUDA output versus reference output
  int error = 0;
  int num_errors = 0;
  for(int i=0;i<num_elements;i++)
  {
    float n = h_graph_nodes_result[i];
    float c = h_graph_nodes_checker_A[i];
    if(!AlmostEqual2sComplement(n,c,maxUlps)) 
    {
      num_errors++;
      if (num_errors < 10)
      {
            printf("%d:%.3f::",i, n-c);
      }
      error = 1;
    }
  }
  
  if(error)
  {
    printf("Output of CUDA version and normal version didn't match! \n");
  }
  else
  {
    printf("Worked! CUDA and reference output match. \n");
  }

  // deallocate memory
  free(h_graph_indices);
  free(h_inv_edges_per_node);
  free(h_graph_edges);
  free(h_graph_nodes_input);
  free(h_graph_nodes_result);
  free(h_graph_nodes_checker_A);
  free(h_graph_nodes_checker_B);
}


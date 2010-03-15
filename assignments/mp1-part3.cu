// This is machine problem 1, part 3, page ranking
// The problem is to compute the rank of a set of webpages
// given a link graph, aka a graph where each node is a webpage,
// and each edge is a link from one page to another.
// We're going to use the Pagerank algorithm (http://en.wikipedia.org/wiki/Pagerank),
// specifically the iterative algorithm for calculating the rank of a page
// We're going to run 20 iterations of the propage step.
// Implement the corresponding host and kernel code in CUDA.
// The result of the CUDA version should end up in h_graph_nodes_A and should be equivalent
// to the output produced by the reference version.
// We'll be testing the code with randomized graph sizes, edges per node, etc., so your
// code should not hardcode any constants.


#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

 
#define EPSILON 0.0001f 

void host_graph_propagate(uint *graph_indices, uint *graph_edges, float *graph_nodes_in, float *graph_nodes_out, float * inv_edges_per_node, int array_length)
{
	for(int i=0;i<array_length;i++)
	{
		float sum = 0.f; 
		for(int j=graph_indices[i];j<graph_indices[i+1];j++)
		{
			sum += graph_nodes_in[graph_edges[j]]*inv_edges_per_node[graph_edges[j]];
		}
		graph_nodes_out[i] = 0.5f/(float)array_length + 0.5f*sum;
	}
}

void host_graph_iterate(uint *graph_indices, uint *graph_edges, float *graph_nodes_A, float *graph_nodes_B, float * inv_edges_per_node, int nr_iterations, int array_length)
{
	assert((nr_iterations % 2) == 0);
	int iter=0;
	for(;iter<nr_iterations;iter+=2)
	{
		host_graph_propagate(graph_indices, graph_edges, graph_nodes_A, graph_nodes_B, inv_edges_per_node, array_length);
		host_graph_propagate(graph_indices, graph_edges, graph_nodes_B, graph_nodes_A, inv_edges_per_node, array_length);
	}
}

// your kernel code here...


int main(void)
{
  // create arrays of 2M elements
  int num_elements = 128*128*128;
  int avg_edges = 8;
  int iterations = 20;
  
  // pointers to host & device arrays
  uint *h_graph_indices = 0;
  float *h_inv_edges_per_node = 0;
  uint *h_graph_edges = 0;
  float *h_graph_nodes_A = 0;
  float *h_graph_nodes_B = 0;
  float *h_graph_nodes_checker_A = 0;
  float *h_graph_nodes_checker_B = 0;
  
  

  // malloc host array
  // index array has to be n+1 so that the last thread can 
  // still look at its neighbor for a stopping point
  h_graph_indices = (uint*)malloc((num_elements+1) * sizeof(uint));
  h_inv_edges_per_node = (float*)malloc((num_elements) * sizeof(float));
  h_graph_edges = (uint*)malloc(num_elements * avg_edges * sizeof(uint));
  h_graph_nodes_A = (float*)malloc(num_elements * sizeof(float));
  h_graph_nodes_B = (float*)malloc(num_elements * sizeof(float));
  h_graph_nodes_checker_A = (float*)malloc(num_elements * sizeof(float));
  h_graph_nodes_checker_B = (float*)malloc(num_elements * sizeof(float));
  
  // if either memory allocation failed, report an error message
  if(h_graph_indices == 0 || h_graph_edges == 0 || h_graph_nodes_A == 0 || h_graph_nodes_B == 0 || 
	 h_inv_edges_per_node == 0 || h_graph_nodes_checker_A == 0 || h_graph_nodes_checker_B == 0)
  {
    printf("couldn't allocate memory\n");
    return 1;
  }

  // generate random input string
  // initialize
  srand(1);
   
  h_graph_indices[0] = 0;
  for(int i=0;i< num_elements;i++)
  {
	// FIXME: better randomization of number of edges
	int nr_edges = (i % 15) + 1;
	h_inv_edges_per_node[i] = 1.f/(float)nr_edges;
	h_graph_indices[i+1] = h_graph_indices[i] + nr_edges;
	if(h_graph_indices[i+1] >= (num_elements * avg_edges))
	{
		printf("more edges than we have space for\n");
		return 1;
	}
	for(int j=h_graph_indices[i];j<h_graph_indices[i+1];j++)
	{
		h_graph_edges[j] = rand() % num_elements;
	}
	// FIXME: better randomization of input vector
	h_graph_nodes_A[i] =  1.f/(float)num_elements;
	h_graph_nodes_checker_A[i] =  h_graph_nodes_A[i];
  }
  

  // the result of the computation on the gpu should end up in h_graph_nodes_A
      
  // generate reference output
  host_graph_iterate(h_graph_indices, h_graph_edges, h_graph_nodes_checker_A, h_graph_nodes_checker_B, h_inv_edges_per_node, iterations, num_elements);
  
  // check CUDA output versus reference output
  int error = 0;
  for(int i=0;i<num_elements;i++)
  {
	float n = h_graph_nodes_A[i];
	float c = h_graph_nodes_checker_A[i];
	if((n - c)*(n - c) > EPSILON) 
	{
		printf("%d:%.3f::",i,h_graph_nodes_A[i] - h_graph_nodes_checker_A[i]);
		error = 1;
	}
	
  }
  
  if(error)
  {
	printf("Output of CUDA version and normal version didn't match! \n");
  }
  else {
	printf("Worked! CUDA and reference output match. \n");
  }
 
  // deallocate memory
  free(h_graph_indices);
  free(h_inv_edges_per_node);
  free(h_graph_edges);
  free(h_graph_nodes_A);
  free(h_graph_nodes_B);
  free(h_graph_nodes_checker_A);
  free(h_graph_nodes_checker_B);
}


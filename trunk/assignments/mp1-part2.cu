// This is machine problem 1, part 2, force evaluation
//
// The problem is to take two sets of charged particles, 
// where each particle has a position and a charge associated with itself,
// and calculate the force between specific pairs of particles. 
// An index array holds the information which particle in set B should be
// paired with which particle in set A. Note that some indices will be
// out of range and you should output a force vector of zero for those particles.

#include <stdlib.h>
#include <stdio.h>

#define EPSILON 0.00001f

float4 force_calc(float4 A, float4 B) 
{
	// particles are composed of {position,charge}
	float x = B.x - A.x;
	float y = B.y - A.y;
	float z = B.z - A.z;
	float rsq = x*x + y*y + z*z;
	// avoid divide by zero
	if(rsq < EPSILON)
	{
		rsq += EPSILON;
	}
	float r = sqrt(rsq);
	float f = A.w * B.w / rsq;
	// a force vector is made up of {direction,force}
	float4 fv = make_float4(x/r,y/r,z/r,f);
	return fv;
}
 
void host_force_eval(float4 *set_A, float4 *set_B, int * indices, float4 *force_vectors, int array_length)
{
	for(int i=0;i<array_length;i++)
	{
		if(indices[i] < array_length && indices[i] >= 0)
		{
			force_vectors[i] = force_calc(set_A[i],set_B[indices[i]]);
		}
		else {
			force_vectors[i] = make_float4(0.0,0.0,0.0,0.0);
		}
	}
}


// your kernel here...


int main(void)
{
  // create arrays of 16K elements
  int num_elements = 256*256;

  // pointers to host & device arrays
  float4 *h_set_A = 0;
  float4 *h_set_B = 0;
  int *h_indices = 0;
  float4 *h_force_vectors = 0;
  float4 *h_force_vectors_checker = 0;
  
  // malloc host arrays
  h_set_A = (float4*)malloc(num_elements * sizeof(float4));
  h_set_B = (float4*)malloc(num_elements * sizeof(float4));
  h_indices = (int*)malloc(num_elements * sizeof(int));
  h_force_vectors = (float4*)malloc(num_elements * sizeof(float4));
  h_force_vectors_checker = (float4*)malloc(num_elements * sizeof(float4));
  
  // any gpu memory allocation should go here
  
  // if any memory allocation failed, report an error message
  if(h_set_A == 0 || h_set_B == 0 || h_force_vectors == 0 || h_indices == 0 || h_force_vectors_checker == 0)
  {
    printf("couldn't allocate memory\n");
    return 1;
  }

  // generate random input string
  // initialize
  srand(1);
    
  for(int i=0;i< num_elements;i++)
  {
	h_set_A[i] = make_float4(rand(),rand(),rand(),rand()); 
	h_set_B[i] = make_float4(rand(),rand(),rand(),rand());
	// some indices will be invalid
	h_indices[i] = rand() % (num_elements + 2);
	
  }
  
  // add your code for copying data to and from the GPU and running your kernel here
  // the output of your gpu kernel should end up in h_force_vectors
  
  // generate reference output
  host_force_eval(h_set_A, h_set_B, h_indices, h_force_vectors_checker, num_elements);
  
  // check CUDA output versus reference output
  int error = 0;
  for(int i=0;i<num_elements;i++)
  {
	float4 v = h_force_vectors[i];
	float4 vc = h_force_vectors_checker[i];
	if( (v.x - vc.x)*(v.x - vc.x) > EPSILON ||
		(v.y - vc.y)*(v.y - vc.y) > EPSILON ||
		(v.z - vc.z)*(v.z - vc.z) > EPSILON ||
		(v.w - vc.w)*(v.w - vc.w) > EPSILON) 
	{ 
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
  free(h_set_A);
  free(h_set_B);
  free(h_indices);
  free(h_force_vectors);
  free(h_force_vectors_checker);
  
  // don't forget to free any gpu memory
}


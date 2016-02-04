/*  random_number_generator.h
	Author: Jitesh
	Created on 04/Feb/2016
	*/
#ifndef __Rand__
#define __Rand__


//function to seed states
__global__ void init_stuff (curandState* state, unsigned long seed){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}
//function to generate random numbers using seeded states and copy them to randArray
__global__ void make_rand (curandState* state, float* randArray){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	randArray[idx] = curand_uniform(&state[idx]);
}
//kernel that generates random number from a normal distribution for mutation
__global__ void make_randNorm (curandState *state, float *randNorm, int nPredators){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > nPredators || nPredators == 0) return;
	randNorm[idx] = curand_normal(&state[idx]);
}
#endif

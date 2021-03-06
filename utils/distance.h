/*  distance.h
	Author: Jitesh
	Created on 04/Feb/2016
	*/
#ifndef __Distance__
#define __Distance__

#include <cuda_runtime.h>
#include <curand.h>
#include "cuda_vector_math.cuh"

	//periodic boundary implementation
	inline __device__ __host__ float doPeriodic(float x, float L){
		x = fmodf(x, L);
		x += L;							// this second fmodf is required because some implementations..
		x = fmodf(x, L);	// ..of fmodf use negative remainders in principle range
		return x; 
	};
	//distance correction for periodic boundary
	inline __device__ __host__ float distCorr(float D, float world){
		D = abs(D);
		D = min(D, world - D);
		return D;
	};
	//function that calculates distance between two particles/vectors
	inline __device__ __host__ float calcDist (float2 a, float2 b, float L){
		float dx, dy, dist;
		dx = a.x - b.x;
		dy = a.y - b.y;
		dx = distCorr(dx, L);
		dy = distCorr(dy, L);
		dist = powf((powf(dx, 2) + powf(dy, 2)), 0.5);
		return dist;
	};
	//function that calculates distance between two particles/vectors
	inline __device__ __host__ float NPDist(float2 a, float2 b, float L){
		float dx, dy, dist;
		dx = a.x - b.x;
		dy = a.y - b.y;
		dist = powf((powf(dx, 2) + powf(dy, 2)), 0.5);
		return dist;
	};

#endif

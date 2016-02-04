/*  union.h
	Author: Jitesh
	Created on 04/Feb/2016
	*/
#ifndef __Union__
#define __Union__


	//function that returns the root/parent of a particle
	inline __device__ __host__ int root (int i, int *id){
		while (i != id[i]){
			id[i] = id[id[i]]; //path compression
			i = id[i];
		}	
		return i;
	};

	//union function
	inline __device__ __host__ void unite (int p, int q, int *id, int *sz){
		int i = root(p, id);
		int j = root(q, id);
		if (i != j){
			if (sz[i] <= sz[j]){
				id[i] = j;
				sz[j] += sz[i];
			}
			else{
				id[j] = i;
				sz[i] += sz[j];
			}
		}	
	};
#endif

/*   kernel.cu
	Author: Jitesh
	Created on 25/July/2015
	*/

#include <cuda_runtime.h>
#include <curand.h>
#include "Swarm.h"
#include "Predator.h"
#include "../utils/random_number_generator.h"

//function that checks distance between every particle from predator and if they are with in Rd, applies repulsion
__device__ __host__ void predRepulsion(Particle *d_particles, Particle *d_particlesNew, int nParticles, Predator *d_predators, int nPredators, float L, int pid, float *d_preyPredDist, float *d_preyPredDistNP, float2 * d_sumdir){
	float w = 0;
	for (int i = 0; i < nPredators; i++){
		d_preyPredDistNP[(nParticles*i)+pid] = NPDist(d_particlesNew[pid].coord, d_predators[i].coord, L);
		d_preyPredDist[(nParticles*i)+pid] = calcDist(d_particlesNew[pid].coord, d_predators[i].coord, L);
	if (d_preyPredDist[(nParticles*i)+pid] < Particle::Rd && d_preyPredDistNP[(nParticles*i)+pid] < L/2)
		d_particlesNew[pid].dir = (1-w)*((d_particlesNew[pid].coord - d_predators[i].coord) / d_preyPredDist[(nParticles*i)+pid]) + w*d_sumdir[pid];
	else if (d_preyPredDist[(nParticles*i)+pid] < Particle::Rd && d_preyPredDistNP[(nParticles*i)+pid] >= L/2)
		d_particlesNew[pid].dir = (1-w)*((d_predators[i].coord - d_particlesNew[pid].coord) / d_preyPredDist[(nParticles*i)+pid]) + w*d_sumdir[pid];
	}	
}
//checking of attack and birth of new individual with noise drawn from existing population
__device__ __host__ void reproduceOnAttack (Particle *d_particles, Particle *d_particlesNew, int nParticles, Predator *d_predators, int nPredators, float systemSize, float* randArray, int idx, float *d_randNorm, float *d_preyPredDist, int *d_attack){
	int childId;
	float sigma = 0.05;
	for (int i = 0; i < nPredators; i++){
		d_preyPredDist[(nParticles*i)+idx] = calcDist(d_particlesNew[idx].coord, d_predators[i].coord, systemSize);
		if (d_preyPredDist[(nParticles*i)+idx] < Predator::Ra){
			d_particles[idx].coord.x = randArray[idx] * systemSize;
			d_particles[idx].coord.y = randArray[idx+1] * systemSize;
			childId = int(randArray[idx] * (nParticles-1));
			d_particlesNew[idx].eta = d_particles[childId].eta + (d_randNorm[i] * sigma); //birth with mutation
			d_attack[i] = 1;
			if (d_particles[idx].eta < 0) d_particles[idx].eta = 0.0;
		}
	}
}
//kernel that updates position and velocities to each particle
__global__ void updateKernel (Particle *d_particles, Particle *d_particlesNew, int nParticles, Predator *d_predators, int nPredators, float systemSize, float2 *d_sumdir, float *d_c, float *d_dist, float* randArray, float *d_uniteIdx, float *d_uniteIdy, float *d_randNorm, float *d_preyPredDist, float *d_preyPredDistNP, int *d_attack){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nParticles)	//checking idx to be between 0 and maximum number of particles
		return;
	d_sumdir[idx].x = 0.0; d_sumdir[idx].y = 0.0;
	d_c[idx] = 0.0;
	//call alignment kernel/function to calculate average direction of particles in vicinity
	//alignmentKernel <<<(nParticles / 256) + 1, 256 >>>(d_particles, nParticles, systemSize, idx, d_sumdir, d_c, d_dist);
	alignmentFunction (d_particles, d_particlesNew, nParticles, systemSize, idx, d_sumdir, d_c, d_dist, d_uniteIdx, d_uniteIdy);
	__syncthreads();
	predRepulsion(d_particles, d_particlesNew, nParticles, d_predators, nPredators, systemSize, idx, d_preyPredDist, d_preyPredDistNP, d_sumdir);
	//call function that manifests reproduction after an attack
	reproduceOnAttack (d_particles, d_particlesNew, nParticles, d_predators, nPredators, systemSize, randArray, idx, d_randNorm, d_preyPredDist, d_attack);
	__syncthreads();
	//call particle position and velocity updater
	updateParticle (d_particles, d_particlesNew, systemSize, randArray, idx);
}
//function to launch random number initialiser kernel
void Swarm::launchRandInit(unsigned long t){
	init_stuff <<<((nParticles - 1) / 256) + 1, 256>>> (d_state, t);
}
//function to launch random number generation and updation kernel
void Swarm::launchUpdateKernel(int nParticles, float systemSize, int nPredators){
	make_rand <<<((nParticles - 1) / 256) + 1, 256>>> (d_state, randArray);
	make_randNorm <<<((nPredators - 1) / 32) + 1,32>>> (d_state, d_randNorm, nPredators);
	cudaCopyPred();	//copy predator attribute to device
	updateKernel <<<((nParticles - 1) / 256) + 1, 256>>> (d_particles, d_particlesNew, nParticles, d_predators, nPredators, systemSize, d_sumdir, d_c, d_dist, randArray, d_uniteIdx, d_uniteIdy, d_randNorm, d_preyPredDist, d_preyPredDistNP, d_attack);
	cudaBackCopy();
	predUpdater(h_predators, nPredators, h_particles, nParticles, systemSize, h_attack);
}

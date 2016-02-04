/*   Particle.cpp
	Author: Jitesh
	Created on 25/July/2015
	*/
#include "Particle.h"
#include <math.h>

Particle::Particle(){}
//This function initialises coordinate vectors and velocitiy vectors of one particle
void Particle::init(float systemSize, float noise){
	coord.x = (1.0 * rand() / RAND_MAX) * systemSize;	
	coord.y = (1.0 * rand() / RAND_MAX) * systemSize;
	theta = (2.0 * rand() / RAND_MAX) * M_PI;
	dir.x = cos(theta);
	dir.y = sin(theta);
	vel.x = speed * dir.x;
	vel.y = speed * dir.y;
	eta = /*(1.0 * rand() / RAND_MAX) * */noise;
}

Particle::~Particle(){}

//update particle velocity and coordinates
__device__ __host__ void updateParticle (Particle *d_particles, Particle *d_particlesNew, float systemSize, float* randArray, int idx){
	//calculate theta
	d_particlesNew[idx].theta = atan2(d_particlesNew[idx].dir.y, d_particlesNew[idx].dir.x);
	//Adding noise to theta
	d_particlesNew[idx].theta = d_particlesNew[idx].theta + (d_particlesNew[idx].eta * ((randArray[idx] * 2.0) - 1) / 2.0);
	//calculate directions from theta
	d_particlesNew[idx].dir.x = cos(d_particlesNew[idx].theta);
	d_particlesNew[idx].dir.y = sin(d_particlesNew[idx].theta);
	//updating velocity of particles
	d_particlesNew[idx].vel = d_particlesNew[idx].dir * Particle::speed;
	//updating coordinates of particles
	d_particlesNew[idx].coord.x = d_particles[idx].coord.x + d_particlesNew[idx].vel.x;
	d_particlesNew[idx].coord.y = d_particles[idx].coord.y + d_particlesNew[idx].vel.y;	
	//implementing periodic boundary
	d_particlesNew[idx].coord.x = doPeriodic(d_particlesNew[idx].coord.x, systemSize);		
	d_particlesNew[idx].coord.y = doPeriodic(d_particlesNew[idx].coord.y, systemSize);
	d_particles[idx] = d_particlesNew[idx];
}

//function that checks if two particles are neighbours and align their direction if it is so.
__device__ __host__ void alignmentFunction (Particle *d_particles, Particle *d_particlesNew, int nParticles, float L, int pid, float2 * d_sumdir, float * d_c, float *d_dist, float *d_uniteIdx, float *d_uniteIdy){
	float w = 1.0;
	for (int cid = 0; cid < nParticles; cid++){
		if (cid == pid)	continue;
		//distance
		d_dist[pid*nParticles+cid] = calcDist(d_particles[pid].coord, d_particles[cid].coord, L);			
		//calculate total number of particles in the vicinity and sum their directions
		d_uniteIdx[pid*nParticles+cid] = 0.0;
		d_uniteIdy[pid*nParticles+cid] = 0.0;
		if (d_dist[pid*nParticles+cid] <= Particle::Rs){
			d_sumdir[pid].x += d_particles[cid].dir.x;
			d_sumdir[pid].y += d_particles[cid].dir.y;
			d_c[pid] = d_c[pid] + 1.0;
			d_uniteIdx[pid*nParticles+cid] = pid;
			d_uniteIdy[pid*nParticles+cid] = cid;
		}
	}
	//alignment (update direction with respect to average direction of particles in vicinity)
	d_particlesNew[pid].dir.x = (w * d_particles[pid].dir.x + d_sumdir[pid].x) / (d_c[pid] + w);
	d_particlesNew[pid].dir.y = (w * d_particles[pid].dir.y + d_sumdir[pid].y) / (d_c[pid] + w);
}


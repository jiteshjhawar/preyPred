/*   Predator.cu
	Author: Jitesh
	Created on 04/Feb/2016
	*/

#include "Swarm.h"
#include "Predator.h"
#include <math.h>

Predator::Predator(){}
//This function initialises coordinate vectors and velocitiy vectors of one particle
void Predator::init(float systemSize, float noise){
	coord.x = (1.0 * rand() / RAND_MAX) * systemSize;	
	coord.y = (1.0 * rand() / RAND_MAX) * systemSize;
	theta = (2.0 * rand() / RAND_MAX) * M_PI;
	dir.x = cos(theta);
	dir.y = sin(theta);
	vel.x = speed * dir.x;
	vel.y = speed * dir.y;
	eta = noise;
}
Predator::~Predator(){}
//function to update predator's position
__host__ __device__ void predUpdater(Predator *h_predators, int nPredators, Particle *h_particles, int nParticles, float L, int *h_attack){
	float dist1, dist2, preyDist, preyDistNP;
	float w = 0.4;
	for (int i = 0; i < nPredators; i++){
		if (h_attack[i] == 1) continue;
		int index = 0;
		for (int j = 0; j < nParticles-1; j++){
			dist1 = calcDist(h_particles[index].coord, h_predators[i].coord, L);
			dist2 = calcDist(h_particles[j+1].coord, h_predators[i].coord, L);
			if (dist1 > dist2)
				index = j+1;
		}
		preyDistNP = NPDist(h_particles[index].coord, h_predators[i].coord, L);
		preyDist = calcDist(h_particles[index].coord, h_predators[i].coord, L);
		if ( preyDistNP > L / 2 && preyDist < Predator::Rd)
			h_predators[i].dir = (1-w)*(h_predators[i].coord - h_particles[index].coord) / preyDist + w * h_predators[i].dir;
		else if ( preyDistNP <= L / 2 && preyDist < Predator::Rd)
			h_predators[i].dir = (1-w)*(h_particles[index].coord - h_predators[i].coord) / preyDist + w * h_predators[i].dir;
		h_predators[i].theta = atan2(h_predators[i].dir.y, h_predators[i].dir.x);
		h_predators[i].dir.x = cos(h_predators[i].theta);
		h_predators[i].dir.y = sin(h_predators[i].theta);
		h_predators[i].vel = (h_predators[i].dir * Predator::speed);			
		//updating coordinates of particles
		h_predators[i].coord.x += h_predators[i].vel.x;
		h_predators[i].coord.y += h_predators[i].vel.y;
		//implementing periodic boundary
		h_predators[i].coord.x = doPeriodic(h_predators[i].coord.x, L);		
		h_predators[i].coord.y = doPeriodic(h_predators[i].coord.y, L);
	}
}

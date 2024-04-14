#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <omp.h>

using namespace std;

// Particle class
class Particle {
public:
    vector<double> position;
    vector<double> velocity;
    vector<double> personalBestPos;
    double fitness;
    double personalBestFitness;

    Particle(int dim) {
        position.resize(dim);
        velocity.resize(dim);
        personalBestPos.resize(dim);
        fitness = 0.0;
        personalBestFitness = 0.0;
    }
};

// Objective function (e.g., sphere function)
double sphereFunction(const vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * xi;
    }
    return sum;
}

// PSO function
vector<double> pso(int dim, int numParticles, int maxIterations, double& elapsedTime) {
    // Initialize particles
    vector<Particle> swarm;
    for (int i = 0; i < numParticles; ++i) {
        Particle p(dim);
        for (int j = 0; j < dim; ++j) {
            p.position[j] = (rand() / (RAND_MAX + 1.0)) * 10 - 5; // Initialize position randomly between -5 and 5
            p.velocity[j] = 0.0; // Initialize velocity to 0
        }
        swarm.push_back(p);
    }

    // PSO iterations
    vector<double> globalBestPos(dim, 0.0);
    double globalBestFitness = numeric_limits<double>::max();
    double startTime = omp_get_wtime(); // Start time measurement
    #pragma omp parallel shared(swarm, globalBestPos, globalBestFitness)
    {
        vector<double> localBestPos(dim, 0.0);
        double localBestFitness = numeric_limits<double>::max();
        #pragma omp for
        for (int i = 0; i < numParticles; ++i) {
            // Evaluate fitness and update personal best
            swarm[i].fitness = sphereFunction(swarm[i].position);
            if (swarm[i].fitness < swarm[i].personalBestFitness) {
                swarm[i].personalBestPos = swarm[i].position;
                swarm[i].personalBestFitness = swarm[i].fitness;
            }
            if (swarm[i].fitness < localBestFitness) {
                #pragma omp critical
                {
                    if (swarm[i].fitness < localBestFitness) {
                        localBestPos = swarm[i].position;
                        localBestFitness = swarm[i].fitness;
                    }
                }
            }
        }
        #pragma omp critical
        {
            if (localBestFitness < globalBestFitness) {
                globalBestPos = localBestPos;
                globalBestFitness = localBestFitness;
            }
        }

        // Update velocities and positions
        #pragma omp for
        for (int i = 0; i < numParticles; ++i) {
            for (int j = 0; j < dim; ++j) {
                double r1 = (rand() / (RAND_MAX + 1.0));
                double r2 = (rand() / (RAND_MAX + 1.0));
                swarm[i].velocity[j] = 0.5 * swarm[i].velocity[j] +
                                        2.0 * r1 * (swarm[i].personalBestPos[j] - swarm[i].position[j]) +
                                        2.0 * r2 * (globalBestPos[j] - swarm[i].position[j]);
                swarm[i].position[j] += swarm[i].velocity[j];
            }
        }
    }
    double endTime = omp_get_wtime(); // End time measurement
    elapsedTime = endTime - startTime; // Calculate elapsed time

    return globalBestPos;
}

int main() {
    omp_set_num_threads(4);
    srand(time(nullptr)); // Seed random number generator

    int dim = 50; // Dimension of the problem
    int numParticles = 1000; // Number of particles in the swarm
    int maxIterations = 1000; // Maximum number of iterations
    double elapsedTime;

    vector<double> solution = pso(dim, numParticles, maxIterations, elapsedTime);

    cout << "Optimal solution found: ";
    for (double val : solution) {
        cout << val << " ";
    }
    cout << endl;

    cout << "Time taken: " << elapsedTime << " seconds" << endl;

    return 0;
}


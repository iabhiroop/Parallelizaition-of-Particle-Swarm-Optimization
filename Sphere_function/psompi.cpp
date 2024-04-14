#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <mpi.h>

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
vector<double> pso(int dim, int numParticles, int maxIterations, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

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
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Evaluate fitness in parallel
        #pragma omp parallel for
        for (int i = 0; i < numParticles; ++i) {
            swarm[i].fitness = sphereFunction(swarm[i].position);
        }

        // Update personal best and global best in parallel
        #pragma omp parallel for
        for (int i = 0; i < numParticles; ++i) {
            if (swarm[i].fitness < swarm[i].personalBestFitness || iter == 0) {
                swarm[i].personalBestPos = swarm[i].position;
                swarm[i].personalBestFitness = swarm[i].fitness;
            }
            #pragma omp critical
            {
                if (swarm[i].fitness < globalBestFitness) {
                    globalBestPos = swarm[i].position;
                    globalBestFitness = swarm[i].fitness;
                }
            }
        }

        // Update velocities and positions in parallel
        #pragma omp parallel for
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

    return globalBestPos;
}

int main(int argc, char *argv[]) {
    srand(time(nullptr)); // Seed random number generator

    MPI_Init(&argc, &argv);

    int dim = 50; // Dimension of the problem
    int numParticles = 1000; // Number of particles in the swarm
    int maxIterations = 1000; // Maximum number of iterations

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    double startTime = MPI_Wtime(); // Start time measurement

    vector<double> solution = pso(dim, numParticles, maxIterations, comm);

    double endTime = MPI_Wtime(); // End time measurement

    if (rank == 0) {
        cout << "Optimal solution found: ";
        for (double val : solution) {
            cout << val << " ";
        }
        cout << endl;

        cout << "Time taken: " << endTime - startTime << " seconds" << endl;
    }

    MPI_Finalize();

    return 0;
}


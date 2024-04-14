#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N_DIM 10
#define N_PARTICLES 100
#define MAX_ITER 100
#define INERTIA_WEIGHT 0.7
#define C1 1.5
#define C2 1.5
#define MAX_VEL 0.5
#define TOLERANCE 1e-5

// Function to optimize (Sphere function)
double sphere(double x[N_DIM]) {
    double result = 0.0;
    for (int i = 0; i < N_DIM; i++) {
        result += x[i] * x[i];
    }
    return result;
}

// Structure for a particle
typedef struct {
    double position[N_DIM];
    double velocity[N_DIM];
    double best_position[N_DIM];
    double best_fitness;
} Particle;

// Initialize a particle
void init_particle(Particle *particle) {
    for (int i = 0; i < N_DIM; i++) {
        particle->position[i] = (double)rand() / RAND_MAX * 20.0 - 10.0;
        particle->velocity[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        particle->best_position[i] = particle->position[i];
    }
    particle->best_fitness = sphere(particle->position);
}

// Update particle's velocity and position
void update_particle(Particle *particle, double global_best[N_DIM]) {
    for (int i = 0; i < N_DIM; i++) {
        double r1 = (double)rand() / RAND_MAX;
        double r2 = (double)rand() / RAND_MAX;
        
        double cognitive_component = C1 * r1 * (particle->best_position[i] - particle->position[i]);
        double social_component = C2 * r2 * (global_best[i] - particle->position[i]);
        
        particle->velocity[i] = INERTIA_WEIGHT * particle->velocity[i] + cognitive_component + social_component;
        if (particle->velocity[i] > MAX_VEL) particle->velocity[i] = MAX_VEL;
        else if (particle->velocity[i] < -MAX_VEL) particle->velocity[i] = -MAX_VEL;
        
        particle->position[i] += particle->velocity[i];
    }
    double fitness = sphere(particle->position);
    if (fitness < particle->best_fitness) {
        particle->best_fitness = fitness;
        for (int i = 0; i < N_DIM; i++) {
            particle->best_position[i] = particle->position[i];
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Particle particles[N_PARTICLES];
    double global_best[N_DIM];
    double global_best_fitness = INFINITY;

    // Initialize particles
    srand(rank); // Use rank as seed for reproducibility
    double start_time = MPI_Wtime();
    for (int i = 0; i < N_PARTICLES; i++) {
        init_particle(&particles[i]);
    }
    double init_time = MPI_Wtime() - start_time;

    // PSO optimization
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Find global best
        for (int i = 0; i < N_PARTICLES; i++) {
            if (particles[i].best_fitness < global_best_fitness) {
                global_best_fitness = particles[i].best_fitness;
                for (int j = 0; j < N_DIM; j++) {
                    global_best[j] = particles[i].best_position[j];
                }
            }
        }

        // Broadcast global best to all processes
        MPI_Bcast(global_best, N_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Update particles
        for (int i = rank; i < N_PARTICLES; i += size) {
            update_particle(&particles[i], global_best);
        }

        // Check convergence
        double min_fitness;
        MPI_Allreduce(&global_best_fitness, &min_fitness, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        if (min_fitness < TOLERANCE) break;
    }
    double optimization_time = MPI_Wtime() - start_time;

    // Print results
    if (rank == 0) {
        printf("Optimal solution found:\n");
        printf("Fitness: %lf\n", global_best_fitness);
        printf("Solution: ");
        for (int i = 0; i < N_DIM; i++) {
            printf("%lf ", global_best[i]);
        }
        printf("\n");
        printf("Initialization time: %lf seconds\n", init_time);
        printf("Optimization time: %lf seconds\n", optimization_time - init_time);
    }

    MPI_Finalize();
    return 0;
}


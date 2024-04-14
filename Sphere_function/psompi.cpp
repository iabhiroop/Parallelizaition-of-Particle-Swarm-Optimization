#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <mpi.h>

// Define the sphere function
double sphere_function(const std::vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * xi;
    }
    return sum;
}

// Generate random initial positions and velocities for particles
void initialize_particles(std::vector<std::vector<double>>& positions,
                          std::vector<std::vector<double>>& velocities,
                          int num_particles, int num_dimensions) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-5.0, 5.0); // Adjust bounds as needed

    for (int i = 0; i < num_particles; ++i) {
        std::vector<double> position(num_dimensions);
        std::vector<double> velocity(num_dimensions);
        for (int j = 0; j < num_dimensions; ++j) {
            position[j] = dis(gen);
            velocity[j] = dis(gen);
        }
        positions.push_back(position);
        velocities.push_back(velocity);
    }
}

// Evaluate objective function for particles
void evaluate_objective_function(const std::vector<std::vector<double>>& positions,
                                  std::vector<double>& objective_values) {
    objective_values.clear();
    for (const auto& position : positions) {
        double value = sphere_function(position);
        objective_values.push_back(value);
    }
}

// Update particle velocities and positions
void update_particles(std::vector<std::vector<double>>& positions,
                      std::vector<std::vector<double>>& velocities,
                      const std::vector<double>& global_best_position,
                      double c1, double c2, double w) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < positions.size(); ++i) {
        // Update velocity
        for (size_t j = 0; j < positions[i].size(); ++j) {
            double r1 = dis(gen);
            double r2 = dis(gen);
            velocities[i][j] = w * velocities[i][j] + c1 * r1 * (global_best_position[j] - positions[i][j]) + c2 * r2 * (positions[i][j] - positions[i][j]);
        }

        // Update position
        for (size_t j = 0; j < positions[i].size(); ++j) {
            positions[i][j] += velocities[i][j];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define problem parameters
    int num_particles = 1000;
    int num_dimensions = 50;
    int max_iterations = 1000;
    double c1 = 2.0;
    double c2 = 2.0;
    double w = 0.5;

    // Initialize particles
    std::vector<std::vector<double>> positions;
    std::vector<std::vector<double>> velocities;
    initialize_particles(positions, velocities, num_particles, num_dimensions);

    std::vector<double> objective_values;

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Main PSO loop
    std::vector<double> global_best_position(num_dimensions, std::numeric_limits<double>::max());
    double global_best_value = std::numeric_limits<double>::max();
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        // Evaluate objective function
        evaluate_objective_function(positions, objective_values);

        // Update particle velocities and positions
        update_particles(positions, velocities, global_best_position, c1, c2, w);

        // Find local best position and value
        double local_best_value = std::numeric_limits<double>::max();
        std::vector<double> local_best_position(num_dimensions);
        for (int i = 0; i < num_particles; ++i) {
            if (objective_values[i] < local_best_value) {
                local_best_value = objective_values[i];
                local_best_position = positions[i];
            }
        }

        // Reduce to find global best position and value
        MPI_Allreduce(&local_best_value, &global_best_value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(local_best_position.data(), global_best_position.data(), num_dimensions, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    }

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    // Print final result
    if (rank == 0) {
        std::cout << "Global Best Value: " << global_best_value << std::endl;
        std::cout << "Global Best Position:";
        for (double val : global_best_position) {
            std::cout << " " << val;
        }
        std::cout << std::endl;
        std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}

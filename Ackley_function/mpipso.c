#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <sys/time.h>

#define PI 3.14159265358979323846

double nDimensions, mVelocity, nIterations, seed, nParticles;
double x_min = -32.768;
double x_max = 32.768;

double ackley(double x[], double nDimensions);

double ackley(double x[], double nDimensions) {
    double c = 2 * M_PI;
    double b = 0.2;
    double a = 20;
    double sum1 = 0;
    double sum2 = 0;
    int i;
    for (i = 0; i < nDimensions; i++) {
        sum1 = sum1 + gsl_pow_2(x[i]);
        sum2 = sum2 + cos(c * x[i]);
    }
    double term1 = -a * exp(-b * sqrt(sum1 / nDimensions));
    double term2 = -exp(sum2 / nDimensions);
    return term1 + term2 + a + M_E;
}

int main(int argc, char *argv[]) {
    int i, j;
    double a, b;
    double c1, c2, rho1, rho2, w, fit;

    if (nDimensions == 0) nDimensions = 50;
    if (nParticles == 0) nParticles = 1000;
    if (mVelocity == 0) mVelocity = 60;
    if (nIterations == 0) nIterations = 100;
    if (seed == 0) seed = 1;

    struct timeval TimeValue_Start, TimeValue_Final;
    struct timezone TimeZone_Start, TimeZone_Final;
    long time_start, time_end;
    double time_overhead;

    gettimeofday(&TimeValue_Start, &TimeZone_Start);

    int size, myrank, distributed_particles;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) {
        distributed_particles = (int)nParticles / size;
    }
    MPI_Bcast(&distributed_particles, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (myrank == 0) {
        distributed_particles += (int)nParticles % size;
    }

    double result[distributed_particles];
    int step;
    c1 = c2 = 1.496;
    w = 0.7298;

    double recievingdata[((int)nDimensions + 1) * size];
    double sendingdata[(int)nDimensions + 1];

    gsl_rng_env_setup();
    gsl_rng *r = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(r, time(0));

    double positions[distributed_particles][(int)nDimensions];
    double velocities[distributed_particles][(int)nDimensions];
    double pBestPositions[distributed_particles][(int)nDimensions];
    double pBestFitness[distributed_particles];
    double gBestPosition[(int)nDimensions];
    double gBestFitness = DBL_MAX;

    // Particle initialization
    for (i = 0; i < distributed_particles; i++) {
        for (j = 0; j < nDimensions; j++) {
            a = x_min + (x_max - x_min) * gsl_rng_uniform(r);
            b = x_min + (x_max - x_min) * gsl_rng_uniform(r);
            positions[i][j] = a;
            pBestPositions[i][j] = a;
            velocities[i][j] = (a - b) / 2.;
        }
        pBestFitness[i] = ackley(positions[i], nDimensions);
        if (pBestFitness[i] < gBestFitness) {
            memmove((void *)gBestPosition, (void *)&positions[i], sizeof(double) * nDimensions);
            gBestFitness = pBestFitness[i];
        }
    }

    // Actual calculation
    for (step = 0; step < nIterations; step++) {
        for (i = 0; i < distributed_particles; i++) {
            for (j = 0; j < nDimensions; j++) {
                rho1 = c1 * gsl_rng_uniform(r);
                rho2 = c2 * gsl_rng_uniform(r);
                velocities[i][j] = w * velocities[i][j] + \
                    rho1 * (pBestPositions[i][j] - positions[i][j]) + \
                    rho2 * (gBestPosition[j] - positions[i][j]);
                positions[i][j] += velocities[i][j];

                if (positions[i][j] < x_min) {
                    positions[i][j] = x_min;
                    velocities[i][j] = 0;
                } else if (positions[i][j] > x_max) {
                    positions[i][j] = x_max;
                    velocities[i][j] = 0;
                }
            }

            fit = ackley(positions[i], nDimensions);

            if (fit < pBestFitness[i]) {
                pBestFitness[i] = fit;
                memmove((void *)&pBestPositions[i], (void *)&positions[i],
                        sizeof(double) * nDimensions);
            }

            if (fit < gBestFitness) {
                gBestFitness = fit;
                memmove((void *)gBestPosition, (void *)&positions[i],
                        sizeof(double) * nDimensions);
            }
        }

        for (int k = 0; k < (nDimensions); k++)
            sendingdata[k] = gBestPosition[k];
        sendingdata[(int)nDimensions] = gBestFitness;
        MPI_Gather(&sendingdata, nDimensions + 1, MPI_DOUBLE, &recievingdata, nDimensions + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (myrank == 0) {
            double min = gBestFitness;
            int pos = -1;
            for (int k = 0; k < size; k++) {
                if (min >= recievingdata[k * ((int)nDimensions + 1) + ((int)nDimensions)]) {
                    min = recievingdata[k * ((int)nDimensions + 1) + ((int)nDimensions)];
                    pos = k * ((int)nDimensions + 1);
                }
            }
            gBestFitness = min;
            for (int k = pos; k < nDimensions + pos; k++)
                gBestPosition[k - pos] = recievingdata[k];
        }
        MPI_Bcast(&gBestPosition, nDimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (myrank == 0) {
        printf("Result: %f\n", gBestFitness);
        gettimeofday(&TimeValue_Final, &TimeZone_Final);
        time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
        time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
        time_overhead = (time_end - time_start) / 1000000.0;
        printf("\nTime in Seconds (T): %lf\n", time_overhead);
    }

    gsl_rng_free(r);
    MPI_Finalize();
    return 0;
}


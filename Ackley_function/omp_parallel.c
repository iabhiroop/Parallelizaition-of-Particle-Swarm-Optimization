#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_math.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

#define PI 3.14159265358979323846

double ackley(double x[], double nDimensions) {
    double c = 2*M_PI;
    double b = 0.2;
    double a = 20;
    double sum1 = 0;
    double sum2 = 0;
    int i;
    for (i=0; i<nDimensions; i++) {
        sum1 = sum1 + gsl_pow_2(x[i]);
        sum2 = sum2 + cos(c*x[i]);
    }
    double term1 = -a * exp(-b*sqrt(sum1/nDimensions));
    double term2 = -exp(sum2/nDimensions);
    return term1 + term2 + a + M_E;
}


int main(int argc, char *argv[]) {
    double nDimensions = 50; // Default number of dimensions
    double nParticles = 1000; // Default number of particles
    double mVelocity = 60; // Default maximum velocity
    double nIterations = 100; // Default number of iterations
    double seed = 1; // Default seed value

    struct timeval TimeValue_Start, TimeValue_Final;
    double time_overhead;

    gettimeofday(&TimeValue_Start, NULL);

    int size, myrank, distributed_particles = nParticles;
    double result[(int)distributed_particles];
    double a, b;
    double c1, c2, rho1, rho2, w, fit;
    c1 = c2 = 1.496;
    w = 0.7298;

    gsl_rng_env_setup();
    gsl_rng *r = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(r, time(0));

    double positions[(int)distributed_particles][(int)nDimensions];
    double velocities[(int)distributed_particles][(int)nDimensions];
    double pBestPositions[(int)distributed_particles][(int)nDimensions];
    double pBestFitness[(int)distributed_particles];
    double gBestPosition[(int)nDimensions];
    double gBestFitness = DBL_MAX;
    int min;

    #pragma omp parallel for private(a, b) reduction(min:gBestFitness)
    for (int i = 0; i < distributed_particles; i++) {
        for (int j = 0; j < (int)nDimensions; j++) {
            a = -32.768 + (32.768 - (-32.768)) * gsl_rng_uniform(r);
            b = -32.768 + (32.768 - (-32.768)) * gsl_rng_uniform(r);
            positions[i][j] = a;
            pBestPositions[i][j] = a;
            velocities[i][j] = (a - b) / 2.;
        }
        pBestFitness[i] = ackley(positions[i], nDimensions);
        if (pBestFitness[i] < gBestFitness) {
            gBestFitness = pBestFitness[i];
            memcpy(gBestPosition, positions[i], sizeof(double) * nDimensions);
        }
    }

    for (int step = 0; step < nIterations; step++) {
        #pragma omp parallel num_threads(4) shared(min)
        {
            #pragma omp for private(a, b) 
            for (int i = 0; i < distributed_particles; i++) {
                for (int j = 0; j < nDimensions; j++) {
                    rho1 = c1 * gsl_rng_uniform(r);
                    rho2 = c2 * gsl_rng_uniform(r);
                    velocities[i][j] = w * velocities[i][j] +
                                       rho1 * (pBestPositions[i][j] - positions[i][j]) +
                                       rho2 * (gBestPosition[j] - positions[i][j]);
                    positions[i][j] += velocities[i][j];
                    if (positions[i][j] < -32.768) {
                        positions[i][j] = -32.768;
                        velocities[i][j] = 0;
                    } else if (positions[i][j] > 32.768) {
                        positions[i][j] = 32.768;
                        velocities[i][j] = 0;
                    }
                }

                fit = ackley(positions[i], nDimensions);
                if (fit < pBestFitness[i]) {
                    pBestFitness[i] = fit;
                    memcpy(pBestPositions[i], positions[i], sizeof(double) * nDimensions);
                }
            }

            #pragma omp for reduction(min:gBestFitness)
            for (int i = 0; i < (int)nParticles; i++) {
                if (pBestFitness[i] < gBestFitness) {
                    gBestFitness = pBestFitness[i];
                }
            }

            #pragma omp for  
            for (int i = 0; i < (int)nParticles; i++) {
                if (gBestFitness == pBestFitness[i])
                    min = i;  
            }
        }
        memcpy(gBestPosition, pBestPositions[min], sizeof(double) * nDimensions);
    }

    gsl_rng_free(r);
    
    printf("Result: %f\n", gBestFitness);
    gettimeofday(&TimeValue_Final, NULL);
    time_overhead = (TimeValue_Final.tv_sec - TimeValue_Start.tv_sec) + 
                    1e-6 * (TimeValue_Final.tv_usec - TimeValue_Start.tv_usec);
    printf("\n Time in Seconds (T) : %lf\n", time_overhead);

    return 0;
}


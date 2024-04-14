#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_math.h>
#include <string.h>
#include <time.h>
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

void calculate(double nParticles, double result[]);

int main(int argc, char *argv[]) {
    double nParticles = 1000; // Default number of particles
    double nDimensions = 50; // Default number of dimensions
    double mVelocity = 60; // Default maximum velocity
    double nIterations = 100; // Default number of iterations
    double seed = 1; // Default seed value

    // Argument handling
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "-D") == 0)
            nDimensions = strtol(argv[i + 1], NULL, 10);
        else if (strcmp(argv[i], "-m") == 0)
            nParticles = strtol(argv[i + 1], NULL, 10);
        else if (strcmp(argv[i], "-V") == 0)
            mVelocity = strtol(argv[i + 1], NULL, 10);
        else if (strcmp(argv[i], "-i") == 0)
            nIterations = strtol(argv[i + 1], NULL, 10);
        else if (strcmp(argv[i], "-s") == 0)
            seed = strtol(argv[i + 1], NULL, 10);
    }

    struct timeval TimeValue_Start, TimeValue_Final;
    double time_overhead;

    gettimeofday(&TimeValue_Start, NULL);
    double result[(int)nParticles];
    int j, step;
    double a, b;
    double c1 = 1.496, c2 = 1.496, rho1, rho2, w = 0.7298, fit;
    gsl_rng_env_setup();
    gsl_rng *r = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(r, time(0));

    double positions[(int)nParticles][(int)nDimensions];
    double velocities[(int)nParticles][(int)nDimensions];
    double pBestPositions[(int)nParticles][(int)nDimensions];
    double pBestFitness[(int)nParticles];
    double gBestPosition[(int)nDimensions];
    double gBestFitness = DBL_MAX;

    for (int i = 0; i < nParticles; i++) {
        for (j = 0; j < nDimensions; j++) {
            a = -32.768 + (32.768 - (-32.768)) * gsl_rng_uniform(r);
            b = -32.768 + (32.768 - (-32.768)) * gsl_rng_uniform(r);
            positions[i][j] = a;
            pBestPositions[i][j] = a;
            velocities[i][j] = (a - b) / 2.;
        }
        pBestFitness[i] = ackley(positions[i], nDimensions);
        if (pBestFitness[i] < gBestFitness) {
            memcpy(gBestPosition, positions[i], sizeof(double) * nDimensions);
            gBestFitness = pBestFitness[i];
        }
    }

    for (step = 0; step < nIterations; step++) {
        for (int i = 0; i < nParticles; i++) {
            for (j = 0; j < nDimensions; j++) {
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
            if (fit < gBestFitness) {
                gBestFitness = fit;
                memcpy(gBestPosition, positions[i], sizeof(double) * nDimensions);
            }
        }
    }

    printf("Result: %f\n", gBestFitness);
    gettimeofday(&TimeValue_Final, NULL);
    time_overhead = (TimeValue_Final.tv_sec - TimeValue_Start.tv_sec) + 
                    1e-6 * (TimeValue_Final.tv_usec - TimeValue_Start.tv_usec);
    printf("\n Time in Seconds (T) : %lf\n", time_overhead);

    gsl_rng_free(r);
}


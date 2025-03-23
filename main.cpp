#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <chrono> // timer

using namespace std;
using namespace Eigen;

// Function to apply periodic boundary conditions
double periodic(double dl, double bound) {
    return dl - bound * round(dl / bound);
}

// Fuel function
double fuel_f(double n, double k) {
    return -k * n;
}

// Gamma^(1/2) matrix computation
MatrixXd gamma_half_matrix(double theta, double v0, double zeta, double D1, double c) {
    double sqrt_zeta = sqrt(zeta);
    double sqrt_D1 = sqrt(D1);
    double term1 = (1 + c * zeta - sqrt(1 - 2 * c * zeta + (c * c + v0 * v0) * zeta * zeta)) / (2 * zeta);
    double term2 = (1 + c * zeta + sqrt(1 - 2 * c * zeta + (c * c + v0 * v0) * zeta * zeta)) / (2 * zeta);

    MatrixXd D_half(4, 4);
    D_half << 1 / sqrt_zeta, 0, 0, 0,
            0, sqrt(term1), 0, 0,
            0, 0, sqrt(term2), 0,
            0, 0, 0, sqrt_D1;

    double cos_theta = cos(theta);
    double sin_theta = sin(theta);

    MatrixXd P(4, 4);
    P << -sin_theta, v0 * zeta * cos_theta * cos_theta / (-1 + c * zeta - sqrt(1 - 2 * c * zeta + (c * c + v0 * v0) * zeta * zeta)),
            v0 * zeta * cos_theta * cos_theta / (-1 + c * zeta + sqrt(1 - 2 * c * zeta + (c * c + v0 * v0) * zeta * zeta)), 0,
            cos_theta, v0 * zeta * cos_theta * sin_theta / (-1 + c * zeta - sqrt(1 - 2 * c * zeta + (c * c + v0 * v0) * zeta * zeta)),
            v0 * zeta * cos_theta * sin_theta / (-1 + c * zeta + sqrt(1 - 2 * c * zeta + (c * c + v0 * v0) * zeta * zeta)), 0,
            0, 0, 0, cos_theta,
            0, cos_theta, cos_theta, 0;

    MatrixXd P_inv(4, 4);
    P_inv << -sin_theta, cos_theta, 0, 0,
            -v0 * zeta / (2 * sqrt(c * c * zeta * zeta - 2 * c * zeta + v0 * v0 * zeta * zeta + 1)),
            -v0 * zeta * tan(theta) / (2 * sqrt(c * c * zeta * zeta - 2 * c * zeta + v0 * v0 * zeta * zeta + 1)),
            0,
            (-c * zeta + sqrt(c * c * zeta * zeta - 2 * c * zeta + v0 * v0 * zeta * zeta + 1) + 1) / (2 * sqrt(c * c * zeta * zeta - 2 * c * zeta + v0 * v0 * zeta * zeta + 1) * cos_theta),
            v0 * zeta / (2 * sqrt(c * c * zeta * zeta - 2 * c * zeta + v0 * v0 * zeta * zeta + 1)),
            v0 * zeta * tan(theta) / (2 * sqrt(c * c * zeta * zeta - 2 * c * zeta + v0 * v0 * zeta * zeta + 1)),
            0,
            (c * zeta + sqrt(c * c * zeta * zeta - 2 * c * zeta + v0 * v0 * zeta * zeta + 1) - 1) / (2 * sqrt(c * c * zeta * zeta - 2 * c * zeta + v0 * v0 * zeta * zeta + 1) * cos_theta),
            0, 0, 1 / cos_theta, 0;

    return P * D_half * P_inv;
}

// Main SDE integrator function
void integrator_SDE_fuel(int T, int N, double dt, double v0, double D1, double D2, double D3, double zeta_theta, double J0, double R, double L, double zeta, double c, double k, int dump) {
    // Initialize random number generator
    random_device rd;
    //mt19937 gen(rd());
    std::mt19937 gen(10000); // fixed random seed

    normal_distribution<> noise_dist(0.0, 1.0);

    // Particle properties
    vector<double> x(N), y(N), phi(N), n(N, 200.0);
    for (int i = 0; i < N; ++i) {
        x[i] = uniform_real_distribution<>(-L, L)(gen);
        y[i] = uniform_real_distribution<>(-L, L)(gen);
        phi[i] = uniform_real_distribution<>(0, 2 * M_PI)(gen);
    }

    ofstream dump_file("output.dump");

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < T; ++t) {
        vector<double> x_new(N), y_new(N), phi_new(N), n_new(N);

        for (int i = 0; i < N; ++i) {
            double force_sum = 0.0;
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    double dx = periodic(x[i] - x[j], 2 * L);
                    double dy = periodic(y[i] - y[j], 2 * L);
                    double dist = sqrt(dx * dx + dy * dy);
                    if (dist < R) {
                        force_sum += J0 * sin(phi[i] - phi[j]);
                    }
                }
            }

            MatrixXd gamma_half = gamma_half_matrix(phi[i], v0, zeta, D1, c);
            VectorXd random_noises(4);
            for (int k0 = 0; k0 < 4; ++k0) random_noises[k0] = noise_dist(gen);

            VectorXd correlated_noises = gamma_half * random_noises;

            double noise_theta = correlated_noises[2];
            double theta = (-1.0 / zeta_theta) * force_sum * dt + sqrt(2.0 * D1 * dt) * noise_theta;
            phi_new[i] = phi[i] + theta;

            double noise_n = correlated_noises[3];
            n_new[i] = n[i] + fuel_f(n[i], k) * dt + sqrt(2.0 * D2 * dt) * noise_n;

            double noise_r_x = correlated_noises[0];
            double noise_r_y = correlated_noises[1];
            x_new[i] = x[i] - fuel_f(n[i], k) * v0 * cos(phi_new[i]) * dt + sqrt(2.0 * D3 * dt) * noise_r_x;
            y_new[i] = y[i] - fuel_f(n[i], k) * v0 * sin(phi_new[i]) * dt + sqrt(2.0 * D3 * dt) * noise_r_y;

            x_new[i] = periodic(x_new[i], 2 * L);
            y_new[i] = periodic(y_new[i], 2 * L);
        }

        if (t % dump == 0) {
            dump_file << "ITEM: TIMESTEP\n";
            dump_file << t << "\nITEM: NUMBER OF ATOMS\n" << N << "\nITEM: BOX BOUNDS pp pp pp\n";
            dump_file << -L << " " << L << "\n";
            dump_file << -L << " " << L << "\n";
            dump_file << -L << " " << L << "\n";
            dump_file << "ITEM: ATOMS id x y phi n\n";
            for (int i = 0; i < N; ++i) {
                dump_file << i + 1 << " " << x_new[i] << " " << y_new[i] << " " << phi_new[i] << " " << n_new[i] << "\n";
            }

            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
            std::cout << "Step " << t << ", Elapsed time: " << elapsed << " seconds\n";

        }

        x = x_new;
        y = y_new;
        phi = phi_new;
        n = n_new;
    }

    dump_file.close();
}

int main() {
    // Parameters
    double dt = 0.01;
    int T = 20000;
    int N = 40;
    double v0 = 0.24;
    double D1 = 0.1;
    double D2 = 1.0 / 8.0;
    double D3 = 1.0 / 800.0;
    double zeta_theta = 200.0;
    double zeta = 8.0;
    double J0 = 8.75;
    double R = 1.0;
    double L = 3.1 / 2.0;
    double c = 1.0 / 8.0;
    double k = 0.03;
    int dump = static_cast<int>(round(1.0 / dt));

    integrator_SDE_fuel(T, N, dt, v0, D1, D2, D3, zeta_theta, J0, R, L, zeta, c, k, dump);

    return 0;
}

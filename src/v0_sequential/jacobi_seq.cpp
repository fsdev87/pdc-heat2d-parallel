// ============================================================
// V0 — Sequential Jacobi Solver for 2D Heat Equation
// CS-3006 PDC Project, FAST-NUCES Spring 2026
// ============================================================
//
// PROBLEM:
//   Solve the 2D Laplace (steady-state heat) equation:
//       d²u/dx² + d²u/dy² = 0
//   on a unit square domain [0,1] x [0,1]
//
// BOUNDARY CONDITIONS:
//   u(x, 0) = sin(pi*x)           (bottom edge)
//   u(x, 1) = sin(pi*x)*exp(-pi)  (top edge)
//   u(0, y) = 0                   (left edge)
//   u(1, y) = 0                   (right edge)
//
// ANALYTICAL SOLUTION (used for correctness verification):
//   u(x, y) = sin(pi*x) * exp(-pi*y)
//
// JACOBI METHOD:
//   Discretize the grid into (N+2)x(N+2) points.
//   Boundary cells (outer ring) are fixed to boundary values.
//   Interior cells (1..N x 1..N) are updated each iteration as:
//       u_new[i][j] = (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1]) / 4
//   Two grids are used: u_old (read) and u_new (write).
//   After each iteration, swap pointers.
//   Stop when max change across all interior cells < tolerance.
//
// GRID LAYOUT:
//   Full grid is (N+2) x (N+2), stored as flat 1D array (row-major).
//   Access: grid[i * (N+2) + j]
//   Row 0    = top boundary    (y=1)
//   Row N+1  = bottom boundary (y=0)
//   Col 0    = left boundary   (x=0)
//   Col N+1  = right boundary  (x=1)
//   Interior = rows 1..N, cols 1..N
//
// NOTE ON COORDINATES:
//   x = col_index / (N+1)    ranges from 0 to 1
//   y = row_index / (N+1)    ranges from 0 to 1
//   BUT: row 0 is the TOP (y=1), row N+1 is the BOTTOM (y=0)
//   So: y = 1 - (row_index / (N+1))
//   This matches the boundary condition setup below.
//
// HOW TO COMPILE:
//   g++ -O2 -std=c++17 -o jacobi_seq jacobi_seq.cpp -lm
//
// HOW TO RUN:
//   ./jacobi_seq <grid_size> [tolerance] [fixed_iters]
//   Example: ./jacobi_seq 256
//   Example: ./jacobi_seq 1024 1e-7
//   Example: ./jacobi_seq 1024 1e-7 5000  (fixed iteration mode for benchmarking)
//
// EXECUTION MODES:
//   Tolerance mode (default):  run until max_delta < tol
//   Fixed iteration mode:      run exactly N iterations (for benchmarking)
//   Use fixed mode for all speedup experiments — guarantees identical
//   workload across all process counts.
// ============================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <string>

// ---- Constants ----
const double PI      = M_PI;
const int    MAX_ITER = 2000000;

// ---- Grid access macro (row-major flat array) ----
// Grid dimensions are (N+2) x (N+2)
// i = row index (0 to N+1), j = col index (0 to N+1)
#define IDX(i, j) ((i) * (N + 2) + (j))

// ============================================================
// set_boundary
//   Sets the fixed boundary values on the outer ring of the grid.
//   Called once before the iteration loop. Never called again.
//
//   Bottom boundary (row N+1): u = sin(pi*x)
//   Top boundary    (row 0):   u = sin(pi*x) * exp(-pi)
//   Left boundary   (col 0):   u = 0
//   Right boundary  (col N+1): u = 0
// ============================================================
void set_boundary(std::vector<double>& u, int N) {
    // Bottom boundary row (row N+1) — hottest edge
    for (int j = 0; j <= N + 1; j++) {
        double x = (double)j / (N + 1);
        u[IDX(N + 1, j)] = sin(PI * x);
    }

    // Top boundary row (row 0) — cooler edge
    for (int j = 0; j <= N + 1; j++) {
        double x = (double)j / (N + 1);
        u[IDX(0, j)] = sin(PI * x) * exp(-PI);
    }

    // Left boundary column (col 0) — zero
    for (int i = 0; i <= N + 1; i++) {
        u[IDX(i, 0)] = 0.0;
    }

    // Right boundary column (col N+1) — zero
    for (int i = 0; i <= N + 1; i++) {
        u[IDX(i, N + 1)] = 0.0;
    }
}

// ============================================================
// compute_max_error
//   After convergence, compare numerical solution against the
//   known analytical solution: u(x,y) = sin(pi*x) * exp(-pi*y)
//   Returns the maximum absolute error across all interior cells.
// ============================================================
double compute_max_error(const std::vector<double>& u, int N) {
    double max_err = 0.0;
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            // y increases downward in grid indexing
            // row 0 = top = y=1, row N+1 = bottom = y=0
            double x = (double)j / (N + 1);
            double y = 1.0 - (double)i / (double)(N + 1);
            double analytical = sin(PI * x) * exp(-PI * y);
            double err = std::abs(u[IDX(i, j)] - analytical);
            max_err = std::max(max_err, err);
        }
    }
    return max_err;
}

// ============================================================
// jacobi_solve
//   Main solver loop. Uses two grids (u and u_new).
//   Each iteration:
//     1. For every interior cell, compute average of 4 neighbors
//        from u (old values) and store in u_new
//     2. Track max change (for convergence check)
//     3. Swap u and u_new (pointer swap, no copy)
//     4. If max change < tolerance, stop (or if fixed_iters reached)
//
//   Two modes:
//     fixed_iters < 0: tolerance mode (stop when max_delta < tol)
//     fixed_iters > 0: fixed iteration mode (stop exactly at fixed_iters)
//   Returns number of iterations taken.
// ============================================================
int jacobi_solve(std::vector<double>& u, int N, double tol, int fixed_iters) {
    // Allocate second grid for new values
    std::vector<double> u_new(u.size(), 0.0);

    // Copy boundary values into u_new too (boundaries never change
    // but u_new needs them so boundary neighbors read correctly)
    set_boundary(u_new, N);

    int iter = 0;
    double max_delta = 1.0; // initialize above tolerance
    bool use_fixed = (fixed_iters > 0);

    while (iter < MAX_ITER) {
        max_delta = 0.0;
        iter++;

        // Update all interior cells
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                // 5-point stencil: average of north, south, east, west
                double val = (u[IDX(i-1, j)] +   // north (row above)
                              u[IDX(i+1, j)] +   // south (row below)
                              u[IDX(i, j-1)] +   // west  (col left)
                              u[IDX(i, j+1)]     // east  (col right)
                             ) * 0.25;

                u_new[IDX(i, j)] = val;

                // Track maximum change for convergence check
                double delta = std::abs(val - u[IDX(i, j)]);
                max_delta = std::max(max_delta, delta);
            }
        }

        // Swap grids: u_new becomes u_old for next iteration
        // std::swap just swaps the internal pointers — O(1), no data copy
        std::swap(u, u_new);

        // Progress output every 1000 iterations
        if (iter % 1000 == 0) {
            if (use_fixed)
                std::cout << "  iter " << iter << " / " << fixed_iters << std::endl;
            else
                std::cout << "  iter " << iter
                          << "  max_delta = " << std::scientific << max_delta
                          << std::endl;
        }

        // Check stopping condition
        if (use_fixed) {
            if (iter >= fixed_iters) break;  // Fixed mode: stop at exact count
        } else {
            if (max_delta < tol) break;      // Tolerance mode: stop when converged
        }
    }

    return iter;
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
    // ---- Parse arguments ----
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <grid_size> [tolerance] [fixed_iters]" << std::endl;
        std::cerr << "Example: " << argv[0] << " 256 1e-4" << std::endl;
        std::cerr << "Example: " << argv[0] << " 1024 1e-7 5000  (fixed iteration mode)" << std::endl;
        return 1;
    }

    int    N           = std::stoi(argv[1]);
    double tol         = (argc >= 3) ? std::stod(argv[2]) : 1e-7;
    int    fixed_iters = (argc >= 4) ? std::stoi(argv[3]) : -1;  // -1 = tolerance mode

    std::cout << "======================================" << std::endl;
    std::cout << "V0 Sequential Jacobi Solver" << std::endl;
    std::cout << "Grid size  : " << N << " x " << N << std::endl;
    if (fixed_iters > 0) {
        std::cout << "Mode       : fixed " << fixed_iters << " iterations" << std::endl;
    } else {
        std::cout << "Tolerance  : " << tol << std::endl;
    }
    std::cout << "Max iter   : " << MAX_ITER << std::endl;
    std::cout << "======================================" << std::endl;

    // ---- Allocate grid: (N+2) x (N+2) flat array ----
    // Extra +2 for boundary ring. All initialized to 0.
    std::vector<double> u((N + 2) * (N + 2), 0.0);

    // ---- Set boundary conditions ----
    set_boundary(u, N);

    // ---- Start timer ----
    auto t_start = std::chrono::high_resolution_clock::now();

    // ---- Run solver ----
    int iterations = jacobi_solve(u, N, tol, fixed_iters);

    // ---- Stop timer ----
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // ---- Compute error vs analytical solution ----
    double max_error = compute_max_error(u, N);

    // ---- Print results ----
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "======================================" << std::endl;
    std::cout << "Iterations : " << iterations << std::endl;
    std::cout << "Time       : " << elapsed << " seconds" << std::endl;
    std::cout << "Max error  : " << max_error << " (vs analytical)" << std::endl;
    std::cout << "======================================" << std::endl;

    // ---- CSV output for experiment scripts ----
    // Format: version, N, procs, threads, time, iterations, max_error
    std::cout << "CSV: v0," << N << ",1,1,"
              << elapsed << ","
              << iterations << ","
              << max_error << std::endl;

    return 0;
}
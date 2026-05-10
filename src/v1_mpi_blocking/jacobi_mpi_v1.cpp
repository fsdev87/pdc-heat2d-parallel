// ============================================================
// V1 — MPI Parallel Jacobi Solver (Blocking Communication)
// CS-3006 PDC Project, FAST-NUCES Spring 2026
// ============================================================
//
// WHAT THIS VERSION DOES:
//   Parallelizes the Jacobi solver using MPI with 1D horizontal
//   strip decomposition. The NxN grid is divided into horizontal
//   strips — one strip per MPI process. Each process owns
//   (N / num_procs) rows plus 2 ghost rows.
//
// GHOST ROWS:
//   After each iteration, each process needs the bottom row of
//   its upper neighbor and the top row of its lower neighbor to
//   compute its own edge rows. These borrowed rows are called
//   ghost rows. Stored at local index 0 (ghost from above) and
//   local_rows+1 (ghost from below).
//
// LOCAL GRID LAYOUT:
//   Row 0              = ghost from above (or top boundary for rank 0)
//   Row 1..local_rows  = owned interior rows
//   Row local_rows+1   = ghost from below (or bottom boundary for last rank)
//   Col 0              = left boundary (always 0)
//   Col N+1            = right boundary (always 0)
//
// GLOBAL TO LOCAL ROW MAPPING:
//   base_rows  = N / num_procs
//   row_offset = rank * base_rows
//   local row i -> global row (row_offset + i)
//   y coordinate = 1.0 - (row_offset + i) / (N+1)
//
// BLOCKING COMMUNICATION:
//   MPI_Sendrecv simultaneously sends and receives — deadlock-free.
//   Processes BLOCK (wait idle) until both send and receive complete
//   before continuing. This idle wait is what V2 fixes.
//
// TWO MODES:
//   Tolerance mode (default):  run until max_delta < tol
//   Fixed iteration mode:      run exactly N iterations (for benchmarking)
//   Use fixed mode for all speedup experiments — guarantees identical
//   workload across all process counts, no MPI_Allreduce needed.
//
// HOW TO COMPILE:
//   mpicxx -O2 -std=c++17 -o jacobi_v1 jacobi_mpi_v1.cpp -lm
//   OR: make build VERSION=v1
//
// HOW TO RUN:
//   mpirun -np 4 ./jacobi_v1 <N> [tol] [fixed_iters]
//
//   Correctness test (tolerance mode):
//     mpirun -np 4 ./jacobi_v1 256 1e-7
//
//   Performance test (fixed iteration mode):
//     mpirun -np 1 ./jacobi_v1 1024 1e-7 5000
//     mpirun -np 2 ./jacobi_v1 1024 1e-7 5000
//     mpirun -np 4 ./jacobi_v1 1024 1e-7 5000
//     mpirun -np 8 ./jacobi_v1 1024 1e-7 5000
// ============================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <mpi/mpi.h>
#include <iomanip>
#include <algorithm>
#include <string>

// ---- Constants ----
const double PI       = M_PI;
const int    MAX_ITER = 500000;

// ---- Grid access macro ----
// Local grid is (local_rows + 2) x (N + 2), stored flat row-major
// i = row index (0 = ghost/top-boundary, local_rows+1 = ghost/bottom-boundary)
// j = col index (0 = left boundary, N+1 = right boundary)
#define IDX(i, j) ((i) * (N + 2) + (j))

// ============================================================
// set_boundary_local
//   Sets fixed boundary values in the local grid.
//
//   Rank 0 only:        row 0 = top boundary = sin(πx)*exp(-π)
//   Last rank only:     row local_rows+1 = bottom boundary = sin(πx)
//   All ranks:          col 0 = 0 (left), col N+1 = 0 (right)
//
//   Middle ranks' row 0 and row local_rows+1 are ghost rows —
//   they are NOT set here, they come from MPI exchange.
// ============================================================
void set_boundary_local(std::vector<double>& u, int N,
                        int local_rows, int rank, int num_procs) {
    // Top boundary — only rank 0 owns this (y=1)
    if (rank == 0) {
        for (int j = 0; j <= N + 1; j++) {
            double x = (double)j / (N + 1);
            u[IDX(0, j)] = sin(PI * x) * exp(-PI);
        }
    }

    // Bottom boundary — only last rank owns this (y=0)
    if (rank == num_procs - 1) {
        for (int j = 0; j <= N + 1; j++) {
            double x = (double)j / (N + 1);
            u[IDX(local_rows + 1, j)] = sin(PI * x);
        }
    }

    // Left boundary — all ranks (x=0)
    for (int i = 0; i <= local_rows + 1; i++)
        u[IDX(i, 0)] = 0.0;

    // Right boundary — all ranks (x=1)
    for (int i = 0; i <= local_rows + 1; i++)
        u[IDX(i, N + 1)] = 0.0;
}

// ============================================================
// exchange_ghost_rows
//   Exchanges boundary rows with neighboring MPI processes.
//   Uses MPI_Sendrecv (blocking, deadlock-safe).
//
//   Exchange 1: send my row 1 up to 'above',
//               receive into my row 0 from 'above'
//   Exchange 2: send my row local_rows down to 'below',
//               receive into my row local_rows+1 from 'below'
//
//   MPI_PROC_NULL: rank 0 has no 'above', last rank has no 'below'.
//   Sendrecv with MPI_PROC_NULL is a safe no-op — the ghost row
//   already holds the correct boundary value from set_boundary_local.
// ============================================================
void exchange_ghost_rows(std::vector<double>& u, int N,
                         int local_rows, int above, int below) {
    // Send row 1 upward, receive ghost row from above into row 0
    MPI_Sendrecv(
        &u[IDX(1, 0)],           N + 2, MPI_DOUBLE, above, 0,
        &u[IDX(0, 0)],           N + 2, MPI_DOUBLE, above, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // Send last interior row downward, receive ghost row from below
    MPI_Sendrecv(
        &u[IDX(local_rows, 0)],     N + 2, MPI_DOUBLE, below, 1,
        &u[IDX(local_rows + 1, 0)], N + 2, MPI_DOUBLE, below, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
}

// ============================================================
// compute_max_error_local
//   Compares this rank's local interior rows against the
//   analytical solution: u(x,y) = sin(πx) * exp(-πy)
//
//   Global row mapping:
//     row_offset = rank * base_rows
//     local row i -> global row (row_offset + i)
//     y = 1.0 - global_row / (N+1)
//
//   Returns max absolute error across all local interior cells.
// ============================================================
double compute_max_error_local(const std::vector<double>& u, int N,
                                int local_rows, int rank, int num_procs) {
    int base_rows  = N / num_procs;
    int row_offset = rank * base_rows;

    double max_err = 0.0;
    for (int i = 1; i <= local_rows; i++) {
        int global_row = row_offset + i;                          // FIXED: was (i-1)
        double y = 1.0 - (double)global_row / (double)(N + 1);
        for (int j = 1; j <= N; j++) {
            double x = (double)j / (N + 1);
            double analytical = sin(PI * x) * exp(-PI * y);
            double err = std::abs(u[IDX(i, j)] - analytical);
            max_err = std::max(max_err, err);
        }
    }
    return max_err;
}

// ============================================================
// jacobi_solve_mpi
//   Main MPI Jacobi solver loop.
//
//   Two modes controlled by fixed_iters:
//     fixed_iters == -1 : tolerance mode
//       - runs until max global delta < tol
//       - uses MPI_Allreduce every iteration for global check
//       - use this for correctness verification only
//
//     fixed_iters > 0   : fixed iteration mode
//       - runs exactly fixed_iters iterations, ignores tol
//       - NO MPI_Allreduce (removes one global barrier per iter)
//       - use this for ALL performance/speedup experiments
//       - guarantees identical workload across all process counts
//
//   Returns number of iterations completed.
// ============================================================
int jacobi_solve_mpi(std::vector<double>& u, int N, double tol,
                     int fixed_iters, int rank, int num_procs) {

    int local_rows = N / num_procs;
    int above = (rank == 0)             ? MPI_PROC_NULL : rank - 1;
    int below = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    // Allocate second grid. Set boundaries in both.
    std::vector<double> u_new((local_rows + 2) * (N + 2), 0.0);
    set_boundary_local(u,     N, local_rows, rank, num_procs);
    set_boundary_local(u_new, N, local_rows, rank, num_procs);

    int    iter        = 0;
    double global_delta = 1.0;
    bool   use_fixed   = (fixed_iters > 0);

    while (iter < MAX_ITER) {
        iter++;

        // --- Step 1: Exchange ghost rows (BLOCKING) ---
        // Every process sends its edge rows and receives neighbors' edge rows.
        // Blocks until both send and receive complete.
        exchange_ghost_rows(u, N, local_rows, above, below);

        // --- Step 2: Compute all local interior cells ---
        double local_delta = 0.0;
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= N; j++) {
                double val = (u[IDX(i-1, j)] +   // north
                              u[IDX(i+1, j)] +   // south
                              u[IDX(i, j-1)] +   // west
                              u[IDX(i, j+1)]     // east
                             ) * 0.25;

                u_new[IDX(i, j)] = val;

                double delta = std::abs(val - u[IDX(i, j)]);
                if (delta > local_delta) local_delta = delta;
            }
        }

        // --- Step 3: Swap grids ---
        std::swap(u, u_new);
        // Boundaries are preserved: both grids were initialized with
        // set_boundary_local and the compute loop never touches boundary indices.

        // --- Step 4: Check stopping condition ---
        if (use_fixed) {
            // Fixed iteration mode: stop at exact count, no Allreduce needed
            if (iter >= fixed_iters) break;
        } else {
            // Tolerance mode: global convergence check via Allreduce
            MPI_Allreduce(&local_delta, &global_delta, 1,
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (global_delta < tol) break;
        }

        // Progress output every 5000 iterations (rank 0 only)
        if (rank == 0 && iter % 5000 == 0) {
            if (use_fixed)
                std::cout << "  iter " << iter << " / " << fixed_iters << std::endl;
            else
                std::cout << "  iter " << iter
                          << "  global_delta = " << std::scientific << global_delta
                          << std::endl;
        }
    }

    return iter;
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // ---- Parse arguments ----
    int    N           = 256;
    double tol         = 1e-7;
    int    fixed_iters = -1;   // -1 = tolerance mode, >0 = fixed iteration mode

    if (rank == 0) {
        if (argc < 2) {
            std::cerr << "Usage: mpirun -np P " << argv[0]
                      << " <N> [tol] [fixed_iters]" << std::endl;
            std::cerr << "  Correctness: mpirun -np 4 " << argv[0] << " 256 1e-7" << std::endl;
            std::cerr << "  Benchmark:   mpirun -np 4 " << argv[0] << " 1024 1e-7 5000" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        N           = std::stoi(argv[1]);
        tol         = (argc >= 3) ? std::stod(argv[2]) : 1e-7;
        fixed_iters = (argc >= 4) ? std::stoi(argv[3]) : -1;

        if (N % num_procs != 0) {
            std::cerr << "Error: N (" << N << ") must be divisible by num_procs ("
                      << num_procs << ")" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast all parameters to all ranks
    MPI_Bcast(&N,           1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&tol,         1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fixed_iters, 1, MPI_INT,    0, MPI_COMM_WORLD);

    int local_rows = N / num_procs;

    if (rank == 0) {
        std::cout << "======================================" << std::endl;
        std::cout << "V1 MPI Blocking Jacobi Solver" << std::endl;
        std::cout << "Grid size   : " << N << " x " << N << std::endl;
        std::cout << "Processes   : " << num_procs << std::endl;
        std::cout << "Rows/rank   : " << local_rows << std::endl;
        if (fixed_iters > 0)
            std::cout << "Mode        : fixed " << fixed_iters << " iterations" << std::endl;
        else
            std::cout << "Mode        : tolerance " << tol << std::endl;
        std::cout << "======================================" << std::endl;
    }

    // ---- Allocate local grid ----
    std::vector<double> u((local_rows + 2) * (N + 2), 0.0);
    set_boundary_local(u, N, local_rows, rank, num_procs);

    // ---- Run solver ----
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    int iterations = jacobi_solve_mpi(u, N, tol, fixed_iters, rank, num_procs);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t_start;

    // ---- Compute global max error vs analytical solution ----
    double local_err  = compute_max_error_local(u, N, local_rows, rank, num_procs);
    double global_err = 0.0;
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---- Print results (rank 0 only) ----
    if (rank == 0) {
        std::string mode = (fixed_iters > 0) ? "fixed-iters" : "tolerance";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "======================================" << std::endl;
        std::cout << "Mode        : " << mode << std::endl;
        std::cout << "Iterations  : " << iterations << std::endl;
        std::cout << "Time        : " << elapsed << " seconds" << std::endl;
        std::cout << "Max error   : " << global_err << " (vs analytical)" << std::endl;
        std::cout << "======================================" << std::endl;

        // CSV: version,N,num_procs,num_threads,time,iterations,max_error
        std::cout << "CSV: v1," << N << "," << num_procs << ",1,"
                  << elapsed << ","
                  << iterations << ","
                  << global_err << std::endl;
    }

    MPI_Finalize();
    return 0;
}
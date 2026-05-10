// ============================================================
// V4 — Hybrid MPI + OpenMP Jacobi Solver
// CS-3006 PDC Project, FAST-NUCES Spring 2026
// ============================================================
//
// WHAT THIS VERSION ADDS OVER V2:
//   V1-V3 use only MPI — each MPI rank is a single thread.
//   Multi-core CPUs have multiple cores per socket. Pure MPI
//   with one rank per core means lots of MPI processes and lots
//   of MPI communication overhead.
//
//   V4 uses a hybrid approach:
//     - Fewer MPI ranks (one per socket/node, e.g. 2 ranks for 2 machines)
//     - Each rank spawns multiple OpenMP threads to compute its block in parallel
//     - Threads share memory within a rank — no MPI overhead inside
//     - MPI only used for inter-rank communication (ghost rows)
//
//   This reduces MPI communication overhead while exploiting all cores.
//
// OPENMP ADDITION:
//   The inner Jacobi loop is parallelized with:
//     #pragma omp parallel for collapse(2) reduction(max:local_delta) schedule(static)
//
//   collapse(2):     flattens both i and j loops into one parallel region
//   reduction(max):  each thread tracks its own local_delta, OpenMP takes max
//   schedule(static): equal chunk distribution, good for uniform workload
//
// HOW TO COMPILE:
//   mpicxx -O2 -std=c++17 -fopenmp -o jacobi_v4 jacobi_hybrid_v4.cpp -lm
//   OR: make v4
//
// HOW TO RUN:
//   # 2 MPI ranks x 2 threads each (total 4 cores)
//   OMP_NUM_THREADS=2 mpirun -np 2 ./jacobi_v4 1024 1e-6 5000
//
//   # 1 MPI rank x 4 threads (all local, no MPI comm)
//   OMP_NUM_THREADS=4 mpirun -np 1 ./jacobi_v4 1024 1e-6 5000
//
//   # 4 MPI ranks x 1 thread (same as pure MPI, for comparison)
//   OMP_NUM_THREADS=1 mpirun -np 4 ./jacobi_v4 1024 1e-6 5000
//
// EXPERIMENT CONFIGURATIONS (all using 4 total cores):
//   NP=1, THREADS=4  -> max thread sharing, zero MPI comm
//   NP=2, THREADS=2  -> balanced hybrid
//   NP=4, THREADS=1  -> pure MPI equivalent
// ============================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <mpi/mpi.h>
#include <omp.h>
#include <iomanip>
#include <algorithm>
#include <string>

const double PI       = M_PI;
const int    MAX_ITER = 300000;

#define IDX(i, j) ((i) * (N + 2) + (j))

void set_boundary_local(std::vector<double>& u, int N,
                        int local_rows, int rank, int num_procs) {
    if (rank == 0)
        for (int j = 0; j <= N + 1; j++) {
            double x = (double)j / (N + 1);
            u[IDX(0, j)] = sin(PI * x) * exp(-PI);
        }

    if (rank == num_procs - 1)
        for (int j = 0; j <= N + 1; j++) {
            double x = (double)j / (N + 1);
            u[IDX(local_rows + 1, j)] = sin(PI * x);
        }

    for (int i = 0; i <= local_rows + 1; i++) {
        u[IDX(i, 0)]     = 0.0;
        u[IDX(i, N + 1)] = 0.0;
    }
}

double compute_max_error_local(const std::vector<double>& u, int N,
                                int local_rows, int rank, int num_procs) {
    int row_offset = rank * (N / num_procs);
    double max_err = 0.0;
    for (int i = 1; i <= local_rows; i++) {
        int global_row = row_offset + i;
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

int jacobi_solve_hybrid(std::vector<double>& u, int N, double tol,
                         int fixed_iters, int rank, int num_procs, int num_threads) {

    int local_rows = N / num_procs;
    int above = (rank == 0)             ? MPI_PROC_NULL : rank - 1;
    int below = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    std::vector<double> u_new((local_rows + 2) * (N + 2), 0.0);
    set_boundary_local(u,     N, local_rows, rank, num_procs);
    set_boundary_local(u_new, N, local_rows, rank, num_procs);

    int    iter        = 0;
    double global_delta = 1.0;
    bool   use_fixed   = (fixed_iters > 0);

    while (iter < MAX_ITER) {
        iter++;
        double local_delta = 0.0;

        // --- Step 1: Ghost row exchange (MPI, blocking) ---
        MPI_Sendrecv(&u[IDX(1, 0)],           N + 2, MPI_DOUBLE, above, 0,
                     &u[IDX(0, 0)],           N + 2, MPI_DOUBLE, above, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&u[IDX(local_rows, 0)],     N + 2, MPI_DOUBLE, below, 1,
                     &u[IDX(local_rows + 1, 0)], N + 2, MPI_DOUBLE, below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // --- Step 2: Compute all interior cells with OpenMP threads ---
        // collapse(2): parallelize both i and j loops as one flat loop
        // reduction(max:local_delta): each thread computes its own max, OpenMP combines
        // schedule(static): divide iterations evenly among threads upfront
        #pragma omp parallel for collapse(2) reduction(max:local_delta) schedule(static) num_threads(num_threads)
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= N; j++) {
                double val = (u[IDX(i-1, j)] +
                              u[IDX(i+1, j)] +
                              u[IDX(i, j-1)] +
                              u[IDX(i, j+1)]) * 0.25;
                u_new[IDX(i, j)] = val;
                double delta = std::abs(val - u[IDX(i, j)]);
                if (delta > local_delta) local_delta = delta;
            }
        }

        // --- Step 3: Swap grids ---
        std::swap(u, u_new);

        // --- Step 4: Stopping condition ---
        if (use_fixed) {
            if (iter >= fixed_iters) break;
        } else {
            MPI_Allreduce(&local_delta, &global_delta, 1,
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (global_delta < tol) break;
        }

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

int main(int argc, char* argv[]) {
    // MPI_Init_thread: request thread support since MPI + OpenMP are both active
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int    N           = 256;
    double tol         = 1e-6;
    int    fixed_iters = -1;

    if (rank == 0) {
        if (argc < 2) {
            std::cerr << "Usage: OMP_NUM_THREADS=T mpirun -np P " << argv[0]
                      << " <N> [tol] [fixed_iters]" << std::endl;
            std::cerr << "Example: OMP_NUM_THREADS=2 mpirun -np 2 " << argv[0]
                      << " 1024 1e-6 5000" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        N           = std::stoi(argv[1]);
        tol         = (argc >= 3) ? std::stod(argv[2]) : 1e-6;
        fixed_iters = (argc >= 4) ? std::stoi(argv[3]) : -1;
        if (N % num_procs != 0) {
            std::cerr << "Error: N must be divisible by num_procs" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&N,           1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&tol,         1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fixed_iters, 1, MPI_INT,    0, MPI_COMM_WORLD);

    // Get number of OpenMP threads from OMP_NUM_THREADS env var
    int num_threads = omp_get_max_threads();
    int local_rows  = N / num_procs;

    if (rank == 0) {
        std::cout << "======================================" << std::endl;
        std::cout << "V4 Hybrid MPI+OpenMP Jacobi Solver" << std::endl;
        std::cout << "Grid size   : " << N << " x " << N << std::endl;
        std::cout << "MPI ranks   : " << num_procs << std::endl;
        std::cout << "OMP threads : " << num_threads << std::endl;
        std::cout << "Total cores : " << num_procs * num_threads << std::endl;
        std::cout << "Rows/rank   : " << local_rows << std::endl;
        if (fixed_iters > 0)
            std::cout << "Mode        : fixed " << fixed_iters << " iterations" << std::endl;
        else
            std::cout << "Mode        : tolerance " << tol << std::endl;
        std::cout << "======================================" << std::endl;
    }

    std::vector<double> u((local_rows + 2) * (N + 2), 0.0);
    set_boundary_local(u, N, local_rows, rank, num_procs);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    int iterations = jacobi_solve_hybrid(u, N, tol, fixed_iters,
                                          rank, num_procs, num_threads);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t_start;

    double local_err  = compute_max_error_local(u, N, local_rows, rank, num_procs);
    double global_err = 0.0;
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "======================================" << std::endl;
        std::cout << "Iterations  : " << iterations << std::endl;
        std::cout << "Time        : " << elapsed << " seconds" << std::endl;
        std::cout << "Max error   : " << global_err << " (vs analytical)" << std::endl;
        std::cout << "======================================" << std::endl;
        // CSV: version,N,num_procs,num_threads,time,iterations,max_error
        std::cout << "CSV: v4," << N << "," << num_procs << "," << num_threads << ","
                  << elapsed << "," << iterations << "," << global_err << std::endl;
    }

    MPI_Finalize();
    return 0;
}

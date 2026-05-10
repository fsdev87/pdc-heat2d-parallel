// ============================================================
// V2 — MPI Non-Blocking Jacobi Solver (Computation-Communication Overlap)
// CS-3006 PDC Project, FAST-NUCES Spring 2026
// ============================================================
//
// WHAT THIS VERSION ADDS OVER V1:
//   V1 uses blocking MPI_Sendrecv — every process sends ghost rows
//   and then WAITS IDLE until the transfer completes before computing.
//   That idle wait time is wasted.
//
//   V2 uses non-blocking MPI_Isend/MPI_Irecv to OVERLAP communication
//   with computation:
//     1. Post non-blocking sends and receives for ghost rows (returns immediately)
//     2. While ghost data is in transit, compute ALL INTERIOR rows
//        (rows 2 to local_rows-1 — these don't touch ghost rows at all)
//     3. Call MPI_Waitall to wait for ghost data to arrive
//     4. Now compute BOUNDARY rows (rows 1 and local_rows) which need ghost data
//
//   Result: communication latency is hidden behind useful computation.
//   The total wall time shrinks because compute and communication overlap.
//
// WHY INTERIOR ROWS CAN BE COMPUTED WITHOUT GHOST DATA:
//   Row 1 needs ghost row 0 (from above) — cannot compute early
//   Rows 2..local_rows-1 only need rows 1..local_rows — fully local, compute now
//   Row local_rows needs ghost row local_rows+1 (from below) — cannot compute early
//
// EDGE CASE:
//   If local_rows <= 2, there are no interior rows to overlap with.
//   We fall back to computing everything after Waitall.
//
// HOW TO COMPILE:
//   mpicxx -O2 -std=c++17 -o jacobi_v2 jacobi_mpi_v2.cpp -lm
//   OR: make v2
//
// HOW TO RUN:
//   Correctness: mpirun -np 4 ./jacobi_v2 256 1e-7
//   Benchmark:   mpirun -np 4 ./jacobi_v2 1024 1e-7 5000
// ============================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <mpi/mpi.h>
#include <iomanip>
#include <algorithm>
#include <string>

const double PI       = M_PI;
const int    MAX_ITER = 500000;

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

// ============================================================
// compute_row: compute one interior row of the stencil
// ============================================================
inline void compute_row(const std::vector<double>& u,
                        std::vector<double>& u_new,
                        int row, int N, double& local_delta) {
    for (int j = 1; j <= N; j++) {
        double val = (u[IDX(row-1, j)] +
                      u[IDX(row+1, j)] +
                      u[IDX(row, j-1)] +
                      u[IDX(row, j+1)]) * 0.25;
        u_new[IDX(row, j)] = val;
        double delta = std::abs(val - u[IDX(row, j)]);
        if (delta > local_delta) local_delta = delta;
    }
}

int jacobi_solve_mpi(std::vector<double>& u, int N, double tol,
                     int fixed_iters, int rank, int num_procs) {

    int local_rows = N / num_procs;
    int above = (rank == 0)             ? MPI_PROC_NULL : rank - 1;
    int below = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    std::vector<double> u_new((local_rows + 2) * (N + 2), 0.0);
    set_boundary_local(u,     N, local_rows, rank, num_procs);
    set_boundary_local(u_new, N, local_rows, rank, num_procs);

    int    iter       = 0;
    double global_delta = 1.0;
    bool   use_fixed  = (fixed_iters > 0);
    bool   can_overlap = (local_rows > 2);  // need at least 3 rows to overlap

    MPI_Request reqs[4];

    while (iter < MAX_ITER) {
        iter++;
        double local_delta = 0.0;

        // --- Step 1: Post non-blocking receives FIRST (always before sends) ---
        MPI_Irecv(&u[IDX(0, 0)],            N + 2, MPI_DOUBLE, above, 1, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&u[IDX(local_rows+1, 0)], N + 2, MPI_DOUBLE, below, 0, MPI_COMM_WORLD, &reqs[1]);

        // --- Step 2: Post non-blocking sends ---
        MPI_Isend(&u[IDX(1, 0)],           N + 2, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(&u[IDX(local_rows, 0)],  N + 2, MPI_DOUBLE, below, 1, MPI_COMM_WORLD, &reqs[3]);

        // --- Step 3: OVERLAP — compute interior rows while comms in flight ---
        // Interior rows = rows 2 to local_rows-1 (don't need ghost data)
        if (can_overlap) {
            for (int i = 2; i <= local_rows - 1; i++)
                compute_row(u, u_new, i, N, local_delta);
        }

        // --- Step 4: Wait for ghost data ---
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // --- Step 5: Now compute boundary rows that need ghost data ---
        compute_row(u, u_new, 1,          N, local_delta);
        compute_row(u, u_new, local_rows, N, local_delta);

        // If no overlap (small local_rows), we already computed all interior above
        // after Waitall so rows 1 and local_rows cover everything for local_rows==2
        // For local_rows==1, row 1 == local_rows so it's computed once (fine)

        // --- Step 6: Swap grids ---
        std::swap(u, u_new);

        // --- Step 7: Stopping condition ---
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
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int    N           = 256;
    double tol         = 1e-7;
    int    fixed_iters = -1;

    if (rank == 0) {
        if (argc < 2) {
            std::cerr << "Usage: mpirun -np P " << argv[0]
                      << " <N> [tol] [fixed_iters]" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        N           = std::stoi(argv[1]);
        tol         = (argc >= 3) ? std::stod(argv[2]) : 1e-7;
        fixed_iters = (argc >= 4) ? std::stoi(argv[3]) : -1;
        if (N % num_procs != 0) {
            std::cerr << "Error: N must be divisible by num_procs" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&N,           1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&tol,         1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fixed_iters, 1, MPI_INT,    0, MPI_COMM_WORLD);

    int local_rows = N / num_procs;

    if (rank == 0) {
        std::cout << "======================================" << std::endl;
        std::cout << "V2 MPI Non-Blocking Jacobi Solver" << std::endl;
        std::cout << "Grid size   : " << N << " x " << N << std::endl;
        std::cout << "Processes   : " << num_procs << std::endl;
        std::cout << "Rows/rank   : " << local_rows << std::endl;
        std::cout << "Overlap     : " << (local_rows > 2 ? "YES" : "NO (too few rows)") << std::endl;
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

    int iterations = jacobi_solve_mpi(u, N, tol, fixed_iters, rank, num_procs);

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
        std::cout << "CSV: v2," << N << "," << num_procs << ",1,"
                  << elapsed << "," << iterations << "," << global_err << std::endl;
    }

    MPI_Finalize();
    return 0;
}

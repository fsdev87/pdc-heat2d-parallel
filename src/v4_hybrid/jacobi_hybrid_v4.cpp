// ============================================================
// V4 — Hybrid MPI + OpenMP with 2D Decomposition & Non-blocking MPI
// CS-3006 PDC Project, FAST-NUCES Spring 2026
// ============================================================
//
// WHAT THIS VERSION ADDS OVER V3:
//   V3 uses 2D decomposition with non-blocking MPI, but each MPI
//   rank runs as a single thread. On multi-core CPUs, this wastes
//   intra-node parallelism potential.
//
//   V4 builds on V3 by adding OpenMP thread-level parallelism:
//     - 2D Cartesian decomposition (reduces communication volume at scale)
//     - Non-blocking MPI with computation-communication overlap
//     - Each MPI rank spawns multiple OpenMP threads for its local block
//     - Reduction combines thread-level deltas and MPI-level deltas
//
// ARCHITECTURE:
//   - Fewer MPI ranks (one per node/socket)
//   - Each rank has 2D block of the grid
//   - Threads within a rank compute in parallel
//   - MPI handles 4-directional ghost exchange with overlap
//
// HOW TO COMPILE:
//   mpicxx -O2 -std=c++17 -fopenmp -o jacobi_v4 jacobi_hybrid_v4.cpp -lm
//   OR: make v4
//
// HOW TO RUN:
//   # 2 MPI ranks (2x1 or 1x2 grid), 2 threads each
//   OMP_NUM_THREADS=2 mpirun -np 2 ./jacobi_v4 1024 1e-7 5000
//
//   # 4 MPI ranks (2x2 grid), 1 thread each (equivalent to V3)
//   OMP_NUM_THREADS=1 mpirun -np 4 ./jacobi_v4 1024 1e-7 5000
//
//   # 1 MPI rank, 4 threads (pure OpenMP, no MPI comm)
//   OMP_NUM_THREADS=4 mpirun -np 1 ./jacobi_v4 1024 1e-7 5000
// ============================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <mpi/mpi.h>
#include <omp.h>
#include <iomanip>
#include <algorithm>

const double PI       = M_PI;
const int    MAX_ITER = 500000;

// Local grid width = local_cols + 2 (including ghost columns)
#define IDX(i, j) ((i) * (local_cols + 2) + (j))

// ============================================================
// set_boundary_local_2d
//   Sets boundary values for this process's 2D block.
// ============================================================
void set_boundary_local_2d(std::vector<double>& u,
                           int local_rows, int local_cols,
                           int row_offset, int col_offset, int N,
                           bool on_top, bool on_bottom,
                           bool on_left, bool on_right) {
    // Top boundary row (ghost row 0)
    // Only iterate over interior columns, not ghost columns
    if (on_top) {
        for (int j = 1; j <= local_cols; j++) {
            int global_col = col_offset + j;
            double x = (double)global_col / (N + 1);
            u[IDX(0, j)] = sin(PI * x) * exp(-PI);
        }
    }

    // Bottom boundary row
    // Only iterate over interior columns, not ghost columns
    if (on_bottom) {
        for (int j = 1; j <= local_cols; j++) {
            int global_col = col_offset + j;
            double x = (double)global_col / (N + 1);
            u[IDX(local_rows + 1, j)] = sin(PI * x);
        }
    }

    // Left boundary column
    if (on_left)
        for (int i = 0; i <= local_rows + 1; i++)
            u[IDX(i, 0)] = 0.0;

    // Right boundary column
    if (on_right)
        for (int i = 0; i <= local_rows + 1; i++)
            u[IDX(i, local_cols + 1)] = 0.0;
}

double compute_max_error_local_2d(const std::vector<double>& u,
                                   int local_rows, int local_cols,
                                   int row_offset, int col_offset, int N) {
    double max_err = 0.0;
    #pragma omp parallel for reduction(max:max_err) collapse(2) schedule(static)
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= local_cols; j++) {
            int global_row = row_offset + i;
            int global_col = col_offset + j;
            double y = 1.0 - (double)global_row / (double)(N + 1);
            double x = (double)global_col / (N + 1);
            double analytical = sin(PI * x) * exp(-PI * y);
            double err = std::abs(u[IDX(i, j)] - analytical);
            max_err = std::max(max_err, err);
        }
    }
    return max_err;
}

int jacobi_solve_hybrid_2d(std::vector<double>& u, int N, double tol,
                             int fixed_iters, MPI_Comm cart_comm,
                             int local_rows, int local_cols,
                             int north, int south, int west, int east,
                             MPI_Datatype col_type, int num_threads) {
    int rank;
    MPI_Comm_rank(cart_comm, &rank);

    std::vector<double> u_new(u.size(), 0.0);

    int    iter        = 0;
    double global_delta = 1.0;
    bool   use_fixed   = (fixed_iters > 0);

    while (iter < MAX_ITER) {
        iter++;
        double local_delta = 0.0;

        // --- Non-blocking ghost exchange: 4 directions ---
        MPI_Request reqs[8];

        // Post receives FIRST
        MPI_Irecv(&u[IDX(0, 0)],              local_cols + 2, MPI_DOUBLE, north, 1, cart_comm, &reqs[0]);
        MPI_Irecv(&u[IDX(local_rows + 1, 0)], local_cols + 2, MPI_DOUBLE, south, 0, cart_comm, &reqs[1]);
        MPI_Irecv(&u[IDX(1, 0)],              1, col_type, west,  3, cart_comm, &reqs[2]);
        MPI_Irecv(&u[IDX(1, local_cols + 1)], 1, col_type, east,  2, cart_comm, &reqs[3]);

        // Post sends
        MPI_Isend(&u[IDX(1, 0)],           local_cols + 2, MPI_DOUBLE, north, 0, cart_comm, &reqs[4]);
        MPI_Isend(&u[IDX(local_rows, 0)],  local_cols + 2, MPI_DOUBLE, south, 1, cart_comm, &reqs[5]);
        MPI_Isend(&u[IDX(1, 1)],           1, col_type, west,  2, cart_comm, &reqs[6]);
        MPI_Isend(&u[IDX(1, local_cols)],  1, col_type, east,  3, cart_comm, &reqs[7]);

        // OVERLAP: compute interior cells with OpenMP (don't depend on ghost data)
        #pragma omp parallel for reduction(max:local_delta) collapse(2) schedule(static) num_threads(num_threads)
        for (int i = 2; i <= local_rows - 1; i++) {
            for (int j = 2; j <= local_cols - 1; j++) {
                double val = (u[IDX(i-1, j)] +
                              u[IDX(i+1, j)] +
                              u[IDX(i, j-1)] +
                              u[IDX(i, j+1)]) * 0.25;
                u_new[IDX(i, j)] = val;
                double delta = std::abs(val - u[IDX(i, j)]);
                local_delta = std::max(local_delta, delta);
            }
        }

        // Wait for communication
        MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

        // Compute edge cells that needed ghost data (can still use OpenMP)
        #pragma omp parallel for reduction(max:local_delta) schedule(static) num_threads(num_threads)
        for (int idx = 0; idx < (2 * local_cols + 2 * local_rows - 4); idx++) {
            int i, j;
            // Map flat index to edge position
            if (idx < local_cols) {
                // Top edge (row 1, cols 1..local_cols)
                i = 1; j = idx + 1;
            } else if (idx < 2 * local_cols) {
                // Bottom edge (row local_rows, cols 1..local_cols)
                i = local_rows; j = idx - local_cols + 1;
            } else if (idx < 2 * local_cols + local_rows - 2) {
                // Left edge (rows 2..local_rows-1, col 1), skip corners
                i = idx - 2 * local_cols + 2; j = 1;
            } else {
                // Right edge (rows 2..local_rows-1, col local_cols), skip corners
                i = idx - 2 * local_cols - local_rows + 4; j = local_cols;
            }
            double val = (u[IDX(i-1, j)] +
                          u[IDX(i+1, j)] +
                          u[IDX(i, j-1)] +
                          u[IDX(i, j+1)]) * 0.25;
            u_new[IDX(i, j)] = val;
            double delta = std::abs(val - u[IDX(i, j)]);
            local_delta = std::max(local_delta, delta);
        }

        std::swap(u, u_new);

        // Preserve boundary values
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int j = 0; j <= local_cols + 1; j++) {
            u[IDX(0, j)] = u_new[IDX(0, j)];
            u[IDX(local_rows + 1, j)] = u_new[IDX(local_rows + 1, j)];
        }
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int i = 0; i <= local_rows + 1; i++) {
            u[IDX(i, 0)] = u_new[IDX(i, 0)];
            u[IDX(i, local_cols + 1)] = u_new[IDX(i, local_cols + 1)];
        }

        if (use_fixed) {
            if (iter >= fixed_iters) break;
        } else {
            MPI_Allreduce(&local_delta, &global_delta, 1,
                          MPI_DOUBLE, MPI_MAX, cart_comm);
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int    N           = 256;
    double tol         = 1e-7;
    int    fixed_iters = -1;

    if (rank == 0) {
        if (argc < 2) {
            std::cerr << "Usage: OMP_NUM_THREADS=T mpirun -np P " << argv[0]
                      << " <N> [tol] [fixed_iters]" << std::endl;
            std::cerr << "Example: OMP_NUM_THREADS=2 mpirun -np 4 " << argv[0]
                      << " 1024 1e-7 5000" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        N           = std::stoi(argv[1]);
        tol         = (argc >= 3) ? std::stod(argv[2]) : 1e-7;
        fixed_iters = (argc >= 4) ? std::stoi(argv[3]) : -1;
    }

    MPI_Bcast(&N,           1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&tol,         1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fixed_iters, 1, MPI_INT,    0, MPI_COMM_WORLD);

    // Create 2D Cartesian topology
    int dims[2]    = {0, 0};
    int periods[2] = {0, 0};
    MPI_Dims_create(num_procs, 2, dims);

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_get(cart_comm, 2, dims, periods, coords);

    int north, south, west, east;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west,  &east);

    int local_rows  = N / dims[0];
    int local_cols  = N / dims[1];
    int row_offset  = coords[0] * local_rows;
    int col_offset  = coords[1] * local_cols;

    bool on_top    = (coords[0] == 0);
    bool on_bottom = (coords[0] == dims[0] - 1);
    bool on_left   = (coords[1] == 0);
    bool on_right  = (coords[1] == dims[1] - 1);

    int num_threads = omp_get_max_threads();

    if (rank == 0) {
        std::cout << "======================================" << std::endl;
        std::cout << "V4 Hybrid 2D MPI+OpenMP Solver" << std::endl;
        std::cout << "Grid size   : " << N << " x " << N << std::endl;
        std::cout << "MPI ranks   : " << num_procs << std::endl;
        std::cout << "Process grid: " << dims[0] << " x " << dims[1] << std::endl;
        std::cout << "Block size  : " << local_rows << " x " << local_cols << std::endl;
        std::cout << "OMP threads : " << num_threads << std::endl;
        std::cout << "Total cores : " << num_procs * num_threads << std::endl;
        if (fixed_iters > 0)
            std::cout << "Mode        : fixed " << fixed_iters << " iterations" << std::endl;
        else
            std::cout << "Mode        : tolerance " << tol << std::endl;
        std::cout << "======================================" << std::endl;
    }

    MPI_Datatype col_type;
    MPI_Type_vector(local_rows, 1, local_cols + 2, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    std::vector<double> u((local_rows + 2) * (local_cols + 2), 0.0);
    set_boundary_local_2d(u, local_rows, local_cols,
                          row_offset, col_offset, N,
                          on_top, on_bottom, on_left, on_right);

    MPI_Barrier(cart_comm);
    double t_start = MPI_Wtime();

    int iterations = jacobi_solve_hybrid_2d(u, N, tol, fixed_iters, cart_comm,
                                             local_rows, local_cols,
                                             north, south, west, east, col_type, num_threads);

    MPI_Barrier(cart_comm);
    double elapsed = MPI_Wtime() - t_start;

    double local_err  = compute_max_error_local_2d(u, local_rows, local_cols,
                                                    row_offset, col_offset, N);
    double global_err = 0.0;
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "======================================" << std::endl;
        std::cout << "Iterations  : " << iterations << std::endl;
        std::cout << "Time        : " << elapsed << " seconds" << std::endl;
        std::cout << "Max error   : " << global_err << " (vs analytical)" << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "CSV: v4," << N << "," << num_procs << "," << num_threads << ","
                  << elapsed << "," << iterations << "," << global_err << std::endl;
    }

    MPI_Type_free(&col_type);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}

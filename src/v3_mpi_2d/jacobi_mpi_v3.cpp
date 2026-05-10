// ============================================================
// V3 — MPI 2D Cartesian Block Decomposition
// CS-3006 PDC Project, FAST-NUCES Spring 2026
// ============================================================
//
// WHAT THIS VERSION ADDS OVER V2:
//   V1 and V2 use 1D strip decomposition — each process owns a
//   horizontal strip of the full grid width N. Communication per
//   process = 2 full rows = 2N elements per iteration.
//
//   V3 uses 2D block decomposition — the grid is divided into a
//   grid of rectangular blocks using MPI_Cart_create. Each process
//   owns a roughly square block and has 4 neighbors (N/S/E/W).
//
//   Communication comparison:
//     1D strips: 2 * N elements per process per iteration
//     2D blocks: 4 * (N / sqrt(P)) elements per process per iteration
//
//   At P=4:  2D = 4*(N/2) = 2N  (same as 1D)
//   At P=8:  2D < 1D  (2D wins)
//   At P=16: 2D = 4*(N/4) = N   (half of 1D's 2N)
//
//   The advantage of 2D grows with process count — much better scalability.
//
// 2D CARTESIAN TOPOLOGY:
//   MPI_Dims_create: lets MPI pick the best 2D process grid
//     e.g., 4 procs -> 2x2, 8 procs -> 4x2 or 2x4, etc.
//   MPI_Cart_create: creates a communicator with 2D topology
//   MPI_Cart_shift:  finds neighbors in each dimension
//
// GHOST EXCHANGE IN 4 DIRECTIONS:
//   Row exchange (north/south): same as V1/V2, rows are contiguous
//   Column exchange (east/west): columns are NOT contiguous in row-major
//     layout. We use MPI_Type_vector to create a derived datatype
//     that describes a non-contiguous column for efficient transfer.
//
// HOW TO COMPILE:
//   mpicxx -O2 -std=c++17 -o jacobi_v3 jacobi_mpi_v3.cpp -lm
//   OR: make v3
//
// HOW TO RUN:
//   Correctness: mpirun -np 4 ./jacobi_v3 256 1e-7
//   Benchmark:   mpirun -np 4 ./jacobi_v3 1024 1e-7 5000
//
// NOTE: num_procs must produce a valid 2D grid where N is divisible
//       by both dims[0] and dims[1]. For safety use power-of-2 process counts.
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

// Local grid width = local_cols + 2 (including left/right ghost/boundary cols)
#define IDX(i, j) ((i) * (local_cols + 2) + (j))

// ============================================================
// set_boundary_local_2d
//   Sets boundary values for this process's 2D block.
//   Only processes on the global boundary edges set fixed values.
//   Interior process edges are ghost rows/cols filled by MPI exchange.
//
//   on_top:    this process owns global row 0 (y=1, top boundary)
//   on_bottom: this process owns global row N-1 (y=0, bottom boundary)
//   on_left:   this process owns global col 0 (x=0)
//   on_right:  this process owns global col N-1 (x=1)
// ============================================================
void set_boundary_local_2d(std::vector<double>& u,
                           int local_rows, int local_cols,
                           int row_offset, int col_offset, int N,
                           bool on_top, bool on_bottom,
                           bool on_left, bool on_right) {
    // Top boundary row (ghost row 0) — only if this process is on top edge
    if (on_top) {
        for (int j = 0; j <= local_cols + 1; j++) {
            int global_col = col_offset + j - 1;  // -1 because j=1 is first interior col
            if (global_col < 0 || global_col > N + 1) continue;
            double x = (double)global_col / (N + 1);
            u[IDX(0, j)] = sin(PI * x) * exp(-PI);
        }
    }

    // Bottom boundary row — only if this process is on bottom edge
    if (on_bottom) {
        for (int j = 0; j <= local_cols + 1; j++) {
            int global_col = col_offset + j - 1;
            if (global_col < 0 || global_col > N + 1) continue;
            double x = (double)global_col / (N + 1);
            u[IDX(local_rows + 1, j)] = sin(PI * x);
        }
    }

    // Left boundary column — only if this process is on left edge
    if (on_left)
        for (int i = 0; i <= local_rows + 1; i++)
            u[IDX(i, 0)] = 0.0;

    // Right boundary column — only if this process is on right edge
    if (on_right)
        for (int i = 0; i <= local_rows + 1; i++)
            u[IDX(i, local_cols + 1)] = 0.0;
}

double compute_max_error_local_2d(const std::vector<double>& u,
                                   int local_rows, int local_cols,
                                   int row_offset, int col_offset, int N) {
    double max_err = 0.0;
    for (int i = 1; i <= local_rows; i++) {
        int global_row = row_offset + i - 1;  // Local row 1 -> global row row_offset
        double y = 1.0 - (double)global_row / (double)(N + 1);
        for (int j = 1; j <= local_cols; j++) {
            int global_col = col_offset + j - 1;  // Local col 1 -> global col col_offset
            double x = (double)global_col / (N + 1);
            double analytical = sin(PI * x) * exp(-PI * y);
            double err = std::abs(u[IDX(i, j)] - analytical);
            max_err = std::max(max_err, err);
        }
    }
    return max_err;
}

int jacobi_solve_mpi_2d(std::vector<double>& u, int N, double tol,
                         int fixed_iters, MPI_Comm cart_comm,
                         int local_rows, int local_cols,
                         int north, int south, int west, int east,
                         MPI_Datatype col_type) {
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
        MPI_Request reqs[8];  // 4 receives + 4 sends

        // Post receives FIRST (always before sends for deadlock avoidance)
        MPI_Irecv(&u[IDX(0, 0)],              local_cols + 2, MPI_DOUBLE, north, 1, cart_comm, &reqs[0]);  // north ghost row
        MPI_Irecv(&u[IDX(local_rows + 1, 0)], local_cols + 2, MPI_DOUBLE, south, 0, cart_comm, &reqs[1]);  // south ghost row
        MPI_Irecv(&u[IDX(1, 0)],              1, col_type, west,  3, cart_comm, &reqs[2]);  // west ghost col
        MPI_Irecv(&u[IDX(1, local_cols + 1)], 1, col_type, east,  2, cart_comm, &reqs[3]);  // east ghost col

        // Post sends
        MPI_Isend(&u[IDX(1, 0)],           local_cols + 2, MPI_DOUBLE, north, 0, cart_comm, &reqs[4]);  // send north
        MPI_Isend(&u[IDX(local_rows, 0)],  local_cols + 2, MPI_DOUBLE, south, 1, cart_comm, &reqs[5]);  // send south
        MPI_Isend(&u[IDX(1, 1)],           1, col_type, west,  2, cart_comm, &reqs[6]);  // send west
        MPI_Isend(&u[IDX(1, local_cols)],  1, col_type, east,  3, cart_comm, &reqs[7]);  // send east

        // OVERLAP: compute interior cells that don't depend on ghost data
        // Cells in [2..local_rows-1] × [2..local_cols-1] don't need ghost values
        for (int i = 2; i <= local_rows - 1; i++) {
            for (int j = 2; j <= local_cols - 1; j++) {
                double val = (u[IDX(i-1, j)] +
                              u[IDX(i+1, j)] +
                              u[IDX(i, j-1)] +
                              u[IDX(i, j+1)]) * 0.25;
                u_new[IDX(i, j)] = val;
                double delta = std::abs(val - u[IDX(i, j)]);
                if (delta > local_delta) local_delta = delta;
            }
        }

        // Wait for all communication to complete
        MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

        // Now compute edge cells that needed ghost data
        // Top edge (row 1, cols 1..local_cols)
        for (int j = 1; j <= local_cols; j++) {
            double val = (u[IDX(0, j)] +
                          u[IDX(2, j)] +
                          u[IDX(1, j-1)] +
                          u[IDX(1, j+1)]) * 0.25;
            u_new[IDX(1, j)] = val;
            double delta = std::abs(val - u[IDX(1, j)]);
            if (delta > local_delta) local_delta = delta;
        }

        // Bottom edge (row local_rows, cols 1..local_cols)
        for (int j = 1; j <= local_cols; j++) {
            double val = (u[IDX(local_rows-1, j)] +
                          u[IDX(local_rows+1, j)] +
                          u[IDX(local_rows, j-1)] +
                          u[IDX(local_rows, j+1)]) * 0.25;
            u_new[IDX(local_rows, j)] = val;
            double delta = std::abs(val - u[IDX(local_rows, j)]);
            if (delta > local_delta) local_delta = delta;
        }

        // Left edge (rows 2..local_rows-1, col 1) - corners already done
        for (int i = 2; i <= local_rows - 1; i++) {
            double val = (u[IDX(i-1, 1)] +
                          u[IDX(i+1, 1)] +
                          u[IDX(i, 0)] +
                          u[IDX(i, 2)]) * 0.25;
            u_new[IDX(i, 1)] = val;
            double delta = std::abs(val - u[IDX(i, 1)]);
            if (delta > local_delta) local_delta = delta;
        }

        // Right edge (rows 2..local_rows-1, col local_cols) - corners already done
        for (int i = 2; i <= local_rows - 1; i++) {
            double val = (u[IDX(i-1, local_cols)] +
                          u[IDX(i+1, local_cols)] +
                          u[IDX(i, local_cols-1)] +
                          u[IDX(i, local_cols+1)]) * 0.25;
            u_new[IDX(i, local_cols)] = val;
            double delta = std::abs(val - u[IDX(i, local_cols)]);
            if (delta > local_delta) local_delta = delta;
        }

        // Corners (still need these if local_rows >= 2 and local_cols >= 2)
        if (local_rows >= 2 && local_cols >= 2) {
            // Top-left corner (1, 1) - already done in top edge loop
            // Top-right corner (1, local_cols) - already done in top edge loop
            // Bottom-left corner (local_rows, 1) - already done in bottom edge loop
            // Bottom-right corner (local_rows, local_cols) - already done in bottom edge loop
        }

        std::swap(u, u_new);

        // Preserve boundary values: copy from u_new (old u) to u (new grid)
        // After swap, u_new contains the previous iteration's grid with correct boundaries
        // u contains the newly computed values but zero boundaries
        for (int j = 0; j <= local_cols + 1; j++) {
            u[IDX(0, j)] = u_new[IDX(0, j)];  // Top boundary (row 0)
            u[IDX(local_rows + 1, j)] = u_new[IDX(local_rows + 1, j)];  // Bottom boundary
        }
        for (int i = 0; i <= local_rows + 1; i++) {
            u[IDX(i, 0)] = u_new[IDX(i, 0)];  // Left boundary (col 0)
            u[IDX(i, local_cols + 1)] = u_new[IDX(i, local_cols + 1)];  // Right boundary
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
    }

    MPI_Bcast(&N,           1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&tol,         1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fixed_iters, 1, MPI_INT,    0, MPI_COMM_WORLD);

    // ---- Create 2D Cartesian topology ----
    int dims[2]    = {0, 0};
    int periods[2] = {0, 0};   // non-periodic
    MPI_Dims_create(num_procs, 2, dims);  // MPI picks best 2D split

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_get(cart_comm, 2, dims, periods, coords);

    // Find neighbors in each direction
    int north, south, west, east;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);  // row direction
    MPI_Cart_shift(cart_comm, 1, 1, &west,  &east);   // col direction

    // Local block size
    int local_rows  = N / dims[0];
    int local_cols  = N / dims[1];
    int row_offset  = coords[0] * local_rows;  // global row before my first interior row
    int col_offset  = coords[1] * local_cols;  // global col before my first interior col

    // Boundary flags
    bool on_top    = (coords[0] == 0);
    bool on_bottom = (coords[0] == dims[0] - 1);
    bool on_left   = (coords[1] == 0);
    bool on_right  = (coords[1] == dims[1] - 1);

    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);

    if (cart_rank == 0) {
        std::cout << "======================================" << std::endl;
        std::cout << "V3 MPI 2D Cartesian Jacobi Solver" << std::endl;
        std::cout << "Grid size   : " << N << " x " << N << std::endl;
        std::cout << "Processes   : " << num_procs << std::endl;
        std::cout << "Process grid: " << dims[0] << " x " << dims[1] << std::endl;
        std::cout << "Block size  : " << local_rows << " x " << local_cols << std::endl;
        if (fixed_iters > 0)
            std::cout << "Mode        : fixed " << fixed_iters << " iterations" << std::endl;
        else
            std::cout << "Mode        : tolerance " << tol << std::endl;
        std::cout << "======================================" << std::endl;
    }

    // ---- MPI derived type for column exchange ----
    // A column has local_rows elements, each spaced (local_cols+2) doubles apart
    MPI_Datatype col_type;
    MPI_Type_vector(local_rows, 1, local_cols + 2, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    // ---- Allocate local grid: (local_rows+2) x (local_cols+2) ----
    std::vector<double> u((local_rows + 2) * (local_cols + 2), 0.0);

    set_boundary_local_2d(u, local_rows, local_cols,
                          row_offset, col_offset, N,
                          on_top, on_bottom, on_left, on_right);

    MPI_Barrier(cart_comm);
    double t_start = MPI_Wtime();

    int iterations = jacobi_solve_mpi_2d(u, N, tol, fixed_iters, cart_comm,
                                          local_rows, local_cols,
                                          north, south, west, east, col_type);

    MPI_Barrier(cart_comm);
    double elapsed = MPI_Wtime() - t_start;

    double local_err  = compute_max_error_local_2d(u, local_rows, local_cols,
                                                    row_offset, col_offset, N);
    double global_err = 0.0;
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (cart_rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "======================================" << std::endl;
        std::cout << "Iterations  : " << iterations << std::endl;
        std::cout << "Time        : " << elapsed << " seconds" << std::endl;
        std::cout << "Max error   : " << global_err << " (vs analytical)" << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "CSV: v3," << N << "," << num_procs << ",1,"
                  << elapsed << "," << iterations << "," << global_err << std::endl;
    }

    MPI_Type_free(&col_type);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}

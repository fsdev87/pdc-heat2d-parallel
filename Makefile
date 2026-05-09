# ============================================================
# Generic Makefile for PDC Heat Equation Solver (All Versions)
# Supports building, testing, and running any version with any configuration
# ============================================================

# ---- Compiler Settings ----
CXX = mpicxx
CXX_SEQ = g++
CXX_FLAGS = -O2 -std=c++17
OMP_FLAGS = -fopenmp
LDFLAGS = -lm

# ---- Source Files ----
V0_SRC = src/v0_sequential/jacobi_seq.cpp
V1_SRC = src/v1_mpi_blocking/jacobi_mpi_v1.cpp
V2_SRC = src/v2_mpi_nonblocking/jacobi_mpi_v2.cpp
V3_SRC = src/v3_mpi_2d/jacobi_mpi_v3.cpp
V4_SRC = src/v4_hybrid/jacobi_hybrid_v4.cpp

# ---- Target Executables ----
V0_BIN = build/v0/jacobi_seq
V1_BIN = build/v1/jacobi_v1
V2_BIN = build/v2/jacobi_v2
V3_BIN = build/v3/jacobi_v3
V4_BIN = build/v4/jacobi_v4

# ---- Test Configuration ----
GRID_SIZES = 256 1024 4096
TOLERANCES = 1e-6 1e-6
PROCS_V0 = 1
PROCS_MPI = 1 2 4 8 16
THREADS_V4 = 1 2 4

# Default values for generic test
VERSION ?= v0
N ?= 256
PROCS ?= 1
TOL ?= 1e-6
FIXED_ITERS ?= -1
THREADS ?= 1

# ---- MPI/Cluster Settings ----
HOSTFILE = hostfile
MPIRUN = mpirun

# ---- Default Target ----
.PHONY: all
all: v0 v1 v2 v3 v4

# ---- Version Build Targets ----
.PHONY: v0
v0: $(V0_BIN)

.PHONY: v1
v1: $(V1_BIN)

.PHONY: v2
v2: $(V2_BIN)

.PHONY: v3
v3: $(V3_BIN)

.PHONY: v4
v4: $(V4_BIN)

# ---- Build Rules ----
$(V0_BIN): $(V0_SRC)
	@echo "Building V0 (Sequential)..."
	@mkdir -p build/v0
	$(CXX_SEQ) $(CXX_FLAGS) -o $@ $< $(LDFLAGS)
	@echo "V0 built successfully: $@"

$(V1_BIN): $(V1_SRC)
	@echo "Building V1 (MPI Blocking)..."
	@mkdir -p build/v1
	$(CXX) $(CXX_FLAGS) -o $@ $< $(LDFLAGS)
	@echo "V1 built successfully: $@"

$(V2_BIN): $(V2_SRC)
	@echo "Building V2 (MPI Non-Blocking)..."
	@mkdir -p build/v2
	$(CXX) $(CXX_FLAGS) -o $@ $< $(LDFLAGS)
	@echo "V2 built successfully: $@"

$(V3_BIN): $(V3_SRC)
	@echo "Building V3 (MPI 2D Cartesian)..."
	@mkdir -p build/v3
	$(CXX) $(CXX_FLAGS) -o $@ $< $(LDFLAGS)
	@echo "V3 built successfully: $@"

$(V4_BIN): $(V4_SRC)
	@echo "Building V4 (Hybrid MPI+OpenMP)..."
	@mkdir -p build/v4
	$(CXX) $(CXX_FLAGS) $(OMP_FLAGS) -o $@ $< $(LDFLAGS)
	@echo "V4 built successfully: $@"

# ---- Generic Test Target ----
# Usage: make test VERSION=v0 N=256 PROCS=1 [TOL=1e-6] [FIXED_ITERS=-1] [THREADS=1]
#   FIXED_ITERS=-1: tolerance mode (default, stops when converged)
#   FIXED_ITERS=N:   fixed iteration mode (e.g., FIXED_ITERS=5000, for benchmarking)
#   PROCS > 4:      automatically uses cluster (hostfile required)
#   PROCS <= 4:     runs locally on single machine
.PHONY: test
test:
	@echo "Running test: Version=$(VERSION), Grid=$(N), Procs=$(PROCS), Threads=$(THREADS), Tol=$(TOL), FixedIters=$(FIXED_ITERS)"
	@mkdir -p results/raw
	@if [ "$(VERSION)" = "v0" ]; then \
		if [ "$(FIXED_ITERS)" = "-1" ]; then \
			$(V0_BIN) $(N) $(TOL) 2>&1 | tee -a results/raw/$(VERSION)_test.log; \
		else \
			$(V0_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | tee -a results/raw/$(VERSION)_test.log; \
		fi \
	else \
		if [ "$(VERSION)" = "v4" ]; then \
			if [ $(PROCS) -gt 4 ]; then \
				if [ "$(FIXED_ITERS)" = "-1" ]; then \
					OMP_NUM_THREADS=$(THREADS) $(MPIRUN) -np $(PROCS) --hostfile $(HOSTFILE) $(V4_BIN) $(N) $(TOL) 2>&1 | tee -a results/raw/$(VERSION)_test.log; \
				else \
					OMP_NUM_THREADS=$(THREADS) $(MPIRUN) -np $(PROCS) --hostfile $(HOSTFILE) $(V4_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | tee -a results/raw/$(VERSION)_test.log; \
				fi \
			else \
				if [ "$(FIXED_ITERS)" = "-1" ]; then \
					OMP_NUM_THREADS=$(THREADS) $(MPIRUN) -np $(PROCS) $(V4_BIN) $(N) $(TOL) 2>&1 | tee -a results/raw/$(VERSION)_test.log; \
				else \
					OMP_NUM_THREADS=$(THREADS) $(MPIRUN) -np $(PROCS) $(V4_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | tee -a results/raw/$(VERSION)_test.log; \
				fi \
			fi \
		else \
			if [ $(PROCS) -gt 4 ]; then \
				if [ "$(FIXED_ITERS)" = "-1" ]; then \
					$(MPIRUN) -np $(PROCS) --hostfile $(HOSTFILE) $(V$(subst v,,$(VERSION))_BIN) $(N) $(TOL) 2>&1 | tee -a results/raw/$(VERSION)_test.log; \
				else \
					$(MPIRUN) -np $(PROCS) --hostfile $(HOSTFILE) $(V$(subst v,,$(VERSION))_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | tee -a results/raw/$(VERSION)_test.log; \
				fi \
			else \
				if [ "$(FIXED_ITERS)" = "-1" ]; then \
					$(MPIRUN) -np $(PROCS) $(V$(subst v,,$(VERSION))_BIN) $(N) $(TOL) 2>&1 | tee -a results/raw/$(VERSION)_test.log; \
				else \
					$(MPIRUN) -np $(PROCS) $(V$(subst v,,$(VERSION))_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | tee -a results/raw/$(VERSION)_test.log; \
				fi \
			fi \
		fi \
	fi \

# ---- Quick Tests (Single Grid Size) ----
.PHONY: test-v0-quick
test-v0-quick: v0
	@echo "Quick test V0 with N=256, tol=1e-6..."
	@$(MAKE) test VERSION=v0 N=256 PROCS=1 TOL=1e-6

.PHONY: test-v1-quick
test-v1-quick: v1
	@echo "Quick test V1 with N=256, procs=2, tol=1e-6..."
	@$(MAKE) test VERSION=v1 N=256 PROCS=2 TOL=1e-6

.PHONY: test-v2-quick
test-v2-quick: v2
	@echo "Quick test V2 with N=256, procs=2, tol=1e-6..."
	@$(MAKE) test VERSION=v2 N=256 PROCS=2 TOL=1e-6

.PHONY: test-v3-quick
test-v3-quick: v3
	@echo "Quick test V3 with N=256, procs=2, tol=1e-6..."
	@$(MAKE) test VERSION=v3 N=256 PROCS=2 TOL=1e-6

.PHONY: test-v4-quick
test-v4-quick: v4
	@echo "Quick test V4 with N=256, procs=2, threads=2, tol=1e-6..."
	@$(MAKE) test VERSION=v4 N=256 PROCS=2 THREADS=2 TOL=1e-6

# ---- Full Test Suite ----
# Tests each version with multiple configurations
.PHONY: test-full
test-full:
	@echo "Running full test suite..."
	@echo "========================================" > results/raw/full_test_results.csv
	@echo "version,N,procs,threads,time,iterations,max_error" >> results/raw/full_test_results.csv

	# Test V0
	@$(MAKE) test-v0-varying

	# Test V1-V3 with different process counts
	@$(MAKE) test-v1-varying
	@$(MAKE) test-v2-varying
	@$(MAKE) test-v3-varying

	# Test V4 with different process×thread combos
	@$(MAKE) test-v4-varying

	@echo "Full test suite complete. Results in results/raw/full_test_results.csv"

.PHONY: test-v0-varying
test-v0-varying:
	@echo "Testing V0 with different grid sizes..."
	@for N in $(GRID_SIZES); do \
		for TOL in $(TOLERANCES); do \
			echo "Testing V0: N=$$N, tol=$$TOL"; \
			$(V0_BIN) $$N $$TOL | grep "CSV:" >> results/raw/full_test_results.csv; \
		done; \
	done

.PHONY: test-v1-varying
test-v1-varying:
	@echo "Testing V1 with different process counts..."
	@for PROCS in $(PROCS_MPI); do \
		echo "Testing V1: procs=$$PROCS"; \
		$(MPIRUN) -np $$PROCS $(V1_BIN) 1024 1e-6 | grep "CSV:" >> results/raw/full_test_results.csv; \
	done

.PHONY: test-v2-varying
test-v2-varying:
	@echo "Testing V2 with different process counts..."
	@for PROCS in $(PROCS_MPI); do \
		echo "Testing V2: procs=$$PROCS"; \
		$(MPIRUN) -np $$PROCS $(V2_BIN) 1024 1e-6 | grep "CSV:" >> results/raw/full_test_results.csv; \
	done

.PHONY: test-v3-varying
test-v3-varying:
	@echo "Testing V3 with different process counts..."
	@for PROCS in $(PROCS_MPI); do \
		echo "Testing V3: procs=$$PROCS"; \
		$(MPIRUN) -np $$PROCS $(V3_BIN) 1024 1e-6 | grep "CSV:" >> results/raw/full_test_results.csv; \
	done

.PHONY: test-v4-varying
test-v4-varying:
	@echo "Testing V4 with different process×thread combos..."
	@for PROCS in 1 2 4; do \
		for THREADS in 1 2 4; do \
			TOTAL=$$((PROCS * THREADS)); \
			if [ $$TOTAL -le 16 ]; then \
				echo "Testing V4: procs=$$PROCS, threads=$$THREADS"; \
				OMP_NUM_THREADS=$$THREADS $(MPIRUN) -np $$PROCS $(V4_BIN) 1024 1e-6 | grep "CSV:" >> results/raw/full_test_results.csv; \
			fi; \
		done; \
	done

# ---- Cluster Running ----
# Local: runs on single machine (PROCS <= 4)
.PHONY: test-local
test-local:
	@echo "Testing locally (single machine)..."
	@for PROCS in 1 2 4; do \
		echo "Testing V1 locally with $$PROCS processes..."; \
		$(MPIRUN) -np $$PROCS $(V1_BIN) 1024 1e-6 | grep "CSV:"; \
	done

# Cluster: uses hostfile to distribute across multiple machines (PROCS > 4)
.PHONY: test-cluster
test-cluster: check-hostfile
	@echo "Testing on cluster (multiple machines)..."
	@for PROCS in 8 16; do \
		echo "Testing V1 on cluster with $$PROCS processes..."; \
		$(MPIRUN) -np $$PROCS --hostfile $(HOSTFILE) $(V1_BIN) 1024 1e-6 | grep "CSV:"; \
	done

.PHONY: check-hostfile
check-hostfile:
	@if [ ! -f $(HOSTFILE) ]; then \
		echo "Error: hostfile not found! Create $(HOSTFILE) first."; \
		exit 1; \
	fi

# ---- Verification ----
.PHONY: verify
verify:
	@echo "Verifying correctness against analytical solution..."
	@echo "========================================"
	@echo "Testing V0 (Sequential)..."
	@$(V0_BIN) 256 1e-6 | grep -E "(Max error|CSV:)"
	@echo ""
	@echo "Testing V1 (MPI) with 2 processes..."
	@$(MPIRUN) -np 2 $(V1_BIN) 256 1e-6 | grep -E "(Max error|CSV:)"
	@echo ""
	@echo "Errors should be < 0.01 for tolerance 1e-6"
	@echo "Errors should be < 0.001 for tolerance 1e-6"

.PHONY: verify-v1
verify-v1: v1
	@echo "Verifying V1 correctness with different process counts..."
	@for PROCS in 1 2 4 8; do \
		echo "V1 with $$PROCS processes:"; \
		$(MPIRUN) -np $$PROCS $(V1_BIN) 256 1e-6 | grep "Max error"; \
	done

# ---- Cleanup ----
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/
	@rm -f *.o *.d *.out *.exe

.PHONY: clean-results
clean-results:
	@echo "Cleaning results..."
	@rm -rf results/raw/*.log results/raw/*.csv

.PHONY: clean-all
clean-all: clean clean-results
	@echo "Cleaned everything."

# ---- Setup ----
.PHONY: setup
setup:
	@echo "Creating directory structure..."
	@mkdir -p build/v0 build/v1 build/v2 build/v3 build/v4
	@mkdir -p src/v0_sequential src/v1_mpi_blocking src/v2_mpi_nonblocking src/v3_mpi_2d src/v4_hybrid
	@mkdir -p scripts results/raw results/plots
	@echo "Directory structure created."

.PHONY: help
help:
	@echo "PDC Heat Equation Solver - Makefile Help"
	@echo "========================================"
	@echo "Building:"
	@echo "  make all              - Build all versions"
	@echo "  make v0, v1, v2, v3, v4 - Build specific version"
	@echo ""
	@echo "Testing:"
	@echo "  make test VERSION=v0 N=256 PROCS=1 [TOL=1e-6] [FIXED_ITERS=-1] [THREADS=1]"
	@echo "  make test-v0-quick                                 - Quick test V0"
	@echo "  make test-v1-quick                                 - Quick test V1"
	@echo "  make test-full                                     - Full test suite"
	@echo ""
	@echo "Cluster Testing:"
	@echo "  make test-local                                     - Test on single machine"
	@echo "  make test-cluster                                   - Test on cluster (needs hostfile)"
	@echo ""
	@echo "Verification:"
	@echo "  make verify                                         - Verify correctness"
	@echo "  make verify-v1                                      - Verify V1 with different procs"
	@echo ""
	@echo "Setup & Cleanup:"
	@echo "  make setup                                          - Create directories"
	@echo "  make clean                                          - Clean build artifacts"
	@echo "  make clean-all                                      - Clean everything"
	@echo ""
	@echo "Examples:"
	@echo "  make test VERSION=v0 N=256 PROCS=1 TOL=1e-6"
	@echo "  make test VERSION=v1 N=256 PROCS=2 TOL=1e-6"
	@echo "  make test VERSION=v1 N=1024 PROCS=4 TOL=1e-6 FIXED_ITERS=5000"
	@echo "  make test VERSION=v1 N=1024 PROCS=8 TOL=1e-6 FIXED_ITERS=5000  (cluster)"
	@echo "  make test VERSION=v4 N=1024 PROCS=2 THREADS=2 TOL=1e-6"
	@echo ""
	@echo "Automatic Mode Selection:"
	@echo "  PROCS <= 4: Runs locally (single machine)"
	@echo "  PROCS > 4:  Runs on cluster (requires hostfile)"
	@echo ""
	@echo "Execution Modes:"
	@echo "  FIXED_ITERS=-1:  Tolerance mode (default, stops when converged)"
	@echo "  FIXED_ITERS=N:   Fixed N iterations (for fair benchmarking)"
	@echo "                    - Use for all performance/speedup experiments"
	@echo "                    - Guarantees identical workload across process counts"
	@echo "                    - Removes MPI_Allreduce overhead"

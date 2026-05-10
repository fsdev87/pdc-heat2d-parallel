# ============================================================
# Makefile — PDC Heat Equation Solver (All Versions)
# CS-3006 FAST-NUCES Spring 2026
# ============================================================
#
# BUILDING:
#   make all                    build all versions
#   make v0/v1/v2/v3/v4        build specific version
#
# GENERIC RUN:
#   make test VERSION=v0 N=256
#   make test VERSION=v1 N=1024 PROCS=4 FIXED_ITERS=5000
#   make test VERSION=v1 N=1024 PROCS=8 FIXED_ITERS=5000     <- cluster
#   make test VERSION=v1 N=1024 PROCS=16 FIXED_ITERS=5000    <- oversubscribed
#   make test VERSION=v4 N=1024 PROCS=2 THREADS=2 FIXED_ITERS=5000
#
# PROCESS COUNT RULES:
#   PROCS=1,2,4  -> local on your machine
#   PROCS=8      -> cluster via hostfile (try at university)
#   PROCS=16     -> --oversubscribe (context switching, no cluster needed)
#
# SHORTCUTS:
#   make correctness N=256                         verify V0 vs V1
#   make bench-local N=1024 FIXED_ITERS=5000       benchmark PROCS=1,2,4
#   make bench-cluster N=1024 FIXED_ITERS=5000     benchmark PROCS=8
#   make bench-oversubscribe N=1024 FIXED_ITERS=5000  PROCS=16
# ============================================================

CXX_SEQ  = g++
CXX_MPI  = mpicxx

FLAGS_SEQ    = -O2 -std=c++17 -lm
FLAGS_MPI    = -O2 -std=c++17 -lm
FLAGS_HYBRID = -O2 -std=c++17 -fopenmp -lm

V0_SRC = src/v0_sequential/jacobi_seq.cpp
V1_SRC = src/v1_mpi_blocking/jacobi_mpi_v1.cpp
V2_SRC = src/v2_mpi_nonblocking/jacobi_mpi_v2.cpp
V3_SRC = src/v3_mpi_2d/jacobi_mpi_v3.cpp
V4_SRC = src/v4_hybrid/jacobi_hybrid_v4.cpp

V0_BIN = build/v0/jacobi_seq
V1_BIN = build/v1/jacobi_v1
V2_BIN = build/v2/jacobi_v2
V3_BIN = build/v3/jacobi_v3
V4_BIN = build/v4/jacobi_v4

VERSION     ?= v0
N           ?= 256
PROCS       ?= 1
THREADS     ?= 1
TOL         ?= 1e-6
FIXED_ITERS ?= -1
HOSTFILE    ?= hostfile

# ============================================================
# BUILD
# ============================================================
.PHONY: all v0 v1 v2 v3 v4
all: v0 v1 v2 v3 v4

v0: $(V0_BIN)
v1: $(V1_BIN)
v2: $(V2_BIN)
v3: $(V3_BIN)
v4: $(V4_BIN)

$(V0_BIN): $(V0_SRC)
	@mkdir -p build/v0
	$(CXX_SEQ) $(FLAGS_SEQ) -o $@ $<
	@echo "Built $@"

$(V1_BIN): $(V1_SRC)
	@mkdir -p build/v1
	$(CXX_MPI) $(FLAGS_MPI) -o $@ $<
	@echo "Built $@"

$(V2_BIN): $(V2_SRC)
	@mkdir -p build/v2
	$(CXX_MPI) $(FLAGS_MPI) -o $@ $<
	@echo "Built $@"

$(V3_BIN): $(V3_SRC)
	@mkdir -p build/v3
	$(CXX_MPI) $(FLAGS_MPI) -o $@ $<
	@echo "Built $@"

$(V4_BIN): $(V4_SRC)
	@mkdir -p build/v4
	$(CXX_MPI) $(FLAGS_HYBRID) -o $@ $<
	@echo "Built $@"

# ============================================================
# GENERIC TEST
# ============================================================
.PHONY: test
test:
	@mkdir -p results/raw
ifeq ($(VERSION),v0)
	@echo "--- V0 | N=$(N) ---"
	@$(V0_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | tee -a results/raw/$(VERSION)_N$(N).log
else ifeq ($(VERSION),v4)
	@if [ $(PROCS) -gt 8 ]; then \
		echo "--- V4 | N=$(N) | PROCS=$(PROCS) [oversubscribed] | THREADS=$(THREADS) ---"; \
		OMP_NUM_THREADS=$(THREADS) mpirun -np $(PROCS) --oversubscribe $(V4_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | tee -a results/raw/$(VERSION)_N$(N).log; \
	elif [ $(PROCS) -gt 4 ]; then \
		echo "--- V4 | N=$(N) | PROCS=$(PROCS) [cluster] | THREADS=$(THREADS) ---"; \
		OMP_NUM_THREADS=$(THREADS) mpirun -np $(PROCS) --hostfile $(HOSTFILE) $(V4_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | tee -a results/raw/$(VERSION)_N$(N).log; \
	else \
		echo "--- V4 | N=$(N) | PROCS=$(PROCS) [local] | THREADS=$(THREADS) ---"; \
		OMP_NUM_THREADS=$(THREADS) mpirun -np $(PROCS) $(V4_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | tee -a results/raw/$(VERSION)_N$(N).log; \
	fi
else
	@if [ $(PROCS) -gt 8 ]; then \
		echo "--- $(VERSION) | N=$(N) | PROCS=$(PROCS) [oversubscribed] ---"; \
		mpirun -np $(PROCS) --oversubscribe build/$(subst v,,$(VERSION))/jacobi_$(VERSION) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | tee -a results/raw/$(VERSION)_N$(N).log; \
	elif [ $(PROCS) -gt 4 ]; then \
		echo "--- $(VERSION) | N=$(N) | PROCS=$(PROCS) [cluster] ---"; \
		mpirun -np $(PROCS) --hostfile $(HOSTFILE) build/$(subst v,,$(VERSION))/jacobi_$(VERSION) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | tee -a results/raw/$(VERSION)_N$(N).log; \
	else \
		echo "--- $(VERSION) | N=$(N) | PROCS=$(PROCS) [local] ---"; \
		mpirun -np $(PROCS) build/$(subst v,,$(VERSION))/jacobi_$(VERSION) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | tee -a results/raw/$(VERSION)_N$(N).log; \
	fi
endif

# ============================================================
# CORRECTNESS: compare V0 vs V1 on same N
# Usage: make correctness N=256
# ============================================================
.PHONY: correctness
correctness: v0 v1
	@echo "============================================"
	@echo "Correctness Check | N=$(N) | TOL=$(TOL)"
	@echo "All max_error values should be similar"
	@echo "============================================"
	@echo "[V0 Sequential]"
	@$(V0_BIN) $(N) $(TOL) 2>&1 | grep -E "Iterations|Time|Max error"
	@echo "[V1 NP=1]"
	@mpirun -np 1 $(V1_BIN) $(N) $(TOL) 2>&1 | grep -E "Iterations|Time|Max error"
	@echo "[V1 NP=2]"
	@mpirun -np 2 $(V1_BIN) $(N) $(TOL) 2>&1 | grep -E "Iterations|Time|Max error"
	@echo "[V1 NP=4]"
	@mpirun -np 4 $(V1_BIN) $(N) $(TOL) 2>&1 | grep -E "Iterations|Time|Max error"
	@echo "============================================"

# ============================================================
# BENCH-LOCAL: PROCS=1,2,4 on your machine only
# Usage: make bench-local N=1024 FIXED_ITERS=5000
# ============================================================
.PHONY: bench-local
bench-local: all
	@mkdir -p results/raw
	@echo "version,N,num_procs,num_threads,time,iterations,max_error" > results/raw/bench_local.csv
	@echo "=== Local Benchmark | N=$(N) | FIXED_ITERS=$(FIXED_ITERS) ==="
	@echo "--- V0 ---"
	@$(V0_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | grep "^CSV:" | sed 's/CSV: //' | tee -a results/raw/bench_local.csv
	@for p in 1 2 4; do \
		for v in v1 v2 v3; do \
			echo "--- $$v NP=$$p ---"; \
			mpirun -np $$p build/$${v#v}/jacobi_$$v $(N) $(TOL) $(FIXED_ITERS) 2>&1 | grep "^CSV:" | sed 's/CSV: //' | tee -a results/raw/bench_local.csv; \
		done; \
	done
	@echo "--- V4 NP=1 THREADS=4 ---"
	@OMP_NUM_THREADS=4 mpirun -np 1 $(V4_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | grep "^CSV:" | sed 's/CSV: //' | tee -a results/raw/bench_local.csv
	@echo "--- V4 NP=2 THREADS=2 ---"
	@OMP_NUM_THREADS=2 mpirun -np 2 $(V4_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | grep "^CSV:" | sed 's/CSV: //' | tee -a results/raw/bench_local.csv
	@echo "--- V4 NP=4 THREADS=1 ---"
	@OMP_NUM_THREADS=1 mpirun -np 4 $(V4_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | grep "^CSV:" | sed 's/CSV: //' | tee -a results/raw/bench_local.csv
	@echo "Done. Results in results/raw/bench_local.csv"

# ============================================================
# BENCH-CLUSTER: PROCS=8 via hostfile
# Usage: make bench-cluster N=1024 FIXED_ITERS=5000
# ============================================================
.PHONY: bench-cluster
bench-cluster: v1 v2 v3 v4
	@mkdir -p results/raw
	@if [ ! -f $(HOSTFILE) ]; then echo "Error: hostfile not found"; exit 1; fi
	@echo "=== Cluster Benchmark | N=$(N) | NP=8 ==="
	@for v in v1 v2 v3; do \
		echo "--- $$v NP=8 ---"; \
		mpirun -np 8 --hostfile $(HOSTFILE) build/$${v#v}/jacobi_$$v $(N) $(TOL) $(FIXED_ITERS) 2>&1 | grep "^CSV:" | sed 's/CSV: //' | tee -a results/raw/bench_cluster.csv; \
	done
	@echo "--- V4 NP=4 THREADS=2 ---"
	@OMP_NUM_THREADS=2 mpirun -np 4 --hostfile $(HOSTFILE) $(V4_BIN) $(N) $(TOL) $(FIXED_ITERS) 2>&1 | grep "^CSV:" | sed 's/CSV: //' | tee -a results/raw/bench_cluster.csv
	@echo "Done. Results in results/raw/bench_cluster.csv"

# ============================================================
# BENCH-OVERSUBSCRIBE: PROCS=16 with context switching
# Usage: make bench-oversubscribe N=1024 FIXED_ITERS=5000
# ============================================================
.PHONY: bench-oversubscribe
bench-oversubscribe: v1 v2 v3
	@mkdir -p results/raw
	@echo "=== Oversubscribed Benchmark | N=$(N) | NP=16 ==="
	@for v in v1 v2 v3; do \
		echo "--- $$v NP=16 [oversubscribed] ---"; \
		mpirun -np 16 --oversubscribe build/$${v#v}/jacobi_$$v $(N) $(TOL) $(FIXED_ITERS) 2>&1 | grep "^CSV:" | sed 's/CSV: //' | tee -a results/raw/bench_oversubscribe.csv; \
	done
	@echo "Done. Results in results/raw/bench_oversubscribe.csv"

# ============================================================
# CLEAN / SETUP / HELP
# ============================================================
.PHONY: clean clean-results clean-all setup help
clean:
	rm -rf build/
	@echo "Build artifacts removed."

clean-results:
	rm -f results/raw/*.log results/raw/*.csv
	@echo "Results cleared."

clean-all: clean clean-results

setup:
	@mkdir -p build/v0 build/v1 build/v2 build/v3 build/v4
	@mkdir -p results/raw results/plots scripts
	@echo "Directories ready."

help:
	@echo ""
	@echo "BUILD:  make all | make v0 | make v1 | make v2 | make v3 | make v4"
	@echo ""
	@echo "TEST:   make test VERSION=v1 N=1024 PROCS=4 FIXED_ITERS=5000"
	@echo "        PROCS=1,2,4  -> local    PROCS=8 -> cluster    PROCS=16 -> oversubscribed"
	@echo ""
	@echo "CHECK:  make correctness N=256"
	@echo ""
	@echo "BENCH:  make bench-local N=1024 FIXED_ITERS=5000"
	@echo "        make bench-cluster N=1024 FIXED_ITERS=5000"
	@echo "        make bench-oversubscribe N=1024 FIXED_ITERS=5000"
	@echo ""
	@echo "CLEAN:  make clean | make clean-results | make clean-all"
	@echo ""
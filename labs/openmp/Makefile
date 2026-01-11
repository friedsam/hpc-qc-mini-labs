# labs/openmp/Makefile
# Portable OpenMP Makefile (macOS/Linux/WSL). Auto-detects Homebrew GCC on macOS.

UNAME_S := $(shell uname -s)
BREW_GXX := $(shell (ls -1 /opt/homebrew/bin/g++-* /usr/local/bin/g++-* 2>/dev/null || true) | sort -V | tail -n 1)

# If user explicitly set CXX on the make command line, honor it.
# Otherwise, choose a good default compiler, ignoring Conda's injected CXX.
ifeq ($(origin CXX), command line)
  # keep user's choice
else
  ifeq ($(UNAME_S),Darwin)
    # Pick highest Homebrew GCC (works even if /opt/homebrew/bin isn't in PATH for make)
    BREW_GXX := $(shell (ls -1 /opt/homebrew/bin/g++-* /usr/local/bin/g++-* 2>/dev/null || true) | sort -V | tail -n 1)
    ifneq ($(strip $(BREW_GXX)),)
      CXX := $(BREW_GXX)
    else
      CXX := clang++
    endif
  else
    CXX := g++
  endif
endif

# Do NOT inherit Conda's clang-oriented flags for these teaching labs.
override CXXFLAGS := -O3 -std=c++17 $(EXTRA_CXXFLAGS)
override OMPFLAGS := -fopenmp $(EXTRA_OMPFLAGS)

CPP_DIR := cpp
BIN_DIR := bin

WARMUP := $(BIN_DIR)/omp_00_warmup
AMDAHL := $(BIN_DIR)/omp_01_amdahl
MUTEX  := $(BIN_DIR)/omp_02_mutex_bottleneck

.PHONY: help print-vars check-openmp dirs clean build \
        warmup amdahl mutex \
        run-warmup run-amdahl run-mutex install-hints

help:
	@echo "OpenMP mini-labs"
	@echo ""
	@echo "Targets:"
	@echo "  make build                  # build all binaries"
	@echo "  make run-warmup  T=4         # run warmup with T threads"
	@echo "  make run-amdahl  T=1|2|4|8   # run amdahl with T threads"
	@echo "  make run-mutex   T=4         # run mutex bottleneck with T threads"
	@echo "  make clean"
	@echo "  make print-vars"
	@echo "  make install-hints"
	@echo ""
	@echo "Override compiler explicitly:"
	@echo "  make build CXX=g++-15"
	@echo "  make build CXX=g++-14"

print-vars:
	@echo "UNAME_S  = $(UNAME_S)"
	@echo "CXX      = $(CXX)"
	@echo "CXXPATH  = $$(command -v $(CXX) 2>/dev/null || echo '(not in PATH)')"
	@echo "CXXFLAGS = $(CXXFLAGS)"
	@echo "OMPFLAGS = $(OMPFLAGS)"

check-openmp:
	@echo "Compiler: $$($(CXX) --version | head -n 1)"
	@if [ "$(UNAME_S)" = "Darwin" ] && [ -z "$(BREW_GXX)" ]; then \
	  echo ""; \
	  echo "ERROR: No Homebrew GCC found at /opt/homebrew/bin/g++-*"; \
	  echo "Fix: brew install gcc"; \
	  echo ""; \
	  exit 1; \
	fi
	@if $(CXX) --version 2>/dev/null | grep -qi "clang"; then \
	  echo ""; \
	  echo "ERROR: clang detected. Apple clang does not support '-fopenmp' by default."; \
	  echo "Fix (macOS): make build CXX=/opt/homebrew/bin/g++-15"; \
	  echo ""; \
	  exit 1; \
	fi
	@# compile-test OpenMP
	@printf '#include <omp.h>\nint main(){return 0;}\n' | \
	  $(CXX) $(CXXFLAGS) $(OMPFLAGS) -x c++ - -o /tmp/omp_make_check >/dev/null 2>&1 || ( \
	  echo ""; \
	  echo "ERROR: OpenMP test compile failed with CXX='$(CXX)'."; \
	  echo "Fix: on macOS install Homebrew GCC (brew install gcc)."; \
	  echo ""; \
	  exit 1; \
	)
	@rm -f /tmp/omp_make_check

dirs:
	@mkdir -p $(BIN_DIR)

clean:
	@rm -rf $(BIN_DIR)

build: check-openmp dirs warmup amdahl mutex

warmup: $(WARMUP)
amdahl: $(AMDAHL)
mutex:  $(MUTEX)

$(WARMUP): $(CPP_DIR)/omp_00_warmup.cpp | dirs
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

$(AMDAHL): $(CPP_DIR)/omp_01_amdahl.cpp | dirs
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

$(MUTEX): $(CPP_DIR)/omp_02_mutex_bottleneck.cpp | dirs
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

run-warmup: warmup
	@OMP_NUM_THREADS=$(T) $(WARMUP)

run-amdahl: amdahl
	@OMP_NUM_THREADS=$(T) $(AMDAHL)

run-mutex: mutex
	@OMP_NUM_THREADS=$(T) $(MUTEX)

install-hints:
	@echo "Install hints:"
	@echo ""
	@echo "macOS:"
	@echo "  brew install gcc"
	@echo "  make build     # auto-detects /opt/homebrew/bin/g++-NN"
	@echo ""
	@echo "Ubuntu/Debian (Linux/WSL):"
	@echo "  sudo apt-get update && sudo apt-get install -y g++ make"
	@echo "  make build"
	@echo ""
	@echo "Windows:"
	@echo "  Use WSL2 (Ubuntu) and follow the Ubuntu/Debian steps."
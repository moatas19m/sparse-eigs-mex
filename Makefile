# speigs -- standalone build (core + tests + benchmarks). No MATLAB needed.
# The MEX gateway is built separately from MATLAB: cd matlab; build_mex
#
#   make            build core, tests, benchmarks
#   make test       run the validation suite
#   make bench      run the end-to-end benchmark
#   make CC=gcc-16  build with GNU gcc instead of clang

CC      ?= clang
SS_INC  ?= $(shell pkg-config --cflags-only-I CHOLMOD 2>/dev/null || echo -I/opt/homebrew/include/suitesparse)
SS_LIB  ?= $(shell pkg-config --libs-only-L CHOLMOD 2>/dev/null || echo -L/opt/homebrew/lib)
CFLAGS  ?= -O2 -Wall -Wextra -Iinclude $(SS_INC)
LDLIBS  ?= $(SS_LIB) -lcholmod -lumfpack -lsuitesparseconfig -lm

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  LDLIBS += -framework Accelerate
  # Homebrew GCC on macOS does not find the SDK headers on its own.
  ifneq (,$(findstring gcc,$(CC)))
    CFLAGS += -isysroot $(shell xcrun --show-sdk-path)
  endif
else
  LDLIBS += -llapack -lblas
endif

BUILD := build
BINS  := $(BUILD)/test_speigs $(BUILD)/bench_model

all: $(BINS)

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/speigs_core.o: src/speigs_core.c include/speigs.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/test_speigs: test/test_speigs.c $(BUILD)/speigs_core.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/bench_model: bench/bench_model.c $(BUILD)/speigs_core.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test: $(BUILD)/test_speigs
	./$(BUILD)/test_speigs 16

bench: $(BUILD)/bench_model
	./$(BUILD)/bench_model 40 6

clean:
	rm -rf $(BUILD)

.PHONY: all test bench clean

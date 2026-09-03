# speigs — fast sparse eigenpairs as a MEX-C function

`[V,D] = speigs(A,k)` — the `k` smallest- (or largest-) magnitude eigenvalues and
eigenvectors of a sparse square matrix, aiming for **≥ 5× faster than MATLAB's
`eigs`** by exploiting symmetry, positive-definiteness, and nonzero structure.

**Status: design + environment set up. Solver not yet written.**

Read [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) first — it contains the
feasibility evidence, the design, and the honest list of risks.

## Why this can be faster than `eigs`

For the smallest-magnitude case `eigs` shift-inverts using MATLAB's `lu`, i.e.
**UMFPACK unsymmetric LU — even when the matrix is symmetric positive definite** —
with AMD ordering, and it re-enters the MATLAB interpreter once per matrix-vector
product (on R2017a and older, via ARPACK reverse communication).

I instead use **CHOLMOD supernodal Cholesky** with **METIS nested-dissection**
ordering and keep the entire Lanczos loop in C. Measured on a 3-D Laplacian,
n = 64,000 (Apple M1, Accelerate BLAS):

| | `eigs`' path | mine | ratio |
|---|---|---|---|
| factorization | 2.964 s | 0.781 s | **3.8×** |
| one solve | 0.1272 s | 0.0142 s | **9.0×** |

Shift-invert does one factorization and 40–300 solves, so the end-to-end model
gives **6.0×–8.2×**, widening with problem size and difficulty.

## Layout

```
docs/REQUIREMENTS.md        design, evidence, risks   <- start here
docs/BENCHMARK-PROTOCOL.md  how the 5x claim gets defended
probe/                      feasibility probes (throwaway, not the deliverable)
src/ include/               solver + MEX gateway      (not yet written)
matlab/build_mex.m          MEX build script
build_suitesparse.sh        vendored static SuiteSparse (64-bit indices)
bench/fetch_matrices.sh     SuiteSparse Matrix Collection test set
```

## Setup

```bash
./build_suitesparse.sh          # static SuiteSparse for the host arch
./bench/fetch_matrices.sh       # benchmark matrices
# then, in MATLAB:
cd matlab; build_mex
```

## Reproducing the feasibility probes

```bash
clang -O2 -I/opt/homebrew/include probe/factor_probe.c -o build/factor_probe \
      -L/opt/homebrew/lib -lcholmod -lumfpack -lsuitesparseconfig -lm
./build/factor_probe 40        # UMFPACK LU vs CHOLMOD supernodal, n=64000
./build/psd_probe   30         # the singular / semi-definite case
```

## Licence

**GPL-3.0.** This is not a free choice: the fast path links CHOLMOD's Supernodal
module, which is GPL-2-or-later, so the combined work must be GPL-compatible.

| Component | Licence |
|---|---|
| CHOLMOD Core / Cholesky | LGPL-2.1+ |
| **CHOLMOD Supernodal** (the fast path) | **GPL-2+** |
| UMFPACK (indefinite fallback) | GPL-2+ |
| AMD, CAMD, COLAMD, CCOLAMD | BSD-3 |
| METIS 5.2 | Apache-2.0 |

Dropping to a permissive licence means giving up the supernodal factorization,
which is where most of the measured speedup comes from.

## Report

The feasibility evaluation is published at
**https://moatas19m.github.io/sparse-eigs-mex/** and mirrored in
[`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md).

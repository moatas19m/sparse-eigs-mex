# speigs — fast sparse eigenpairs as a MEX-C function

`[V,D] = speigs(A,k)` — the `k` smallest- (or largest-) magnitude eigenvalues and
eigenvectors of a sparse square matrix, aiming for **≥ 5× faster than MATLAB's
`eigs`** by exploiting symmetry, positive-definiteness, and nonzero structure.

**Status: solver written, compiled and validated. MEX gateway written, not yet run
against MATLAB.**

Read [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) for the design, or the
[published report](https://moatas19m.github.io/sparse-eigs-mex/) for the results.

## Result

Measured end to end against the cost `eigs` pays for the same work (UMFPACK LU plus
the same number of triangular solves), on 3-D Laplacians, k = 6, Apple M1:

| n | speedup |
|---:|---:|
| 8,000 | 3.60× |
| 15,625 | 3.72× |
| 32,768 | 3.90× |
| 64,000 | **5.19×** (7-trial median 5.64×) |
| 97,336 | **7.15×** |

**The 5× requirement is met from n ≈ 58,000 upward, and missed below it.** The
shortfall at small n is structural: UMFPACK's LU produces less fill on smaller
problems, so the advantage of exploiting symmetry shrinks with it. Changing the
fill-reducing ordering does not close it.

## Why it is faster where it is

For the smallest-magnitude case `eigs` shift-inverts using MATLAB's `lu`, i.e.
**UMFPACK unsymmetric LU — even when the matrix is symmetric positive definite** —
with AMD ordering, and on R2017a it re-enters the MATLAB interpreter once per
matrix-vector product via ARPACK's reverse communication.

This uses **CHOLMOD supernodal Cholesky** with **METIS nested dissection** and keeps
the whole Lanczos loop in C. Method: thick-restart Lanczos with full
reorthogonalisation over a pluggable operator, so `'sm'` and `'lm'` share one path.

## Validation

`make test` — 39 checks, 0 failures, under both clang and gcc. Correctness is
checked against the **closed-form** Laplacian spectra rather than another numerical
code: eigenvalues agree to 1e-12, the singular PSD case returns λ₀ = -4.3e-19 with a
constant null vector, and all three copies of a multiplicity-3 eigenvalue are found
with mutually orthogonal eigenvectors.

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

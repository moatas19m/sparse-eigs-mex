# `speigs` — k smallest/largest magnitude eigenpairs of a sparse matrix, as a MEX-C function

Evaluation and design. **No solver code written yet** — this document plus the
build scaffolding is the deliverable at this stage.

Constraints fixed by the requester:

| Item | Value | Consequence |
|---|---|---|
| MATLAB version | **R2017a or older** | `eigs` is a thin wrapper over **ARPACK** (Fortran IRLM), with reverse communication crossing the MEX boundary **once per matrix-vector product**. Pre-R2018a MEX API; `-largeArrayDims` mandatory. |
| Matrix class | **SPD, possibly semi-definite** | Best case. Enables Lanczos + **supernodal Cholesky** instead of Arnoldi + unsymmetric LU. The semi-definite sub-case is the main correctness trap. |
| Dominant mode | **k smallest magnitude** | Cost is dominated by the sparse factorization, which is exactly where the win is. |
| Licence | GPL acceptable (academic) | Unlocks SuiteSparse CHOLMOD (supernodal, GPL-2+) and UMFPACK. |
| Bar | **≥ 5× faster than `eigs`** | See "Is 5× achievable?" — measured evidence below. |

---

## 1. Why `eigs` is slow here, precisely

For `sigma = 0` / `'sm'`, `eigs` does **shift-invert**: it factorizes `A` and runs
Lanczos on `A⁻¹`. Two costs dominate:

1. **One sparse factorization.** `eigs` calls MATLAB's `lu`, i.e. **UMFPACK
   unsymmetric multifrontal LU** — even when `A` is symmetric positive definite.
2. **One triangular solve per iteration**, typically 40–300 of them for `k ≈ 6`.

`eigs` does *not* use Cholesky for `A`, does *not* use nested-dissection ordering,
and does *not* inspect the nonzero structure for reducibility, disconnection, or
bandedness. On R2017a and older there is a third cost: ARPACK's reverse
communication returns control to the MATLAB interpreter on **every** matvec.

My replacement attacks all three.

---

## 2. Is 5× achievable? — measured, not asserted

I built a standalone probe (`probe/factor_probe.c`, `probe/psd_probe.c`) that times
the two factorization paths on the same SPD matrix — a 3-D 7-point Laplacian, which
has realistic FEM-like fill. Apple M1 (4 P-cores), Accelerate BLAS, SuiteSparse 7.14.0.

**n = 64,000, nnz = 438,400:**

| | UMFPACK LU (what `eigs` uses) | CHOLMOD supernodal LLᵀ + METIS (mine) | ratio |
|---|---|---|---|
| analyze | 0.252 s | 0.517 s | 0.49× |
| factorize | 2.712 s | 0.265 s | **10.2×** |
| **factorization total** | **2.964 s** | **0.781 s** | **3.8×** |
| one solve | 0.1272 s | 0.0142 s | **9.0×** |
| nnz(L(+U)) *actual* | 4.12e7 | 1.44e7 | 2.87× |
| flops *actual* | 6.53e10 | 1.62e10 | 4.03× |

The flop ratio is ~4× (half from exploiting symmetry, half from METIS nested
dissection beating AMD in 3-D); the rest of the wall-clock win is supernodal
**BLAS-3** efficiency over UMFPACK's smaller frontal matrices.

**End-to-end model** for `k = 6` (factorization + n_solves × solve):

| iterations | `eigs` path | mine | speedup |
|---|---|---|---|
| 40 solves | 8.05 s | 1.35 s | **5.97×** |
| 100 solves | 15.68 s | 2.20 s | **7.13×** |
| 300 solves | 41.1 s | 5.04 s | **8.15×** |

The speedup **grows with iteration count**, because the per-solve advantage (9.0×)
exceeds the factorization advantage (3.8×). Harder problems favour mine.
This is before threading gains, before eliminating ARPACK's per-matvec interpreter
round trip, and before any structure exploitation.

**Verdict: the 5× bar is met with margin for n ≳ 20,000, and the margin widens
with problem size and difficulty.**

### Where it is at risk — stated plainly

- **Small matrices.** At n = 4,096 the measured factorization+solve advantage is
  only ≈ 3.5×. There I must clear the bar via the dense LAPACK fallback and by
  eliminating `eigs`' per-iteration overhead, not via the factorization. For a
  specific small matrix I may land at 3–4×.
- **If his real problems already run in under ~0.1 s**, a 5× speedup is
  academic. Worth confirming the sizes actually in use.
- **Very cheap-to-factor matrices** (near-banded, tiny fill): the fill ratio
  collapses toward the pure-symmetry factor of 2, narrowing the factorization win.
  The solve win largely survives.
- **All numbers above are component-level measurements plus a model.** They are
  not an end-to-end `eigs` benchmark, because there is no MATLAB on this machine
  (see §7, Blockers).

---

## 3. The semi-definite case — the real correctness trap

He explicitly flagged "possibly semi-definite". If `A` is singular, shift-invert at
σ = 0 asks us to invert a singular matrix. I probed this on a Neumann Laplacian
(row sums zero ⇒ PSD, rank n−1, null space = constants), n = 27,000:

```
cholmod factorize -> status = CHOLMOD_NOT_POSDEF, L->minor = 26999 of 27000
umfpack numeric   -> status = UMFPACK_OK,         rcond = 9.075e-13
```

Two things follow.

1. **CHOLMOD detects the rank deficiency exactly** and reports the failing column,
   giving us a clean, free rank signal.
2. **UMFPACK reports success on a singular matrix** with `rcond ≈ 9e-13`. This is
   the path `eigs` takes: it proceeds on a numerically meaningless factorization.
   So on semi-definite input my function is not merely faster — it is *more
   correct*. This is worth demonstrating to him explicitly.

**Handling.** Estimate `‖A‖₁`; factorize `A + δI` with δ escalating geometrically
from `‖A‖₁·√eps`. Probed: succeeds at every δ tested, down to δ = 1.2e-11.
Run Lanczos on `(A+δI)⁻¹`, then **discard the shifted eigenvalue estimates and
recompute each λ as a Rayleigh quotient `vᵀAv/vᵀv` against the original A**, and
report residuals `‖Av − λv‖` against the original A. Consequence: **δ affects only
the convergence rate, never the final accuracy.**

**Multiplicity.** A PSD matrix with a d-dimensional null space has λ = 0 with
multiplicity d. Plain Lanczos famously misses repeated eigenvalues. I use **block
Lanczos** (block size b ≥ expected multiplicity) with full reorthogonalization and
explicit locking. This is a correctness requirement, not an optimization.

---

## 4. Exploiting nonzero structure

His requirement — "exploit any existing nonzero structure" — is where the
order-of-magnitude wins live. All tests are O(nnz) and run before any numerics.

| Structure | Test | Payoff |
|---|---|---|
| Diagonal | pattern check | eigenvalues = diagonal, O(n log n). Done. |
| Triangular | all i ≥ j (or ≤) | eigenvalues = diagonal, O(nnz). Done. |
| **Disconnected / block-diagonal** | connected components (union-find) | Solve each block independently and merge. Superlinear saving. `eigs` does none of this. Common in assembled/substructured FEM. |
| **Narrow bandwidth** | bandwidth after RCM | LAPACK banded Cholesky `dpbtrf`; far faster and lower memory than any sparse factorization. |
| Zero rows/cols | O(nnz) | Exact zero eigenvalues; deflate them out immediately. |
| Symmetry | pattern + value compare | Lanczos not Arnoldi; Cholesky not LU. |
| Small n or large k/n | `n ≤ ~1000` or `k > n/4` | Dense LAPACK `dsyevr` (MRRR, selected eigenvalues). Beats any Krylov method and is bulletproof — `eigs` is notoriously poor in this regime. |

Note: for a *non*-symmetric matrix the analogous big win is block-triangularization
via Tarjan SCC (spectrum = union of diagonal blocks). Not needed for SPD, but it is
how I keep the "works for any matrix" promise if the SPD assumption is ever relaxed.

---

## 5. Design

**API — drop-in compatible with `eigs`:**
```matlab
d          = speigs(A, k)            % eigenvalues only (skips vector recovery)
[V,D]      = speigs(A, k)            % k smallest magnitude (default)
[V,D,flag] = speigs(A, k, 'sm'|'lm')
[V,D,flag] = speigs(A, k, opts)      % tol, maxit, p, sigma, v0, block, threads, ordering, reuse
```

**Pipeline:**
0. **Zero-copy binding.** MATLAB's sparse CSC *is* CHOLMOD's format with
   `itype = CHOLMOD_LONG`. I wrap `jc`/`ir`/`pr` with no conversion and no copy.
1. **Structure analysis** (§4) — may terminate early or reduce the problem.
2. **Definiteness and shift** (§3) — attempt Cholesky at σ=0; escalate δ if not PD.
3. **Ordering**, chosen from measured data. METIS costs ~0.45 s more analysis at
   n=64k but saves 0.22 s per factorization and 7.4 ms per solve versus AMD;
   break-even at ≈ 30 solves. Since shift-invert does 40–300, **METIS for n ≳ 20k,
   AMD below.**
4. **Block Lanczos, thick restart** (Wu–Simon) on `(A+δI)⁻¹`:
   - `cholmod_l_solve2` to reuse workspace — no malloc/free per iteration, and it
     takes multiple right-hand sides, so a block turns BLAS-2 solves into BLAS-3.
   - Full reorthogonalization against the n×p basis (p = max(2k, k+10)); cost
     O(np) per step is negligible against the solve, and it eliminates ghost
     eigenvalues.
   - Lock converged Ritz pairs; deflate.
   - Deterministic seeded start vector (reproducible run to run — `eigs` is not).
5. **Recovery.** Projected eigenproblem via LAPACK `dstemr`; re-orthonormalize;
   Rayleigh-quotient refinement against original A; undo Stage-1 permutations.
6. **Robustness.** `utIsInterruptPending()` polled per restart so Ctrl-C works;
   cleanup on every error path before `mexErrMsgIdAndTxt`; explicit memory guard on
   the n×p basis.

**Stretch goal:** cache the factorization keyed by matrix contents, so parameter
sweeps and continuation loops over the same `A` skip refactorization entirely.
Potentially >20× on that workflow.

---

## 6. Benchmark protocol — how I prove the 5×

A speed claim a professor will accept has to be defended, not asserted.
See `docs/BENCHMARK-PROTOCOL.md`. Essentials:

- **Match accuracy, not tolerance.** Compare at equal achieved residual
  `‖Av − λv‖/(‖A‖‖v‖)`; report residuals for both codes. A faster wrong answer is
  not faster.
- **Verify same eigenvalues found**, to tolerance, as a set.
- `timeit`, median of ≥ 7 runs, with warm-up (JIT, MEX load, first-touch paging).
- Report thread counts for both; give single-threaded numbers too, so the win
  cannot be dismissed as "you just used more cores".
- **Real matrices** from the SuiteSparse Matrix Collection, not `sprandsym` —
  random sparse matrices have no structure, are unrepresentative, and would
  actually flatter mine on fill.
- Report **per-matrix** speedups, the geometric mean, **and the worst case**.

---

## 7. Blockers and open questions

1. **BLOCKING — no MATLAB.** Not on this machine, so the end-to-end benchmark
   cannot be run here. Need: his **OS and architecture**, and exact MATLAB version.
2. **R2017a toolchain reality.** R2017a is Intel-only and expects Xcode ≈ 8.x.
   This machine is Apple Silicon running macOS 26 / clang 17 — R2017a will not
   build (or likely run) here. The MEX almost certainly must be compiled on his
   machine. Homebrew's arm64 SuiteSparse cannot link into an x86_64 `.mexmaci64`,
   so I ship a **vendored static SuiteSparse build** (`build_suitesparse.sh`)
   targeting his platform. Static linking also avoids dylib/rpath deployment pain.
3. **ABI:** pre-R2018a `mwIndex` is 32-bit unless `-largeArrayDims`. I require
   `-largeArrayDims` and must build SuiteSparse with matching 64-bit indices
   (`cholmod_l_*`). Getting this wrong yields silent corruption on large matrices.
4. **What sizes does he actually run?** Drives whether this is worth doing at all,
   and where to tune the dense/sparse and AMD/METIS thresholds.
5. Are the matrices ever **complex Hermitian**, or always real symmetric?

## 8. Dependency licences

| Component | Licence |
|---|---|
| CHOLMOD Core/Cholesky | LGPL-2.1+ |
| **CHOLMOD Supernodal** | **GPL-2+** (the fast path; approved for academic use) |
| UMFPACK (indefinite fallback) | GPL-2+ |
| AMD, CAMD, COLAMD, CCOLAMD | BSD-3 |
| METIS 5.2 | Apache-2.0 |
| My code | to be chosen |

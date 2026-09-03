# Benchmark protocol

The claim is "≥ 5× faster than `eigs`". This is how I make that claim defensible.

## Rules

1. **Matched accuracy, not matched tolerance.** For every matrix, report for both
   codes the achieved residual `max_i ‖A vᵢ − λᵢ vᵢ‖₂ / (‖A‖₁ ‖vᵢ‖₂)`, computed
   against the *original* A. A timing comparison at unequal accuracy is void.
2. **Same spectrum.** Assert the two eigenvalue sets agree to
   `|λᵢ − λ̂ᵢ| ≤ tol·max(‖A‖, |λᵢ|)`. Catches "fast but converged to the wrong end".
3. **Warm-up.** Discard the first run of each code (MEX load, JIT, first-touch
   page faults). Then `timeit` or median of ≥ 7 runs; report median and MAD.
4. **Threads.** Record `maxNumCompThreads` and OMP thread count. Report a
   single-threaded column as well, so the result cannot be dismissed as extra cores.
5. **Include factorization time on both sides.** `eigs` pays it internally; so do I.
6. **Same k and same starting vector** where the API permits (`opts.v0`), to remove
   run-to-run variance from `eigs`' random start.
7. Report **per matrix**, plus geometric mean, **plus the worst case**. Publishing
   only the mean is how benchmarks lose credibility.

## Test set — SuiteSparse Matrix Collection (real problems, not `sprandsym`)

**SPD, spanning three orders of magnitude in n:**
`nos3`, `bcsstk16`, `Dubcova2`, `parabolic_fem`, `ecology2`, `thermal2`,
`G3_circuit`, `af_shell3`, `StocF-1465`, `Flan_1565`

**Singular / semi-definite** (the case he flagged) — graph and Neumann Laplacians,
where the null space is nontrivial and `eigs`' UMFPACK path silently reports success
on a singular factorization.

**Structured**, to exercise the §4 fast paths: block-diagonal (disconnected mesh),
narrow-banded, diagonal, and near-triangular.

**Adversarial to me**, reported honestly: small n (≈ 2–5k), tiny fill, clustered
spectra, and large k relative to n.

## Reported artifact

One table: matrix, n, nnz, k, `eigs` time, `speigs` time, speedup, both residuals,
both eigenvalue-set agreements, thread counts. Plus a plot of speedup vs n.

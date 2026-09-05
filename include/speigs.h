/* speigs -- k smallest/largest magnitude eigenpairs of a sparse symmetric matrix.
 *
 * The numerical core has NO MATLAB dependency: it takes raw CSC arrays and
 * returns eigenpairs, so it can be compiled, tested and profiled standalone.
 * src/speigs_mex.c is a thin gateway over this API.
 *
 * Copyright (C) 2026.  GPL-3.0 (forced by linking CHOLMOD's Supernodal module).
 */
#ifndef SPEIGS_H
#define SPEIGS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Index type. Matches MATLAB's mwIndex under -largeArrayDims and CHOLMOD's
 * cholmod_l_* (long) interface. Getting this wrong silently corrupts results
 * on matrices with more than 2^31 nonzeros, so it is fixed here, not inferred. */
typedef int64_t speigs_int;

typedef enum {
    SPEIGS_SM = 0,   /* k smallest magnitude  (shift-invert) */
    SPEIGS_LM = 1    /* k largest  magnitude  (direct Lanczos) */
} speigs_which;

/* Which path the structure analysis chose. Reported so benchmarks can say
 * *why* a given matrix was fast, rather than just that it was. */
typedef enum {
    SPEIGS_PATH_DIAG      = 0,  /* matrix is diagonal: eigenvalues read off  */
    SPEIGS_PATH_DENSE     = 1,  /* small n or large k/n: LAPACK dsyevr       */
    SPEIGS_PATH_SHIFTINV  = 2,  /* Cholesky shift-invert Lanczos (fast path) */
    SPEIGS_PATH_DIRECT    = 3,  /* plain Lanczos on A ('lm')                 */
    SPEIGS_PATH_BLOCKS    = 4,  /* split into connected components           */
    SPEIGS_PATH_LU        = 5   /* indefinite: UMFPACK LU shift-invert       */
} speigs_path;

typedef struct {
    double      tol;        /* residual tol, ||Av-lv|| <= tol*||A||_1; 0 => 1e-14 */
    speigs_int  maxit;      /* max restarts;      0 => 300                        */
    speigs_int  ncv;        /* Krylov basis size; 0 => max(2k+1, 20)              */
    int         dense_max;  /* use dense path when n <= this; -1 => 800, 0 => off */
    int         detect;     /* 1 => run structure analysis (default), 0 => off    */
    int         verbose;
    uint64_t    seed;       /* deterministic start vector; 0 => 88172645463325252 */
    int         ordering;   /* 0 = auto (size-based), 1 = AMD, 2 = METIS */
    double      shift0;     /* initial delta for a singular A, RELATIVE to ||A||_1;
                             * 0 => tuned default. Exposed because the right value
                             * is an empirical question, not a theoretical one.   */
} speigs_opts;

typedef struct {
    int         flag;       /* 0 = all k converged, 1 = partial               */
    speigs_int  nconv;
    speigs_int  restarts;
    speigs_int  nops;       /* operator applications (solves or matvecs)      */
    double      shift;      /* delta actually used in A + delta*I             */
    double      anorm;      /* ||A||_1                                        */
    speigs_path path;
    speigs_int  ncomp;      /* connected components detected                  */
    double      t_analyze, t_factor, t_iter;
    double      t_op, t_ortho, t_resid, t_ritz;  /* iter-phase breakdown */
} speigs_info;

void speigs_default_opts(speigs_opts *o);

/* A: n-by-n symmetric in CSC. Full storage (both triangles, as MATLAB stores
 * sparse symmetric matrices) or upper triangle only -- either works; only the
 * upper triangle is read. Row indices must be sorted within each column.
 *
 * lambda[k]      out, eigenvalues, sorted by |lambda|
 * V[n*k]         out, eigenvectors, column-major; may be NULL to skip recovery
 * resid[k]       out, ||A v - lambda v||_2; may be NULL
 *
 * Returns 0 on success, negative on error (see speigs_errmsg). */
int speigs(speigs_int n,
           const speigs_int *Ap, const speigs_int *Ai, const double *Ax,
           speigs_int k, speigs_which which, const speigs_opts *opts,
           double *lambda, double *V, double *resid, speigs_info *info);

const char *speigs_errmsg(int code);

#define SPEIGS_OK             0
#define SPEIGS_ERR_ARG      (-1)
#define SPEIGS_ERR_MEM      (-2)
#define SPEIGS_ERR_FACTOR   (-3)
#define SPEIGS_ERR_LAPACK   (-4)
#define SPEIGS_ERR_INTERNAL (-5)

#ifdef __cplusplus
}
#endif
#endif /* SPEIGS_H */

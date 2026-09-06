/* speigs core -- thick-restart Lanczos with full reorthogonalisation, over a
 * pluggable operator so 'sm' (shift-invert) and 'lm' (direct) share one path.
 *
 * No MATLAB dependency by design: this compiles and runs standalone, which is
 * what makes it testable against analytically known spectra.
 *
 * GPL-3.0.  Links CHOLMOD (Supernodal module is GPL-2+).
 */
#include "speigs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "cholmod.h"
#include "umfpack.h"

/* LAPACK/BLAS integer width. MATLAB's libmwlapack uses ptrdiff_t under
 * -largeArrayDims; a standalone build against Accelerate/OpenBLAS uses int.
 * Override with -DSPEIGS_LAPACK_INT=ptrdiff_t when building the MEX. */
#ifndef SPEIGS_LAPACK_INT
#define SPEIGS_LAPACK_INT int
#endif
typedef SPEIGS_LAPACK_INT lapack_int;

extern double ddot_ (lapack_int*, const double*, lapack_int*, const double*, lapack_int*);
extern double dnrm2_(lapack_int*, const double*, lapack_int*);
extern void   dscal_(lapack_int*, const double*, double*, lapack_int*);
extern void   dgemv_(const char*, lapack_int*, lapack_int*, const double*,
                     const double*, lapack_int*, const double*, lapack_int*,
                     const double*, double*, lapack_int*);
extern void   dsyev_(const char*, const char*, lapack_int*, double*, lapack_int*,
                     double*, double*, lapack_int*, lapack_int*);
extern void   dpbtrf_(const char*, lapack_int*, lapack_int*, double*, lapack_int*, lapack_int*);
extern void   dpbtrs_(const char*, lapack_int*, lapack_int*, lapack_int*, const double*,
                      lapack_int*, double*, lapack_int*, lapack_int*);

#define MAXI(a,b) ((a)>(b)?(a):(b))
#define MINI(a,b) ((a)<(b)?(a):(b))

static double wall(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                          return t.tv_sec + 1e-9*t.tv_nsec; }

/* ---------------------------------------------------------------- rng ---- */
/* xorshift64*, so a run is reproducible. eigs uses a random start vector and
 * is therefore not reproducible run to run; this is a deliberate difference. */
static uint64_t rng_next(uint64_t *s){
    uint64_t x=*s; x^=x>>12; x^=x<<25; x^=x>>27; *s=x; return x*0x2545F4914F6CDD1DULL;
}
static void rand_vec(uint64_t *s, speigs_int n, double *v){
    for (speigs_int i=0;i<n;i++) v[i] = (double)(rng_next(s)>>11) / 9007199254740992.0 - 0.5;
}

/* ----------------------------------------------------------- operator ---- */
typedef struct {
    int             mode;      /* 0 = A x ; 1 = Cholesky ; 2 = LU ; 3 = banded */
    cholmod_sparse *A;
    cholmod_common *cc;
    cholmod_factor *L;         /* mode 1 */
    cholmod_dense  *X, *Y, *E; /* cholmod_l_solve2 workspace, reused         */
    void           *Numeric;   /* mode 2 (UMFPACK) */
    const speigs_int *Ap, *Ai; const double *Ax;
    /* mode 3: LAPACK banded Cholesky on the RCM-permuted matrix */
    double         *AB;        /* band factor, (kd+1) x n, column major */
    speigs_int     *perm;      /* RCM permutation                       */
    double         *bw1,*bw2;  /* permute scratch                       */
    speigs_int      kd;
    speigs_int      nops;
    speigs_int      n;
} speigs_op;

/* y = A x, always -- used for true residuals regardless of the Krylov operator */
static void op_multA(speigs_op *op, const double *x, double *y){
    cholmod_dense xd, yd;
    memset(&xd,0,sizeof xd); memset(&yd,0,sizeof yd);
    xd.nrow=xd.d=op->n; xd.ncol=1; xd.nzmax=op->n; xd.x=(void*)x;
    xd.xtype=CHOLMOD_REAL; xd.dtype=CHOLMOD_DOUBLE;
    yd = xd; yd.x = y;
    double alpha[2]={1.0,0.0}, beta[2]={0.0,0.0};
    cholmod_l_sdmult(op->A, 0, alpha, beta, &xd, &yd, op->cc);
}

/* y = M x, where M is the Krylov operator */
static int op_apply(speigs_op *op, const double *x, double *y){
    op->nops++;
    if (op->mode == 0){ op_multA(op, x, y); return 0; }

    if (op->mode == 1){
        cholmod_dense bd; memset(&bd,0,sizeof bd);
        bd.nrow=bd.d=op->n; bd.ncol=1; bd.nzmax=op->n; bd.x=(void*)x;
        bd.xtype=CHOLMOD_REAL; bd.dtype=CHOLMOD_DOUBLE;
        /* solve2 reuses X/Y/E across calls: no malloc per iteration, and this
         * is called hundreds of times, so it matters. */
        if (!cholmod_l_solve2(CHOLMOD_A, op->L, &bd, NULL, &op->X, NULL,
                              &op->Y, &op->E, op->cc)) return SPEIGS_ERR_FACTOR;
        memcpy(y, op->X->x, (size_t)op->n * sizeof(double));
        return 0;
    }
    if (op->mode == 3){
        /* permute -> banded solve -> unpermute */
        for (speigs_int i=0;i<op->n;i++) op->bw1[i] = x[op->perm[i]];
        lapack_int N=(lapack_int)op->n, KD=(lapack_int)op->kd, one=1,
                   ldab=(lapack_int)(op->kd+1), ierr=0;
        dpbtrs_("U",&N,&KD,&one,op->AB,&ldab,op->bw1,&N,&ierr);
        if (ierr) return SPEIGS_ERR_FACTOR;
        for (speigs_int i=0;i<op->n;i++) y[op->perm[i]] = op->bw1[i];
        return 0;
    }
    /* mode 2: UMFPACK LU (indefinite fallback) */
    double ctl[UMFPACK_CONTROL]; umfpack_dl_defaults(ctl);
    int st = umfpack_dl_solve(UMFPACK_A, op->Ap, op->Ai, op->Ax,
                              y, x, op->Numeric, ctl, NULL);
    return (st < 0) ? SPEIGS_ERR_FACTOR : 0;
}

/* ------------------------------------------------------- orthogonalise ---- */
/* w <- (I - B B^T) w, applied twice (DGKS). If hout, accumulate the projection
 * coefficients -- that is exactly the projected-matrix column, so full
 * reorthogonalisation gives us the thick-restart arrow structure for free. */
static void orth_block(speigs_int n, speigs_int m, const double *B, double *w,
                       double *hout, double *tmp){
    if (m <= 0) return;
    lapack_int N=(lapack_int)n, M=(lapack_int)m, one=1;
    double p1=1.0, m1=-1.0, z=0.0;
    for (int pass=0; pass<2; pass++){
        dgemv_("T",&N,&M,&p1,B,&N,w,&one,&z,tmp,&one);
        dgemv_("N",&N,&M,&m1,B,&N,tmp,&one,&p1,w,&one);
        if (hout) for (speigs_int i=0;i<m;i++) hout[i]+=tmp[i];
    }
}

/* index sort by |theta| descending */
typedef struct { double key; int idx; } kv_t;
static int kv_desc(const void*a,const void*b){
    double x=((const kv_t*)a)->key, y=((const kv_t*)b)->key;
    return (x<y)-(x>y);
}
static int kv_asc(const void*a,const void*b){
    double x=((const kv_t*)a)->key, y=((const kv_t*)b)->key;
    return (x>y)-(x<y);
}

/* --------------------------------------------------------- the Lanczos ---- */
static int lanczos(speigs_op *op, speigs_int n, speigs_int k,
                   const speigs_opts *o, double anorm,
                   double *lambda, double *Vout, double *resid,
                   speigs_info *info)
{
    int rc = SPEIGS_OK;
    speigs_int ncv = o->ncv ? o->ncv : MAXI(2*k+1, 20);
    if (ncv > n)   ncv = n;
    if (ncv <= k)  ncv = MINI(n, k+1);
    if (ncv < 2)   ncv = MINI(n, 2);
    speigs_int maxit = o->maxit ? o->maxit : 300;
    double tol = o->tol > 0 ? o->tol : 1e-14;
    double rtol = tol * MAXI(anorm, 1.0);

    double *V=NULL,*H=NULL,*Hs=NULL,*theta=NULL,*tmp=NULL,*w=NULL,*Ax=NULL,
           *x=NULL,*Vn=NULL,*lwork=NULL,*thsave=NULL;
    kv_t  *ord=NULL;
    lapack_int lw = 0, ierr = 0, M, one = 1;

    V   = (double*)calloc((size_t)n*(ncv+1), sizeof(double));
    Vn  = (double*)calloc((size_t)n*(ncv+1), sizeof(double));
    H   = (double*)calloc((size_t)ncv*ncv, sizeof(double));
    Hs  = (double*)calloc((size_t)ncv*ncv, sizeof(double));
    theta=(double*)calloc(ncv,sizeof(double));
    thsave=(double*)calloc(ncv,sizeof(double));
    tmp = (double*)calloc(MAXI(ncv,4), sizeof(double));
    w   = (double*)calloc(n,sizeof(double));
    Ax  = (double*)calloc(n,sizeof(double));
    x   = (double*)calloc(n,sizeof(double));
    ord = (kv_t*) calloc(ncv,sizeof(kv_t));
    if(!V||!Vn||!H||!Hs||!theta||!thsave||!tmp||!w||!Ax||!x||!ord){ rc=SPEIGS_ERR_MEM; goto done; }

    /* LAPACK workspace query for the small dense symmetric eigenproblem */
    { M=(lapack_int)ncv; double q; lapack_int lq=-1;
      dsyev_("V","U",&M,Hs,&M,theta,&q,&lq,&ierr);
      lw = (lapack_int)q; if (lw < 4*(lapack_int)ncv) lw = 4*(lapack_int)ncv;
      lwork=(double*)calloc((size_t)lw,sizeof(double));
      if(!lwork){ rc=SPEIGS_ERR_MEM; goto done; } }

    uint64_t seed = o->seed ? o->seed : 88172645463325252ULL;

    /* start vector: deterministic, so runs are reproducible */
    rand_vec(&seed, n, w);
    { lapack_int N=(lapack_int)n; double nr=dnrm2_(&N,w,&one);
      if (nr==0.0){ rc=SPEIGS_ERR_INTERNAL; goto done; }
      double s=1.0/nr; dscal_(&N,&s,w,&one); memcpy(V,w,(size_t)n*sizeof(double)); }

    speigs_int j = 0, restarts = 0, nconv = 0;

    for (;;) {
        /* ---- extend the basis from j to ncv ---- */
        for (; j < ncv; j++){
            { double ts=wall();
              rc = op_apply(op, V + (size_t)j*n, w);
              info->t_op += wall()-ts; }
            if (rc) goto done;

            { double ts=wall();
              for (speigs_int i=0;i<=j;i++) H[i + (size_t)j*ncv] = 0.0;
              orth_block(n, j+1, V, w, H + (size_t)j*ncv, tmp);
              info->t_ortho += wall()-ts; }

            lapack_int N=(lapack_int)n;
            double beta = dnrm2_(&N,w,&one), beta_rec = beta;

            /* Invariant subspace reached. Inject a fresh vector orthogonal to
             * everything found: this is what lets repeated eigenvalues (the
             * null space of a semi-definite A) be discovered at all. */
            if (beta <= 1e-13 * MAXI(anorm,1.0)){
                beta_rec = 0.0;
                for (int att=0; att<3 && beta <= 1e-13*MAXI(anorm,1.0); att++){
                    rand_vec(&seed, n, w);
                    orth_block(n, j+1, V, w, NULL, tmp);
                    beta = dnrm2_(&N,w,&one);
                }
                if (beta <= 1e-13*MAXI(anorm,1.0)){ ncv = j+1; break; }
            }
            { double s=1.0/beta; dscal_(&N,&s,w,&one); }
            memcpy(V + (size_t)(j+1)*n, w, (size_t)n*sizeof(double));
            if (j+1 < ncv){
                H[(j+1) + (size_t)j*ncv] = beta_rec;
                H[j + (size_t)(j+1)*ncv] = beta_rec;
            }
        }

        /* ---- projected eigenproblem (dense, ncv is small) ---- */
        speigs_int m = ncv;
        for (speigs_int c=0;c<m;c++)
            for (speigs_int r=0;r<m;r++) Hs[r + (size_t)c*m] = H[r + (size_t)c*ncv];
        M=(lapack_int)m;
        dsyev_("V","U",&M,Hs,&M,theta,lwork,&lw,&ierr);
        if (ierr){ rc=SPEIGS_ERR_LAPACK; goto done; }

        for (speigs_int i=0;i<m;i++){ ord[i].key=fabs(theta[i]); ord[i].idx=(int)i; }
        qsort(ord,(size_t)m,sizeof(kv_t),kv_desc);   /* largest |theta| first */

        /* ---- true residuals against A for the k wanted Ritz pairs ---- */
        nconv = 0;
        speigs_int nwant = MINI(k,m);
        double tr0 = wall();
        for (speigs_int t=0;t<nwant;t++){
            int i = ord[t].idx;
            lapack_int N=(lapack_int)n, MM=(lapack_int)m;
            double p1=1.0, z=0.0;
            dgemv_("N",&N,&MM,&p1,V,&N,Hs+(size_t)i*m,&one,&z,x,&one);
            double xn = dnrm2_(&N,x,&one);
            if (xn > 0){ double s=1.0/xn; dscal_(&N,&s,x,&one); }

            op_multA(op, x, Ax);
            double lam = ddot_(&N,x,&one,Ax,&one);      /* Rayleigh quotient   */
            double r2 = 0.0;                            /* ||A x - lam x||_2   */
            for (speigs_int q=0;q<n;q++){ double d=Ax[q]-lam*x[q]; r2+=d*d; }
            double r = sqrt(r2);

            if (Vout)  memcpy(Vout + (size_t)t*n, x, (size_t)n*sizeof(double));
            lambda[t] = lam;
            if (resid) resid[t] = r;
            if (r <= rtol) nconv++;
        }
        info->t_resid += wall()-tr0;

        if (nconv >= MINI(k,m) || restarts >= maxit || m < ncv) break;
        restarts++;

        /* ---- thick restart: retain the top-l Ritz vectors ---- */
        speigs_int l = MINI(k + 5, ncv - 1);
        if (l < 1) l = 1;
        speigs_int nret = 0;
        double tk0 = wall();
        for (speigs_int t=0; t<m && nret<l; t++){
            int i = ord[t].idx;
            lapack_int N=(lapack_int)n, MM=(lapack_int)m; double p1=1.0, z=0.0;
            dgemv_("N",&N,&MM,&p1,V,&N,Hs+(size_t)i*m,&one,&z,
                   Vn+(size_t)nret*n,&one);
            orth_block(n, nret, Vn, Vn+(size_t)nret*n, NULL, tmp);
            double nr = dnrm2_(&N, Vn+(size_t)nret*n, &one);
            if (nr < 1e-10) continue;                 /* collapsed: drop it */
            double s=1.0/nr; dscal_(&N,&s,Vn+(size_t)nret*n,&one);
            thsave[nret] = theta[i];
            nret++;
        }
        /* the (m+1)-th Lanczos vector continues the recurrence */
        info->t_ritz += wall()-tk0;
        memcpy(w, V + (size_t)m*n, (size_t)n*sizeof(double));
        orth_block(n, nret, Vn, w, NULL, tmp);
        { lapack_int N=(lapack_int)n; double nr=dnrm2_(&N,w,&one);
          if (nr < 1e-10){ rand_vec(&seed,n,w); orth_block(n,nret,Vn,w,NULL,tmp);
                           nr=dnrm2_(&N,w,&one); }
          if (nr < 1e-10){ break; }
          double s=1.0/nr; dscal_(&N,&s,w,&one); }
        memcpy(Vn + (size_t)nret*n, w, (size_t)n*sizeof(double));

        memcpy(V, Vn, (size_t)n*(nret+1)*sizeof(double));
        memset(H, 0, (size_t)ncv*ncv*sizeof(double));
        for (speigs_int c=0;c<nret;c++) H[c + (size_t)c*ncv] = thsave[c];
        /* the arrow couplings H[0..nret, nret] are regenerated by the full
         * reorthogonalisation on the next step -- nothing else to do. */
        j = nret;
    }

    info->nconv    = nconv;
    info->restarts = restarts;
    info->nops     = op->nops;
    info->flag     = (nconv >= MINI(k,ncv)) ? 0 : 1;

done:
    free(V); free(Vn); free(H); free(Hs); free(theta); free(thsave);
    free(tmp); free(w); free(Ax); free(x); free(ord); free(lwork);
    return rc;
}

/* ------------------------------------------------------------- dense ------ */
static int dense_path(speigs_int n, const speigs_int *Ap, const speigs_int *Ai,
                      const double *Ax, speigs_int k, speigs_which which,
                      double *lambda, double *V, double *resid, speigs_info *info)
{
    double *D = (double*)calloc((size_t)n*n, sizeof(double));
    double *ev= (double*)calloc(n, sizeof(double));
    if(!D||!ev){ free(D); free(ev); return SPEIGS_ERR_MEM; }
    for (speigs_int c=0;c<n;c++)
        for (speigs_int p=Ap[c];p<Ap[c+1];p++){
            speigs_int r=Ai[p];
            D[r + (size_t)c*n] = Ax[p];          /* full storage assumed sym */
            D[c + (size_t)r*n] = Ax[p];
        }
    lapack_int N=(lapack_int)n, lw=-1, ierr=0; double q;
    dsyev_("V","U",&N,D,&N,ev,&q,&lw,&ierr);
    lw=(lapack_int)q; double *wk=(double*)calloc((size_t)MAXI(lw,1),sizeof(double));
    if(!wk){ free(D); free(ev); return SPEIGS_ERR_MEM; }
    dsyev_("V","U",&N,D,&N,ev,wk,&lw,&ierr);
    free(wk);
    if (ierr){ free(D); free(ev); return SPEIGS_ERR_LAPACK; }

    kv_t *ord=(kv_t*)calloc(n,sizeof(kv_t));
    if(!ord){ free(D); free(ev); return SPEIGS_ERR_MEM; }
    for (speigs_int i=0;i<n;i++){ ord[i].key=fabs(ev[i]); ord[i].idx=(int)i; }
    qsort(ord,(size_t)n,sizeof(kv_t), which==SPEIGS_SM ? kv_asc : kv_desc);

    for (speigs_int t=0;t<k;t++){
        int i=ord[t].idx;
        lambda[t]=ev[i];
        if (V) memcpy(V+(size_t)t*n, D+(size_t)i*n, (size_t)n*sizeof(double));
        if (resid) resid[t]=0.0;   /* direct method: residual at machine level */
    }
    info->nconv=k; info->flag=0; info->path=SPEIGS_PATH_DENSE;
    free(D); free(ev); free(ord);
    return SPEIGS_OK;
}

/* -------------------------------------------------------- diagonal -------- */
static int is_diagonal(speigs_int n, const speigs_int *Ap, const speigs_int *Ai){
    for (speigs_int c=0;c<n;c++)
        for (speigs_int p=Ap[c];p<Ap[c+1];p++)
            if (Ai[p]!=c) return 0;
    return 1;
}
static int diag_path(speigs_int n, const speigs_int *Ap, const speigs_int *Ai,
                     const double *Ax, speigs_int k, speigs_which which,
                     double *lambda, double *V, double *resid, speigs_info *info)
{
    double *d=(double*)calloc(n,sizeof(double));
    kv_t  *ord=(kv_t*)calloc(n,sizeof(kv_t));
    if(!d||!ord){ free(d); free(ord); return SPEIGS_ERR_MEM; }
    for (speigs_int c=0;c<n;c++)
        for (speigs_int p=Ap[c];p<Ap[c+1];p++) if (Ai[p]==c) d[c]=Ax[p];
    for (speigs_int i=0;i<n;i++){ ord[i].key=fabs(d[i]); ord[i].idx=(int)i; }
    qsort(ord,(size_t)n,sizeof(kv_t), which==SPEIGS_SM ? kv_asc : kv_desc);
    for (speigs_int t=0;t<k;t++){
        int i=ord[t].idx; lambda[t]=d[i];
        if (V){ memset(V+(size_t)t*n,0,(size_t)n*sizeof(double)); V[(size_t)t*n+i]=1.0; }
        if (resid) resid[t]=0.0;
    }
    info->nconv=k; info->flag=0; info->path=SPEIGS_PATH_DIAG;
    free(d); free(ord); return SPEIGS_OK;
}

/* ------------------------------------------------- connected components ---- */
/* An eigenvalue of A belongs to exactly one connected component, so the global
 * k smallest by magnitude must lie among the per-component k smallest. Solving
 * the blocks independently is therefore exact, not an approximation -- and it
 * turns one large factorization into several small ones, which is superlinearly
 * cheaper. eigs does none of this. */
static speigs_int uf_find(speigs_int *p, speigs_int x){
    while (p[x]!=x){ p[x]=p[p[x]]; x=p[x]; }
    return x;
}
static void uf_union(speigs_int *p, speigs_int a, speigs_int b){
    a=uf_find(p,a); b=uf_find(p,b); if (a!=b) p[a]=b;
}
/* Returns number of components; comp[i] in [0,ncomp). Numerically zero entries
 * do not connect, so a structurally-present-but-zero row is correctly isolated. */
static speigs_int find_components(speigs_int n, const speigs_int *Ap,
                                  const speigs_int *Ai, const double *Ax,
                                  speigs_int *comp)
{
    speigs_int *p = (speigs_int*)malloc((size_t)n*sizeof(speigs_int));
    if (!p) return -1;
    for (speigs_int i=0;i<n;i++) p[i]=i;
    for (speigs_int c=0;c<n;c++)
        for (speigs_int q=Ap[c];q<Ap[c+1];q++)
            if (Ax[q] != 0.0) uf_union(p, Ai[q], c);
    speigs_int *lab=(speigs_int*)malloc((size_t)n*sizeof(speigs_int));
    if (!lab){ free(p); return -1; }
    for (speigs_int i=0;i<n;i++) lab[i]=-1;
    speigs_int nc=0;
    for (speigs_int i=0;i<n;i++){
        speigs_int r=uf_find(p,i);
        if (lab[r]<0) lab[r]=nc++;
        comp[i]=lab[r];
    }
    free(p); free(lab);
    return nc;
}

/* ------------------------------------------------------------- RCM --------- */
/* Reverse Cuthill-McKee, for the banded path. Level-set BFS from a low-degree
 * node, neighbours visited in increasing degree, then reversed. */
static void rcm_order(speigs_int n, const speigs_int *Ap, const speigs_int *Ai,
                      speigs_int *perm)
{
    speigs_int *deg=(speigs_int*)calloc(n,sizeof(speigs_int));
    char *seen=(char*)calloc(n,1);
    speigs_int *q=(speigs_int*)malloc((size_t)n*sizeof(speigs_int));
    if (!deg || !seen || !q){
        free(deg); free(seen); free(q);
        for (speigs_int i=0;i<n;i++) perm[i]=i;   /* identity on allocation failure */
        return;
    }
    for (speigs_int c=0;c<n;c++) deg[c] = Ap[c+1]-Ap[c];
    speigs_int head=0, tail=0;
    for (speigs_int start=0; start<n; start++){
        if (seen[start]) continue;
        /* seed each component at its lowest-degree unvisited node */
        speigs_int s=start, best=deg[start];
        for (speigs_int i=start;i<n;i++)
            if (!seen[i] && deg[i]<best){ best=deg[i]; s=i; }
        seen[s]=1; q[tail++]=s;
        while (head<tail){
            speigs_int v=q[head++], b0=tail;
            for (speigs_int p2=Ap[v];p2<Ap[v+1];p2++){
                speigs_int w=Ai[p2];
                if (!seen[w]){ seen[w]=1; q[tail++]=w; }
            }
            /* insertion-sort the newly added neighbours by degree */
            for (speigs_int a=b0+1;a<tail;a++){
                speigs_int key=q[a], d=deg[key], b=a-1;
                while (b>=b0 && deg[q[b]]>d){ q[b+1]=q[b]; b--; }
                q[b+1]=key;
            }
        }
    }
    for (speigs_int i=0;i<n;i++) perm[i]=q[n-1-i];      /* reverse */
    free(deg); free(seen); free(q);
}
/* bandwidth of P A P^T given perm (perm[newindex] = oldindex) */
static speigs_int bandwidth_of(speigs_int n, const speigs_int *Ap, const speigs_int *Ai,
                               const speigs_int *perm, speigs_int *iperm)
{
    for (speigs_int i=0;i<n;i++) iperm[perm[i]]=i;
    speigs_int kd=0;
    for (speigs_int c=0;c<n;c++){
        speigs_int nc=iperm[c];
        for (speigs_int q=Ap[c];q<Ap[c+1];q++){
            speigs_int nr=iperm[Ai[q]];
            speigs_int d = nr>nc ? nr-nc : nc-nr;
            if (d>kd) kd=d;
        }
    }
    return kd;
}

/* ------------------------------------------------------ block splitting ---- */
/* forward decl: each block is solved by a full recursive call. A block is
 * connected by construction, so the recursion is one level deep. */
int speigs(speigs_int, const speigs_int*, const speigs_int*, const double*,
           speigs_int, speigs_which, const speigs_opts*,
           double*, double*, double*, speigs_info*);

static int blocks_path(speigs_int n, const speigs_int *Ap, const speigs_int *Ai,
                       const double *Ax, speigs_int k, speigs_which which,
                       const speigs_opts *opts, const speigs_int *comp,
                       speigs_int ncomp, double *lambda, double *V,
                       double *resid, speigs_info *info)
{
    int rc = SPEIGS_OK;
    speigs_int *sz=NULL,*off=NULL,*gidx=NULL,*loc=NULL,*fill=NULL;
    speigs_int *sAp=NULL,*sAi=NULL; double *sAx=NULL;
    double **cvec=NULL, *clam=NULL, *cres=NULL;
    speigs_int *ccomp=NULL, *cslot=NULL;
    kv_t *ord=NULL;

    sz  =(speigs_int*)calloc(ncomp,sizeof(speigs_int));
    off =(speigs_int*)calloc(ncomp+1,sizeof(speigs_int));
    gidx=(speigs_int*)malloc((size_t)n*sizeof(speigs_int));
    loc =(speigs_int*)malloc((size_t)n*sizeof(speigs_int));
    fill=(speigs_int*)calloc(ncomp,sizeof(speigs_int));
    if(!sz||!off||!gidx||!loc||!fill){ rc=SPEIGS_ERR_MEM; goto done; }

    for (speigs_int i=0;i<n;i++) sz[comp[i]]++;
    for (speigs_int ci=0;ci<ncomp;ci++) off[ci+1]=off[ci]+sz[ci];
    /* gidx is ascending within each component, which keeps the extracted
     * sub-matrix's row indices sorted -- required by CHOLMOD. */
    for (speigs_int i=0;i<n;i++){
        speigs_int ci=comp[i]; speigs_int slot=off[ci]+fill[ci]++;
        gidx[slot]=i; loc[i]=slot-off[ci];
    }

    speigs_int maxcand=0;
    for (speigs_int ci=0;ci<ncomp;ci++) maxcand += MINI(k, sz[ci]);
    cvec =(double**)calloc(ncomp,sizeof(double*));
    clam =(double*)calloc(maxcand,sizeof(double));
    cres =(double*)calloc(maxcand,sizeof(double));
    ccomp=(speigs_int*)calloc(maxcand,sizeof(speigs_int));
    cslot=(speigs_int*)calloc(maxcand,sizeof(speigs_int));
    ord  =(kv_t*)calloc(maxcand,sizeof(kv_t));
    if(!cvec||!clam||!cres||!ccomp||!cslot||!ord){ rc=SPEIGS_ERR_MEM; goto done; }

    speigs_int ncand=0, nnzmax=Ap[n];
    sAp=(speigs_int*)malloc((size_t)(n+1)*sizeof(speigs_int));
    sAi=(speigs_int*)malloc((size_t)(nnzmax?nnzmax:1)*sizeof(speigs_int));
    sAx=(double*)malloc((size_t)(nnzmax?nnzmax:1)*sizeof(double));
    if(!sAp||!sAi||!sAx){ rc=SPEIGS_ERR_MEM; goto done; }

    for (speigs_int ci=0; ci<ncomp; ci++){
        speigs_int nc=sz[ci], knt=0;
        for (speigs_int j=0;j<nc;j++){
            speigs_int gc=gidx[off[ci]+j];
            sAp[j]=knt;
            for (speigs_int q=Ap[gc];q<Ap[gc+1];q++){
                speigs_int r=Ai[q];
                if (comp[r]==ci){ sAi[knt]=loc[r]; sAx[knt]=Ax[q]; knt++; }
            }
        }
        sAp[nc]=knt;

        speigs_int kc=MINI(k,nc);
        double *lv=(double*)calloc(kc,sizeof(double));
        double *rv=(double*)calloc(kc,sizeof(double));
        double *vv=V ? (double*)calloc((size_t)nc*kc,sizeof(double)) : NULL;
        if(!lv||!rv||(V&&!vv)){ free(lv);free(rv);free(vv); rc=SPEIGS_ERR_MEM; goto done; }

        speigs_info si;
        int r2 = speigs(nc,sAp,sAi,sAx,kc,which,opts,lv,vv,rv,&si);
        if (r2){ free(lv);free(rv);free(vv); rc=r2; goto done; }
        info->nops += si.nops;
        if (si.flag) info->flag = 1;

        cvec[ci]=vv;
        for (speigs_int t=0;t<kc;t++){
            clam[ncand]=lv[t]; cres[ncand]=rv[t];
            ccomp[ncand]=ci;   cslot[ncand]=t; ncand++;
        }
        free(lv); free(rv);
    }

    for (speigs_int i=0;i<ncand;i++){ ord[i].key=fabs(clam[i]); ord[i].idx=(int)i; }
    qsort(ord,(size_t)ncand,sizeof(kv_t), which==SPEIGS_SM?kv_asc:kv_desc);

    for (speigs_int t=0;t<k && t<ncand;t++){
        int i=ord[t].idx; speigs_int ci=ccomp[i], sl=cslot[i], nc=sz[ci];
        lambda[t]=clam[i];
        if (resid) resid[t]=cres[i];
        if (V){
            memset(V+(size_t)t*n, 0, (size_t)n*sizeof(double));
            const double *src = cvec[ci] + (size_t)sl*nc;
            for (speigs_int j=0;j<nc;j++) V[(size_t)t*n + gidx[off[ci]+j]] = src[j];
        }
    }
    info->nconv = MINI(k,ncand);
    info->ncomp = ncomp;
    info->path  = SPEIGS_PATH_BLOCKS;

done:
    if (cvec){ for (speigs_int ci=0;ci<ncomp;ci++) free(cvec[ci]); free(cvec); }
    free(sz); free(off); free(gidx); free(loc); free(fill);
    free(sAp); free(sAi); free(sAx);
    free(clam); free(cres); free(ccomp); free(cslot); free(ord);
    return rc;
}

/* ------------------------------------------------------------ driver ------ */
void speigs_default_opts(speigs_opts *o){
    memset(o,0,sizeof *o);
    o->tol=0.0; o->maxit=0; o->ncv=0; o->dense_max=-1; o->detect=1;
    o->seed=0; o->ordering=0; o->shift0=0.0; o->band_max=0;
}


/* ------------------------------------------------- complex Hermitian ------ */
/* Reduced to a real symmetric problem of twice the size rather than duplicating
 * the whole pipeline in complex arithmetic.
 *
 *   A = R + iS Hermitian  =>  R symmetric, S skew-symmetric
 *   Atilde = [[R, -S], [S, R]]   is real symmetric, 2n x 2n
 *   A(x+iy) = lambda(x+iy)  <=>  Atilde [x;y] = lambda [x;y]
 *
 * Every eigenvalue of A appears twice in Atilde, via [x;y] and [-y;x] -- which
 * are the same complex direction up to a factor of i, so the duplicates are
 * removed by a parallelism test on the recovered complex vectors.
 *
 * The cost is roughly 2x a native complex factorization. That trade is
 * deliberate: it reuses the entire tested real pipeline -- block splitting,
 * banded, dense, the semi-definite shift, multiplicity handling -- instead of
 * reimplementing all of it in complex arithmetic, where every bug would be new
 * and untested. Native complex kernels are the future optimisation, not a
 * correctness prerequisite. */
int speigs_z(speigs_int n,
             const speigs_int *Ap, const speigs_int *Ai,
             const double *Axr, const double *Axi,
             speigs_int k, speigs_which which, const speigs_opts *opts,
             double *lambda, double *Vr, double *Vi, double *resid,
             speigs_info *info)
{
    if (!Axi) return speigs(n,Ap,Ai,Axr,k,which,opts,lambda,Vr,resid,info);
    if (n<=0 || k<=0 || k>n || !Ap || !Ai || !Axr || !lambda) return SPEIGS_ERR_ARG;

    speigs_int n2 = 2*n, nnz = Ap[n], nzR=0, nzS=0;
    for (speigs_int q=0;q<nnz;q++){ if (Axr[q]!=0.0) nzR++; if (Axi[q]!=0.0) nzS++; }
    speigs_int nz2 = 2*(nzR+nzS);

    speigs_int *Bp=(speigs_int*)malloc((size_t)(n2+1)*sizeof(speigs_int));
    speigs_int *Bi=(speigs_int*)malloc((size_t)(nz2?nz2:1)*sizeof(speigs_int));
    double     *Bx=(double*)malloc((size_t)(nz2?nz2:1)*sizeof(double));
    if (!Bp||!Bi||!Bx){ free(Bp);free(Bi);free(Bx); return SPEIGS_ERR_MEM; }

    speigs_int t=0;
    for (speigs_int j=0;j<n;j++){                       /* column j: [R;S] */
        Bp[j]=t;
        for (speigs_int q=Ap[j];q<Ap[j+1];q++)
            if (Axr[q]!=0.0){ Bi[t]=Ai[q];   Bx[t]=Axr[q]; t++; }
        for (speigs_int q=Ap[j];q<Ap[j+1];q++)
            if (Axi[q]!=0.0){ Bi[t]=n+Ai[q]; Bx[t]=Axi[q]; t++; }
    }
    for (speigs_int j=0;j<n;j++){                       /* column n+j: [-S;R] */
        Bp[n+j]=t;
        for (speigs_int q=Ap[j];q<Ap[j+1];q++)
            if (Axi[q]!=0.0){ Bi[t]=Ai[q];   Bx[t]=-Axi[q]; t++; }
        for (speigs_int q=Ap[j];q<Ap[j+1];q++)
            if (Axr[q]!=0.0){ Bi[t]=n+Ai[q]; Bx[t]=Axr[q];  t++; }
    }
    Bp[n2]=t;

    /* ask for extra: each wanted eigenvalue appears twice in the embedding */
    speigs_int kk = MINI(2*k+2, n2);
    double *lam2=(double*)calloc(kk,sizeof(double));
    double *res2=(double*)calloc(kk,sizeof(double));
    double *V2  =(double*)calloc((size_t)n2*kk,sizeof(double));
    int rc = SPEIGS_OK;
    if (!lam2||!res2||!V2){ rc=SPEIGS_ERR_MEM; goto done; }

    rc = speigs(n2,Bp,Bi,Bx,kk,which,opts,lam2,V2,res2,info);
    if (rc) goto done;

    /* recover complex eigenpairs, discarding the i-multiples */
    {
    speigs_int nacc=0;
    double *ar=(double*)calloc((size_t)n*k,sizeof(double));
    double *ai=(double*)calloc((size_t)n*k,sizeof(double));
    if (!ar||!ai){ free(ar); free(ai); rc=SPEIGS_ERR_MEM; goto done; }
    for (speigs_int c=0;c<kk && nacc<k;c++){
        const double *a = V2 + (size_t)c*n2, *b = a + n;
        double nr=0.0; for (speigs_int i=0;i<n;i++) nr += a[i]*a[i]+b[i]*b[i];
        if (nr <= 0.0) continue;
        nr = 1.0/sqrt(nr);
        /* Complex Gram-Schmidt, not a pairwise parallelism test. For an
         * eigenvalue of multiplicity m the embedding yields a 2m-real-
         * dimensional space, and vectors drawn from it can be pairwise
         * non-parallel yet still C-linearly DEPENDENT -- a pairwise test
         * accepts too many and returns the same eigenvalue m+1 times. */
        double *wr=(double*)malloc((size_t)n*sizeof(double));
        double *wi=(double*)malloc((size_t)n*sizeof(double));
        if (!wr||!wi){ free(wr); free(wi); free(ar); free(ai); rc=SPEIGS_ERR_MEM; goto done; }
        for (speigs_int i=0;i<n;i++){ wr[i]=a[i]*nr; wi[i]=b[i]*nr; }
        for (int pass=0; pass<2; pass++)
            for (speigs_int u=0; u<nacc; u++){
                double re=0.0, im=0.0;            /* <v_u, w> = sum conj(u) w */
                for (speigs_int i=0;i<n;i++){
                    double ur=ar[(size_t)u*n+i], ui=ai[(size_t)u*n+i];
                    re += ur*wr[i] + ui*wi[i];
                    im += ur*wi[i] - ui*wr[i];
                }
                for (speigs_int i=0;i<n;i++){     /* w -= <u,w> u */
                    double ur=ar[(size_t)u*n+i], ui=ai[(size_t)u*n+i];
                    wr[i] -= re*ur - im*ui;
                    wi[i] -= re*ui + im*ur;
                }
            }
        double wn=0.0; for (speigs_int i=0;i<n;i++) wn += wr[i]*wr[i]+wi[i]*wi[i];
        wn = sqrt(wn);
        int dup = (wn <= 0.1);
        if (!dup){
            double inv=1.0/wn;
            for (speigs_int i=0;i<n;i++){ wr[i]*=inv; wi[i]*=inv; }
        }
        if (dup){ free(wr); free(wi); continue; }
        for (speigs_int i=0;i<n;i++){
            ar[(size_t)nacc*n+i]=wr[i];
            ai[(size_t)nacc*n+i]=wi[i];
        }
        free(wr); free(wi);
        lambda[nacc]=lam2[c];
        if (resid) resid[nacc]=res2[c];
        nacc++;
    }
    if (Vr) memcpy(Vr,ar,(size_t)n*k*sizeof(double));
    if (Vi) memcpy(Vi,ai,(size_t)n*k*sizeof(double));
    free(ar); free(ai);
    info->nconv = nacc;
    info->flag  = (nacc>=k) ? info->flag : 1;
    }

done:
    free(Bp); free(Bi); free(Bx); free(lam2); free(res2); free(V2);
    return rc;
}

const char *speigs_errmsg(int c){
    switch(c){
      case SPEIGS_OK: return "ok";
      case SPEIGS_ERR_ARG: return "invalid argument";
      case SPEIGS_ERR_MEM: return "out of memory";
      case SPEIGS_ERR_FACTOR: return "factorization failed";
      case SPEIGS_ERR_LAPACK: return "LAPACK failed";
      default: return "internal error";
    }
}

int speigs(speigs_int n, const speigs_int *Ap, const speigs_int *Ai, const double *Ax,
           speigs_int k, speigs_which which, const speigs_opts *opts,
           double *lambda, double *V, double *resid, speigs_info *info)
{
    speigs_opts defo; if(!opts){ speigs_default_opts(&defo); opts=&defo; }
    speigs_info li; if(!info) info=&li;
    memset(info,0,sizeof *info);

    if (n<=0 || !Ap || !Ai || !Ax || !lambda) return SPEIGS_ERR_ARG;
    if (k<=0 || k>n) return SPEIGS_ERR_ARG;

    double t0=wall();
    int dmax = opts->dense_max < 0 ? 800 : opts->dense_max;

    if (opts->detect && is_diagonal(n,Ap,Ai))
        return diag_path(n,Ap,Ai,Ax,k,which,lambda,V,resid,info);

    /* Dense LAPACK beats any Krylov method for small n or large k/n, and is
     * bulletproof where Krylov methods are at their worst. */
    if (dmax>0 && (n<=dmax || 4*k>n)){
        int rc = dense_path(n,Ap,Ai,Ax,k,which,lambda,V,resid,info);
        info->t_iter = wall()-t0;
        return rc;
    }

    /* Split on connected components before doing any numerical work. Exact,
     * not approximate: each eigenvalue lives in exactly one block. */
    if (opts->detect){
        speigs_int *comp=(speigs_int*)malloc((size_t)n*sizeof(speigs_int));
        if (comp){
            speigs_int ncomp = find_components(n,Ap,Ai,Ax,comp);
            if (ncomp > 1){
                int rc = blocks_path(n,Ap,Ai,Ax,k,which,opts,comp,ncomp,
                                     lambda,V,resid,info);
                free(comp);
                info->t_iter = wall()-t0;
                return rc;
            }
            free(comp);
        }
    }

    cholmod_common cc; cholmod_l_start(&cc);
    cc.supernodal = CHOLMOD_SUPERNODAL;
    cc.print = 0; cc.error_handler = NULL;

    cholmod_sparse A; memset(&A,0,sizeof A);
    A.nrow=A.ncol=n; A.nzmax=Ap[n];
    A.p=(void*)Ap; A.i=(void*)Ai; A.x=(void*)Ax;
    A.stype=1;                       /* symmetric; read upper triangle only */
    A.itype=CHOLMOD_LONG; A.xtype=CHOLMOD_REAL; A.dtype=CHOLMOD_DOUBLE;
    A.sorted=1; A.packed=1;

    double anorm = cholmod_l_norm_sparse(&A, 1, &cc);
    if (anorm <= 0) anorm = 1.0;
    info->anorm = anorm;

    speigs_op op; memset(&op,0,sizeof op);
    op.A=&A; op.cc=&cc; op.n=n; op.Ap=Ap; op.Ai=Ai; op.Ax=Ax;

    int rc = SPEIGS_OK;
    cholmod_factor *L=NULL; void *Sym=NULL,*Num=NULL;

    if (which == SPEIGS_LM){
        op.mode = 0;                        /* plain Lanczos on A */
        info->path = SPEIGS_PATH_DIRECT;
    } else {
        /* Ordering choice is measured, not guessed. End-to-end on 3-D
         * Laplacians (median of 3, k=6): at n=8k AMD wins 0.055 s vs 0.092 s
         * because METIS's analysis dominates; by n=15.6k METIS wins 0.199 s vs
         * 0.231 s; at n=64k it wins 1.50 s vs 1.62 s. Crossover sits between
         * 8k and 15.6k, so the threshold is 12k. */
        cc.nmethods = 1; cc.postorder = 1;
        cc.method[0].ordering =
            opts->ordering==1 ? CHOLMOD_AMD :
            opts->ordering==2 ? CHOLMOD_METIS :
            ((n >= 12000) ? CHOLMOD_METIS : CHOLMOD_AMD);

        /* Banded path. If RCM produces a narrow band, LAPACK's banded Cholesky
         * needs no ordering search and no supernode assembly, and works on
         * contiguous memory. Gated on measurement, not on principle: see
         * bench/bench_banded. */
        int bmax = opts->band_max ? opts->band_max : 512;
        if (bmax > 0){
            double tb=wall();
            speigs_int *pm=(speigs_int*)malloc((size_t)n*sizeof(speigs_int));
            speigs_int *ip=(speigs_int*)malloc((size_t)n*sizeof(speigs_int));
            if (pm && ip){
                rcm_order(n,Ap,Ai,pm);
                speigs_int kd = bandwidth_of(n,Ap,Ai,pm,ip);
                double cells = (double)(kd+1)*(double)n;
                /* Gate on band DENSITY, not bandwidth alone. Measured at
                 * n=50,000 with a band that is dense, LAPACK beats CHOLMOD by
                 * 1.15x at kd=1 rising to 3.87x at kd=256. But on a band that
                 * is mostly empty -- a 2-D Laplacian has bandwidth m and only 5
                 * nonzeros per row -- the two are a coin flip (0.90x to 1.19x),
                 * and banded storage burns O(n*kd) memory for zeros. So require
                 * the band to be at least 5% occupied. */
                double occupancy = (kd>0) ? (double)Ap[n] / ((double)n*(double)kd)
                                          : 1.0;
                if (kd <= bmax && cells <= 2.0e7 && occupancy >= 0.05){
                    double *AB=(double*)calloc((size_t)cells,sizeof(double));
                    if (AB){
                        speigs_int ldab=kd+1;
                        for (speigs_int col=0;col<n;col++){
                            speigs_int nc2=ip[col];
                            for (speigs_int q=Ap[col];q<Ap[col+1];q++){
                                speigs_int nr=ip[Ai[q]];
                                speigs_int lo = nr<nc2?nr:nc2, hi = nr<nc2?nc2:nr;
                                AB[(kd+lo-hi) + (size_t)hi*ldab] = Ax[q];
                            }
                        }
                        double dlt=0.0; lapack_int N=(lapack_int)n,
                            KD=(lapack_int)kd, LD=(lapack_int)ldab, ierr=0;
                        double *ABw=(double*)malloc((size_t)cells*sizeof(double));
                        if (ABw){
                            memcpy(ABw,AB,(size_t)cells*sizeof(double));
                            dpbtrf_("U",&N,&KD,ABw,&LD,&ierr);
                            if (ierr>0){         /* not PD: same capped escalation */
                                double d0=anorm*(opts->shift0>0?opts->shift0:1e-6);
                                double dcap=anorm*1e-3; if(d0>dcap) dcap=d0;
                                for (dlt=d0; dlt<=dcap*1.0000001 && ierr>0; dlt*=10.0){
                                    memcpy(ABw,AB,(size_t)cells*sizeof(double));
                                    for (speigs_int j=0;j<n;j++) ABw[kd+(size_t)j*ldab]+=dlt;
                                    ierr=0; dpbtrf_("U",&N,&KD,ABw,&LD,&ierr);
                                }
                                if (ierr>0) dlt=0.0;
                            }
                            if (ierr==0){
                                op.mode=3; op.AB=ABw; op.perm=pm; op.kd=kd;
                                op.bw1=(double*)calloc(n,sizeof(double));
                                op.bw2=(double*)calloc(n,sizeof(double));
                                info->shift=dlt; info->path=SPEIGS_PATH_BANDED;
                                info->t_factor=wall()-tb;
                                free(AB); free(ip);
                                goto have_operator;
                            }
                            free(ABw);
                        }
                        free(AB);
                    }
                }
            }
            free(pm); free(ip);
        }

        double ta=wall();
        L = cholmod_l_analyze(&A,&cc);
        info->t_analyze = wall()-ta;
        if (!L){ rc=SPEIGS_ERR_FACTOR; goto cleanup; }

        double tf=wall();
        double delta = 0.0, beta2[2]={0.0,0.0};
        cholmod_l_factorize_p(&A, beta2, NULL, 0, L, &cc);

        /* Semi-definite handling. A + delta*I is factorized instead; delta is
         * then divided out of nothing at all, because the eigenvalues are
         * recovered as Rayleigh quotients against the ORIGINAL A. So delta
         * changes only the rate of convergence, never the accuracy.
         *
         * The 1e-6 default is measured, not guessed. A "safely tiny" shift of
         * 1e-10 makes theta_0/theta_1 ~ 3e7, which pollutes the small Ritz
         * values enough that convergence never certifies: 3020 solves and a
         * partial-convergence flag, versus 40 solves at 1e-6 for identical
         * accuracy. The good region is broad (1e-6 .. 1e-3 at n=4k and n=14k);
         * 1e-6 is its conservative end. See docs/REQUIREMENTS.md section 3. */
        if (cc.status == CHOLMOD_NOT_POSDEF){
            /* The escalation is deliberately CAPPED at 1e-3*||A||. For a
             * positive semi-definite A, A + delta*I is positive definite for
             * any delta > 0, so the first attempt always succeeds; needing a
             * larger shift proves A has genuinely negative eigenvalues. And a
             * large delta would silently retarget shift-invert to the
             * eigenvalues nearest -delta instead of those nearest zero -- the
             * wrong answer, returned confidently. So we stop and use LU. */
            double d0 = anorm * (opts->shift0 > 0 ? opts->shift0 : 1e-6);
            double dcap = anorm * 1e-3;
            if (d0 > dcap) dcap = d0;
            for (delta = d0; delta <= dcap*1.0000001; delta *= 10.0){
                beta2[0]=delta;
                cholmod_l_factorize_p(&A, beta2, NULL, 0, L, &cc);
                if (cc.status != CHOLMOD_NOT_POSDEF) break;
            }
            if (cc.status == CHOLMOD_NOT_POSDEF) delta = 0.0;
        }

        if (cc.status == CHOLMOD_OK){
            op.mode=1; op.L=L; info->shift=delta; info->path=SPEIGS_PATH_SHIFTINV;
        } else {
            /* Genuinely indefinite: a large shift would retarget shift-invert
             * to the wrong end of the spectrum, so fall back to an unsymmetric
             * LU at sigma = 0. Slower -- this is the correctness path, and it
             * is what preserves "works for any matrix". */
            cholmod_l_free_factor(&L,&cc); L=NULL;
            double ctl[UMFPACK_CONTROL]; umfpack_dl_defaults(ctl);
            if (umfpack_dl_symbolic(n,n,Ap,Ai,Ax,&Sym,ctl,NULL) < 0){ rc=SPEIGS_ERR_FACTOR; goto cleanup; }
            if (umfpack_dl_numeric(Ap,Ai,Ax,Sym,&Num,ctl,NULL) < 0){ rc=SPEIGS_ERR_FACTOR; goto cleanup; }
            op.mode=2; op.Numeric=Num; info->shift=0.0; info->path=SPEIGS_PATH_LU;
        }
        info->t_factor = wall()-tf;
    }
have_operator:

    { double ti=wall();
      rc = lanczos(&op,n,k,opts,anorm,lambda,V,resid,info);
      info->t_iter = wall()-ti; }

    /* sort output by |lambda|: ascending for 'sm', descending for 'lm' */
    if (rc==SPEIGS_OK){
        kv_t *ord=(kv_t*)calloc(k,sizeof(kv_t));
        double *lc=(double*)calloc(k,sizeof(double));
        double *rc2=resid?(double*)calloc(k,sizeof(double)):NULL;
        double *Vc=V?(double*)calloc((size_t)n*k,sizeof(double)):NULL;
        if (ord&&lc&&(!resid||rc2)&&(!V||Vc)){
            for (speigs_int i=0;i<k;i++){ ord[i].key=fabs(lambda[i]); ord[i].idx=(int)i; }
            qsort(ord,(size_t)k,sizeof(kv_t), which==SPEIGS_SM?kv_asc:kv_desc);
            memcpy(lc,lambda,(size_t)k*sizeof(double));
            if (rc2) memcpy(rc2,resid,(size_t)k*sizeof(double));
            if (Vc)  memcpy(Vc,V,(size_t)n*k*sizeof(double));
            for (speigs_int t=0;t<k;t++){
                int i=ord[t].idx; lambda[t]=lc[i];
                if (resid) resid[t]=rc2[i];
                if (V) memcpy(V+(size_t)t*n, Vc+(size_t)i*n, (size_t)n*sizeof(double));
            }
        }
        free(ord); free(lc); free(rc2); free(Vc);
    }

cleanup:
    free(op.AB); free(op.perm); free(op.bw1); free(op.bw2);
    if (op.X) cholmod_l_free_dense(&op.X,&cc);
    if (op.Y) cholmod_l_free_dense(&op.Y,&cc);
    if (op.E) cholmod_l_free_dense(&op.E,&cc);
    if (L)    cholmod_l_free_factor(&L,&cc);
    if (Num)  umfpack_dl_free_numeric(&Num);
    if (Sym)  umfpack_dl_free_symbolic(&Sym);
    cholmod_l_finish(&cc);
    return rc;
}

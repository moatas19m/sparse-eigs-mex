/* FEASIBILITY PROBE -- not part of the deliverable.
 * Question: on an SPD matrix, how much faster is CHOLMOD supernodal LL^T
 * (what we would use) than UMFPACK unsymmetric LU (what eigs' shift-invert uses)?
 * The answer bounds the achievable speedup for 'smallest magnitude'.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "suitesparse/cholmod.h"
#include "suitesparse/umfpack.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

/* 3D 7-point Laplacian on N^3 grid, SPD, in CSC (int64). Classic FEM-like fill. */
static void laplacian3d(int64_t N, int64_t **Ap_o, int64_t **Ai_o, double **Ax_o, int64_t *n_o){
    int64_t n = N*N*N, nzmax = 7*n, k=0;
    int64_t *Ap = malloc((n+1)*sizeof(int64_t)), *Ai = malloc(nzmax*sizeof(int64_t));
    double  *Ax = malloc(nzmax*sizeof(double));
    for (int64_t z=0; z<N; z++) for (int64_t y=0; y<N; y++) for (int64_t x=0; x<N; x++){
        int64_t c = x + N*(y + N*z); Ap[c]=k;
        int64_t nb[6]; int m=0;
        if(z>0) nb[m++]=c-N*N; if(y>0) nb[m++]=c-N; if(x>0) nb[m++]=c-1;
        for(int i=0;i<m;i++){ Ai[k]=nb[i]; Ax[k]=-1.0; k++; }
        Ai[k]=c; Ax[k]=6.0; k++;                      /* diagonal */
        m=0; if(x<N-1) nb[m++]=c+1; if(y<N-1) nb[m++]=c+N; if(z<N-1) nb[m++]=c+N*N;
        for(int i=0;i<m;i++){ Ai[k]=nb[i]; Ax[k]=-1.0; k++; }
    }
    Ap[n]=k; *Ap_o=Ap; *Ai_o=Ai; *Ax_o=Ax; *n_o=n;
}

int main(int argc, char**argv){
    int64_t N = (argc>1)? atoi(argv[1]) : 40;
    int64_t *Ap,*Ai,n; double *Ax;
    laplacian3d(N,&Ap,&Ai,&Ax,&n);
    printf("3D Laplacian  n = %lld  nnz = %lld\n", (long long)n, (long long)Ap[n]);

    /* ---------- UMFPACK unsymmetric LU (the eigs shift-invert path) ---------- */
    double ctl[UMFPACK_CONTROL], info[UMFPACK_INFO];
    umfpack_dl_defaults(ctl);
    void *Sym=NULL,*Num=NULL;
    double t0=now();
    umfpack_dl_symbolic(n,n,Ap,Ai,Ax,&Sym,ctl,info);
    double t1=now();
    umfpack_dl_numeric(Ap,Ai,Ax,Sym,&Num,ctl,info);
    double t2=now();
    double lu_nz = info[UMFPACK_LNZ_ESTIMATE]+info[UMFPACK_UNZ_ESTIMATE];
    double lu_fl = info[UMFPACK_FLOPS_ESTIMATE];
    double *b=malloc(n*sizeof(double)), *xs=malloc(n*sizeof(double));
    for(int64_t i=0;i<n;i++) b[i]=1.0;
    double t3=now();
    for(int r=0;r<10;r++) umfpack_dl_solve(UMFPACK_A,Ap,Ai,Ax,xs,b,Num,ctl,info);
    double t4=now();
    printf("\nUMFPACK LU   analyze %7.3f s   factorize %7.3f s   TOTAL %7.3f s\n",
           t1-t0, t2-t1, t2-t0);
    printf("             nnz(L+U) %.3e   flops %.3e   solve(x10) %.4f s  [%.4f s each]\n",
           lu_nz, lu_fl, t4-t3, (t4-t3)/10);

    /* ---------- CHOLMOD supernodal Cholesky (our path) ---------- */
    cholmod_common c; cholmod_l_start(&c);
    c.supernodal = CHOLMOD_SUPERNODAL;     /* force BLAS-3 supernodal */
    c.nmethods = 1; c.method[0].ordering = CHOLMOD_METIS; c.postorder = 1;
    cholmod_sparse As;
    As.nrow=n; As.ncol=n; As.nzmax=Ap[n]; As.p=Ap; As.i=Ai; As.x=Ax; As.z=NULL;
    As.nz=NULL; As.stype=1; As.itype=CHOLMOD_LONG; As.xtype=CHOLMOD_REAL;
    As.dtype=CHOLMOD_DOUBLE; As.sorted=1; As.packed=1;
    double s0=now();
    cholmod_factor *L = cholmod_l_analyze(&As,&c);
    double s1=now();
    cholmod_l_factorize(&As,L,&c);
    double s2=now();
    cholmod_dense *bd = cholmod_l_ones(n,1,CHOLMOD_REAL,&c);
    double s3=now();
    for(int r=0;r<10;r++){ cholmod_dense *xd=cholmod_l_solve(CHOLMOD_A,L,bd,&c);
                           cholmod_l_free_dense(&xd,&c); }
    double s4=now();
    printf("\nCHOLMOD LL^T analyze %7.3f s   factorize %7.3f s   TOTAL %7.3f s\n",
           s1-s0, s2-s1, s2-s0);
    printf("             nnz(L)   %.3e   flops %.3e   solve(x10) %.4f s  [%.4f s each]\n",
           c.lnz, c.fl, s4-s3, (s4-s3)/10);
    printf("             status: %s\n", c.status==CHOLMOD_OK? "OK (SPD)" :
           (c.status==CHOLMOD_NOT_POSDEF? "NOT POSITIVE DEFINITE":"error"));

    printf("\n==> FACTORIZATION SPEEDUP  %.2fx      SOLVE SPEEDUP  %.2fx\n",
           (t2-t0)/(s2-s0), (t4-t3)/(s4-s3));
    printf("==> fill ratio nnz(L+U)/nnz(L) = %.2f   flop ratio = %.2f\n",
           lu_nz/c.lnz, lu_fl/c.fl);
    return 0;
}

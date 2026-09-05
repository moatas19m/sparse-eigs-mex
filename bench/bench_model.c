/* End-to-end cost of speigs vs the cost eigs incurs for the SAME work.
 *
 * Fair-comparison note: eigs' iteration count is not measured here (no MATLAB
 * on this machine), so it is charged the SAME number of operator applications
 * speigs actually used. That isolates exactly the claim being made -- that the
 * factorization and solve are faster -- and does not credit speigs for
 * converging in fewer iterations, which would be a separate claim.
 */
#include "speigs.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cholmod.h"
#include "umfpack.h"
static double wall(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static void lap3d(int64_t N,int64_t**Ap_o,int64_t**Ai_o,double**Ax_o,int64_t*n_o){
    int64_t n=N*N*N,k=0,*Ap=malloc((n+1)*sizeof(int64_t)),*Ai=malloc(7*n*sizeof(int64_t));
    double*Ax=malloc(7*n*sizeof(double));
    for(int64_t z=0;z<N;z++)for(int64_t y=0;y<N;y++)for(int64_t x=0;x<N;x++){
        int64_t c=x+N*(y+N*z);Ap[c]=k;int64_t lo[3];int m=0;
        if(z>0)lo[m++]=c-N*N;if(y>0)lo[m++]=c-N;if(x>0)lo[m++]=c-1;
        int64_t hi[3];int m2=0;
        if(x<N-1)hi[m2++]=c+1;if(y<N-1)hi[m2++]=c+N;if(z<N-1)hi[m2++]=c+N*N;
        for(int i=0;i<m;i++){Ai[k]=lo[i];Ax[k]=-1.0;k++;}
        Ai[k]=c;Ax[k]=6.0;k++;
        for(int i=0;i<m2;i++){Ai[k]=hi[i];Ax[k]=-1.0;k++;}
    }
    Ap[n]=k;*Ap_o=Ap;*Ai_o=Ai;*Ax_o=Ax;*n_o=n;
}
int main(int argc,char**argv){
    int64_t N=(argc>1)?atoll(argv[1]):40, k=(argc>2)?atoll(argv[2]):6;
    int64_t *Ap,*Ai,n; double *Ax; lap3d(N,&Ap,&Ai,&Ax,&n);
    printf("3-D Laplacian  n=%lld  nnz=%lld  k=%lld\n",
           (long long)n,(long long)Ap[n],(long long)k);

    /* ---- speigs, end to end ---- */
    double *lam=calloc(k,sizeof(double)),*res=calloc(k,sizeof(double));
    double *V=calloc((size_t)n*k,sizeof(double));
    speigs_opts o; speigs_default_opts(&o); speigs_info inf;
    double t0=wall(); int rc=speigs(n,Ap,Ai,Ax,k,SPEIGS_SM,&o,lam,V,res,&inf);
    double t_ours=wall()-t0;
    double rmax=0; for(int64_t i=0;i<k;i++) rmax=fmax(rmax,res[i]);
    printf("\nspeigs   TOTAL %8.3f s   (analyze %.3f  factor %.3f  iter %.3f)\n",
           t_ours,inf.t_analyze,inf.t_factor,inf.t_iter);
    printf("         rc=%d flag=%d nops=%lld restarts=%lld  max resid %.3e\n",
           rc,inf.flag,(long long)inf.nops,(long long)inf.restarts,rmax);
    printf("         iter breakdown: solves %.3f (%.4f each)  ortho %.3f  resid %.3f  ritz %.3f\n",
           inf.t_op, inf.t_op/(double)inf.nops, inf.t_ortho, inf.t_resid, inf.t_ritz);
    printf("         lambda[0] = %.14g\n", lam[0]);

    /* ---- the cost eigs pays: UMFPACK LU + the same number of solves ---- */
    double ctl[UMFPACK_CONTROL],info2[UMFPACK_INFO]; umfpack_dl_defaults(ctl);
    void *Sym=NULL,*Num=NULL;
    double f0=wall();
    umfpack_dl_symbolic(n,n,Ap,Ai,Ax,&Sym,ctl,info2);
    umfpack_dl_numeric(Ap,Ai,Ax,Sym,&Num,ctl,info2);
    double t_fact=wall()-f0;
    double *b=malloc(n*sizeof(double)),*x=malloc(n*sizeof(double));
    for(int64_t i=0;i<n;i++) b[i]=1.0;
    int nprobe=5; double s0=wall();
    for(int r=0;r<nprobe;r++) umfpack_dl_solve(UMFPACK_A,Ap,Ai,Ax,x,b,Num,ctl,info2);
    double t_solve=(wall()-s0)/nprobe;
    double t_eigs = t_fact + (double)inf.nops*t_solve;
    printf("\neigs' path (UMFPACK LU, charged %lld solves -- same count as speigs used)\n",
           (long long)inf.nops);
    printf("         factorization %8.3f s   per solve %.4f s   TOTAL %8.3f s\n",
           t_fact,t_solve,t_eigs);

    printf("\n==> END-TO-END SPEEDUP  %.2fx\n", t_eigs/t_ours);
    printf("    (factorization %.2fx, per-solve %.2fx)\n",
           t_fact/(inf.t_analyze+inf.t_factor),
           t_solve/(inf.t_iter/(double)inf.nops));
    umfpack_dl_free_numeric(&Num); umfpack_dl_free_symbolic(&Sym);
    return 0;
}

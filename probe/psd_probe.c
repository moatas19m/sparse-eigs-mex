/* PROBE 2 -- (a) honest fill/flop actuals, (b) the SEMI-DEFINITE trap.
 * Neumann Laplacian: row sums = 0 => PSD, singular, null space = span{1}.
 * This is exactly the "possibly semi-definite" case: shift-invert at sigma=0
 * asks us to invert a singular matrix. Does CHOLMOD detect it? Does A+dI rescue it?
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "suitesparse/cholmod.h"
#include "suitesparse/umfpack.h"
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

/* dirichlet=1 -> SPD (diag 6);  dirichlet=0 -> Neumann, singular PSD (diag = #neighbours) */
static void lap3d(int64_t N,int dirichlet,double shift,int64_t**Ap_o,int64_t**Ai_o,double**Ax_o,int64_t*n_o){
    int64_t n=N*N*N,k=0,*Ap=malloc((n+1)*sizeof(int64_t)),*Ai=malloc(7*n*sizeof(int64_t));
    double*Ax=malloc(7*n*sizeof(double));
    for(int64_t z=0;z<N;z++)for(int64_t y=0;y<N;y++)for(int64_t x=0;x<N;x++){
        int64_t c=x+N*(y+N*z);Ap[c]=k;int64_t nb[6];int m=0,deg=0;
        if(z>0)nb[m++]=c-N*N; if(y>0)nb[m++]=c-N; if(x>0)nb[m++]=c-1;
        for(int i=0;i<m;i++){Ai[k]=nb[i];Ax[k]=-1.0;k++;}
        deg=m; int m2=0; int64_t nb2[3];
        if(x<N-1)nb2[m2++]=c+1; if(y<N-1)nb2[m2++]=c+N; if(z<N-1)nb2[m2++]=c+N*N;
        deg+=m2;
        Ai[k]=c; Ax[k]=(dirichlet?6.0:(double)deg)+shift; k++;
        for(int i=0;i<m2;i++){Ai[k]=nb2[i];Ax[k]=-1.0;k++;}
    }
    Ap[n]=k;*Ap_o=Ap;*Ai_o=Ai;*Ax_o=Ax;*n_o=n;
}
static cholmod_sparse mk(int64_t n,int64_t*Ap,int64_t*Ai,double*Ax){
    cholmod_sparse A; A.nrow=n;A.ncol=n;A.nzmax=Ap[n];A.p=Ap;A.i=Ai;A.x=Ax;A.z=NULL;A.nz=NULL;
    A.stype=1;A.itype=CHOLMOD_LONG;A.xtype=CHOLMOD_REAL;A.dtype=CHOLMOD_DOUBLE;A.sorted=1;A.packed=1;return A;}

int main(int argc,char**argv){
    int64_t N=(argc>1)?atoi(argv[1]):40; int64_t *Ap,*Ai,n; double *Ax;

    /* ---- (a) honest UMFPACK actuals on the SPD problem ---- */
    lap3d(N,1,0.0,&Ap,&Ai,&Ax,&n);
    double ctl[UMFPACK_CONTROL],info[UMFPACK_INFO]; umfpack_dl_defaults(ctl);
    void*S=NULL,*Nu=NULL;
    umfpack_dl_symbolic(n,n,Ap,Ai,Ax,&S,ctl,info);
    umfpack_dl_numeric(Ap,Ai,Ax,S,&Nu,ctl,info);
    printf("SPD n=%lld  UMFPACK ACTUAL nnz(L)=%.4e nnz(U)=%.4e  L+U=%.4e  flops=%.4e\n",
        (long long)n, info[UMFPACK_LNZ], info[UMFPACK_UNZ],
        info[UMFPACK_LNZ]+info[UMFPACK_UNZ], info[UMFPACK_FLOPS]);
    printf("          strategy used = %s\n",
        info[UMFPACK_STRATEGY_USED]==UMFPACK_STRATEGY_SYMMETRIC?"SYMMETRIC":"UNSYMMETRIC");
    cholmod_common c; cholmod_l_start(&c); c.supernodal=CHOLMOD_SUPERNODAL;
    c.nmethods=1;c.method[0].ordering=CHOLMOD_METIS;c.postorder=1;
    cholmod_sparse A=mk(n,Ap,Ai,Ax);
    cholmod_factor*L=cholmod_l_analyze(&A,&c); cholmod_l_factorize(&A,L,&c);
    printf("          CHOLMOD ACTUAL nnz(L)=%.4e  flops=%.4e   -> TRUE fill ratio %.2f, flop ratio %.2f\n\n",
        c.lnz,c.fl,(info[UMFPACK_LNZ]+info[UMFPACK_UNZ])/c.lnz, info[UMFPACK_FLOPS]/c.fl);
    cholmod_l_free_factor(&L,&c);

    /* ---- (b) the semi-definite trap ---- */
    int64_t *Bp,*Bi,nb; double *Bx;
    lap3d(N,0,0.0,&Bp,&Bi,&Bx,&nb);           /* singular PSD */
    cholmod_sparse B=mk(nb,Bp,Bi,Bx);
    cholmod_factor*LB=cholmod_l_analyze(&B,&c);
    double t0=now(); cholmod_l_factorize(&B,LB,&c); double t1=now();
    printf("SINGULAR PSD (Neumann, null space = constants):\n");
    printf("   cholmod factorize -> status=%d (%s), L->minor=%lld of n=%lld   [%.3f s]\n",
        c.status, c.status==CHOLMOD_NOT_POSDEF?"NOT_POSDEF (detected)":
                  (c.status==CHOLMOD_OK?"OK -- NOT detected!":"other"),
        (long long)LB->minor,(long long)nb,t1-t0);

    /* UMFPACK on the same singular matrix -- what eigs would hit */
    void*S2=NULL,*N2=NULL; double i2[UMFPACK_INFO];
    umfpack_dl_symbolic(nb,nb,Bp,Bi,Bx,&S2,ctl,i2);
    int st=umfpack_dl_numeric(Bp,Bi,Bx,S2,&N2,ctl,i2);
    printf("   umfpack numeric   -> status=%d (%s), rcond=%.3e\n",
        st, st==UMFPACK_WARNING_singular_matrix?"WARNING_singular":(st==UMFPACK_OK?"OK":"other"),
        i2[UMFPACK_RCOND]);

    /* rescue: factorize A + delta*I for a few deltas */
    printf("   rescue by shifting A + delta*I:\n");
    double anorm = 12.0; /* ||A||_inf bound for this operator */
    for (int e=4; e<=12; e+=4){
        double delta = anorm*pow(10.0,-(double)e);
        int64_t *Cp,*Ci,nc; double *Cx; lap3d(N,0,delta,&Cp,&Ci,&Cx,&nc);
        cholmod_sparse C=mk(nc,Cp,Ci,Cx);
        cholmod_factor*LC=cholmod_l_analyze(&C,&c);
        cholmod_l_factorize(&C,LC,&c);
        printf("      delta=%8.2e -> status=%-3d %-22s minor=%lld\n", delta, c.status,
            c.status==CHOLMOD_OK?"OK (factorizable)":"NOT_POSDEF",(long long)LC->minor);
        cholmod_l_free_factor(&LC,&c); free(Cp);free(Ci);free(Cx);
    }
    return 0;
}

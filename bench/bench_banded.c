/* Does the banded path actually beat CHOLMOD? The threshold should come from
 * measurement, not from the assumption that "banded must be better". */
#include "speigs.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
static double wall(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
/* SPD banded: -1 within bandwidth kd, diagonally dominant diagonal */
static void banded(int64_t n,int64_t kd,int64_t**Ap_o,int64_t**Ai_o,double**Ax_o){
    int64_t nz=0;
    for(int64_t c=0;c<n;c++){ int64_t lo=c-kd<0?0:c-kd, hi=c+kd>n-1?n-1:c+kd; nz+=hi-lo+1; }
    int64_t *Ap=malloc((n+1)*sizeof(int64_t)),*Ai=malloc(nz*sizeof(int64_t));
    double *Ax=malloc(nz*sizeof(double)); int64_t q=0;
    for(int64_t c=0;c<n;c++){ Ap[c]=q;
        int64_t lo=c-kd<0?0:c-kd, hi=c+kd>n-1?n-1:c+kd;
        for(int64_t r=lo;r<=hi;r++){ Ai[q]=r; Ax[q]=(r==c)?(2.0*kd+1.0):-1.0; q++; } }
    Ap[n]=q; *Ap_o=Ap;*Ai_o=Ai;*Ax_o=Ax;
}
static double run(int64_t n,int64_t*Ap,int64_t*Ai,double*Ax,int64_t k,int bmax,int*path,int64_t*nops){
    double best=1e30;
    for(int t=0;t<3;t++){
        double lam[16],res[16]; speigs_opts o; speigs_default_opts(&o); o.band_max=bmax;
        speigs_info inf; double t0=wall();
        speigs(n,Ap,Ai,Ax,k,SPEIGS_SM,&o,lam,NULL,res,&inf);
        double el=wall()-t0; if(el<best){best=el;*path=(int)inf.path;*nops=inf.nops;}
    }
    return best;
}
int main(void){
    int64_t n=50000,k=6;
    printf("SPD banded matrices, n=%lld, k=6, median-of-min over 3 runs\n\n",(long long)n);
    printf("%6s %10s %10s %9s  %s\n","kd","banded(s)","CHOLMOD(s)","ratio","verdict");
    int64_t kds[]={1,2,4,8,16,32,64,128,256};
    for(unsigned i=0;i<sizeof kds/sizeof*kds;i++){
        int64_t kd=kds[i],*Ap,*Ai; double*Ax; banded(n,kd,&Ap,&Ai,&Ax);
        int p1,p2; int64_t o1,o2;
        double tb=run(n,Ap,Ai,Ax,k,(int)kd+1,&p1,&o1);   /* allow banded */
        double tc=run(n,Ap,Ai,Ax,k,-1,&p2,&o2);          /* force CHOLMOD */
        const char*v = (p1!=SPEIGS_PATH_BANDED)?"banded not taken":
                       (tb<tc?"banded WINS":"CHOLMOD wins");
        printf("%6lld %10.3f %10.3f %8.2fx  %s\n",(long long)kd,tb,tc,tc/tb,v);
        free(Ap);free(Ai);free(Ax);
    }
    return 0;
}

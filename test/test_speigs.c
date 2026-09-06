/* Standalone validation for the speigs core -- no MATLAB required.
 *
 * The point of using Laplacians here is that their spectra are known in closed
 * form, so correctness is checked against analysis rather than against another
 * numerical code that could be wrong in the same way.
 *
 *   3-D Dirichlet 7-point:  lambda_{pqr} = sum of 2 - 2cos(p*pi/(N+1))
 *   3-D Neumann  (graph):   lambda_{pqr} = sum of 2 - 2cos(p*pi/N),  p from 0
 *                           => singular, PSD, null space = constants
 */
#include "speigs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double wall(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                          return t.tv_sec+1e-9*t.tv_nsec; }
static int fails = 0, checks = 0;

static void check(int ok, const char *what, double got, double want, double tol){
    checks++;
    if (!ok){ fails++; printf("    FAIL  %-42s got %.12g want %.12g (tol %.1e)\n",
                              what, got, want, tol); }
    else      printf("    ok    %-42s %.12g\n", what, got);
}

/* ---- builders ------------------------------------------------------------ */
/* dirichlet: diag 6 ; neumann: diag = degree (row sums zero => singular) */
static void lap3d(int64_t N, int dirichlet,
                  int64_t **Ap_o, int64_t **Ai_o, double **Ax_o, int64_t *n_o){
    int64_t n=N*N*N, k=0;
    int64_t *Ap=malloc((n+1)*sizeof(int64_t)), *Ai=malloc(7*n*sizeof(int64_t));
    double  *Ax=malloc(7*n*sizeof(double));
    for (int64_t z=0;z<N;z++) for (int64_t y=0;y<N;y++) for (int64_t x=0;x<N;x++){
        int64_t c=x+N*(y+N*z); Ap[c]=k;
        int64_t lo[3]; int m=0;
        if(z>0)lo[m++]=c-N*N; if(y>0)lo[m++]=c-N; if(x>0)lo[m++]=c-1;
        int64_t hi[3]; int m2=0;
        if(x<N-1)hi[m2++]=c+1; if(y<N-1)hi[m2++]=c+N; if(z<N-1)hi[m2++]=c+N*N;
        for(int i=0;i<m;i++){ Ai[k]=lo[i]; Ax[k]=-1.0; k++; }
        Ai[k]=c; Ax[k]= dirichlet ? 6.0 : (double)(m+m2); k++;
        for(int i=0;i<m2;i++){ Ai[k]=hi[i]; Ax[k]=-1.0; k++; }
    }
    Ap[n]=k; *Ap_o=Ap; *Ai_o=Ai; *Ax_o=Ax; *n_o=n;
}

static int cmpd(const void*a,const void*b){
    double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y); }

/* exact spectrum of the 3-D Laplacian, sorted ascending */
static double *exact_spectrum(int64_t N,int dirichlet,int64_t *cnt){
    int64_t M=N; double *e1=malloc(M*sizeof(double));
    for (int64_t p=0;p<M;p++)
        e1[p] = dirichlet ? 2.0-2.0*cos((p+1)*M_PI/(N+1))
                          : 2.0-2.0*cos(p*M_PI/N);
    int64_t n=N*N*N; double *all=malloc(n*sizeof(double)); int64_t t=0;
    for(int64_t a=0;a<M;a++)for(int64_t b=0;b<M;b++)for(int64_t c=0;c<M;c++)
        all[t++]=e1[a]+e1[b]+e1[c];
    qsort(all,(size_t)n,sizeof(double),cmpd); free(e1); *cnt=n; return all;
}

/* ---- tests --------------------------------------------------------------- */
static void test_dirichlet(int64_t N, int64_t k){
    printf("\n[1] 3-D Dirichlet Laplacian, SPD -- smallest %lld, vs analytic spectrum\n",
           (long long)k);
    int64_t *Ap,*Ai,n; double *Ax; lap3d(N,1,&Ap,&Ai,&Ax,&n);
    int64_t cnt; double *ex=exact_spectrum(N,1,&cnt);
    double *lam=calloc(k,sizeof(double)), *res=calloc(k,sizeof(double));
    double *V=calloc((size_t)n*k,sizeof(double));
    speigs_opts o; speigs_default_opts(&o); speigs_info inf;
    double t=wall();
    int rc=speigs(n,Ap,Ai,Ax,k,SPEIGS_SM,&o,lam,V,res,&inf);
    t=wall()-t;
    printf("    n=%lld nnz=%lld  rc=%d path=%d nops=%lld restarts=%lld  %.3f s"
           "  (analyze %.3f, factor %.3f, iter %.3f)\n",
           (long long)n,(long long)Ap[n],rc,(int)inf.path,(long long)inf.nops,
           (long long)inf.restarts,t,inf.t_analyze,inf.t_factor,inf.t_iter);
    for (int64_t i=0;i<k;i++){
        char b[64]; snprintf(b,sizeof b,"lambda[%lld] vs analytic",(long long)i);
        check(fabs(lam[i]-ex[i]) <= 1e-9*fmax(1.0,fabs(ex[i])), b, lam[i], ex[i], 1e-9);
    }
    double rmax=0; for(int64_t i=0;i<k;i++) rmax=fmax(rmax,res[i]);
    check(rmax <= 1e-8, "max residual ||Av-lv||", rmax, 0.0, 1e-8);
    check(inf.flag==0, "converged flag", inf.flag, 0, 0);
    free(Ap);free(Ai);free(Ax);free(ex);free(lam);free(res);free(V);
}

static void test_neumann_singular(int64_t N, int64_t k){
    printf("\n[2] 3-D Neumann Laplacian -- SINGULAR PSD (null space = constants)\n");
    int64_t *Ap,*Ai,n; double *Ax; lap3d(N,0,&Ap,&Ai,&Ax,&n);
    int64_t cnt; double *ex=exact_spectrum(N,0,&cnt);
    double *lam=calloc(k,sizeof(double)), *res=calloc(k,sizeof(double));
    double *V=calloc((size_t)n*k,sizeof(double));
    speigs_opts o; speigs_default_opts(&o); speigs_info inf;
    int rc=speigs(n,Ap,Ai,Ax,k,SPEIGS_SM,&o,lam,V,res,&inf);
    printf("    n=%lld rc=%d path=%d shift=%.3e nops=%lld\n",
           (long long)n,rc,(int)inf.path,inf.shift,(long long)inf.nops);
    check(inf.shift>0.0, "adaptive shift engaged (A was singular)", inf.shift, 1, 0);
    check(inf.flag==0, "converged flag (not partial)", inf.flag, 0, 0);
    check(inf.nops<400, "operator applications stayed bounded", (double)inf.nops, 400, 0);
    check(fabs(lam[0]) <= 1e-9, "lambda[0] == 0 (null space found)", lam[0], 0.0, 1e-9);
    for (int64_t i=1;i<k;i++){
        char b[64]; snprintf(b,sizeof b,"lambda[%lld] vs analytic",(long long)i);
        check(fabs(lam[i]-ex[i]) <= 1e-7*fmax(1.0,fabs(ex[i])), b, lam[i], ex[i], 1e-7);
    }
    /* null vector must be constant */
    double mn=V[0],mx=V[0];
    for(int64_t i=0;i<n;i++){ mn=fmin(mn,V[i]); mx=fmax(mx,V[i]); }
    check(fabs(mx-mn) <= 1e-7, "null vector is constant (max-min)", mx-mn, 0.0, 1e-7);
    free(Ap);free(Ai);free(Ax);free(ex);free(lam);free(res);free(V);
}

static void test_multiplicity(int64_t N){
    printf("\n[3] Repeated eigenvalues -- Dirichlet Laplacian has a multiplicity-3 cluster\n");
    int64_t k=6, *Ap,*Ai,n; double *Ax; lap3d(N,1,&Ap,&Ai,&Ax,&n);
    int64_t cnt; double *ex=exact_spectrum(N,1,&cnt);
    double *lam=calloc(k,sizeof(double)), *res=calloc(k,sizeof(double));
    double *V=calloc((size_t)n*k,sizeof(double));
    speigs_opts o; speigs_default_opts(&o); speigs_info inf;
    speigs(n,Ap,Ai,Ax,k,SPEIGS_SM,&o,lam,V,res,&inf);
    /* lambda[1..3] are a genuine triple; plain Lanczos without deflation
     * would return only one copy. */
    printf("    analytic 1..3: %.12g %.12g %.12g\n", ex[1],ex[2],ex[3]);
    printf("    computed 1..3: %.12g %.12g %.12g\n", lam[1],lam[2],lam[3]);
    int trip = fabs(ex[1]-ex[2])<1e-12 && fabs(ex[2]-ex[3])<1e-12;
    check(trip, "test matrix really has a triple eigenvalue", ex[3]-ex[1], 0.0, 1e-12);
    for (int64_t i=1;i<=3;i++){
        char b[64]; snprintf(b,sizeof b,"copy %lld of the triple found",(long long)i);
        check(fabs(lam[i]-ex[i])<=1e-9*fmax(1.0,fabs(ex[i])), b, lam[i], ex[i], 1e-9);
    }
    /* the three eigenvectors must be mutually orthogonal */
    double worst=0;
    for(int a=1;a<=3;a++)for(int b=a+1;b<=3;b++){
        double d=0; for(int64_t q=0;q<n;q++) d+=V[(size_t)a*n+q]*V[(size_t)b*n+q];
        worst=fmax(worst,fabs(d));
    }
    check(worst<=1e-8,"eigenvectors of the triple are orthogonal",worst,0.0,1e-8);
    free(Ap);free(Ai);free(Ax);free(ex);free(lam);free(res);free(V);
}

static void test_lm(int64_t N, int64_t k){
    printf("\n[4] Largest magnitude ('lm') -- no factorization, plain Lanczos\n");
    int64_t *Ap,*Ai,n; double *Ax; lap3d(N,1,&Ap,&Ai,&Ax,&n);
    int64_t cnt; double *ex=exact_spectrum(N,1,&cnt);
    double *lam=calloc(k,sizeof(double)), *res=calloc(k,sizeof(double));
    speigs_opts o; speigs_default_opts(&o); speigs_info inf;
    speigs(n,Ap,Ai,Ax,k,SPEIGS_LM,&o,lam,NULL,res,&inf);
    printf("    path=%d (3 = direct Lanczos, no factorization) nops=%lld\n",
           (int)inf.path,(long long)inf.nops);
    check(inf.path==SPEIGS_PATH_DIRECT,"took the no-factorization path",inf.path,3,0);
    for (int64_t i=0;i<k;i++){
        char b[64]; snprintf(b,sizeof b,"largest[%lld] vs analytic",(long long)i);
        check(fabs(lam[i]-ex[cnt-1-i])<=1e-9*fabs(ex[cnt-1-i]),b,lam[i],ex[cnt-1-i],1e-9);
    }
    free(Ap);free(Ai);free(Ax);free(ex);free(lam);free(res);
}

static void test_diagonal(void){
    printf("\n[5] Diagonal matrix -- structure detection short-circuit\n");
    int64_t n=5000,k=4;
    int64_t *Ap=malloc((n+1)*sizeof(int64_t)),*Ai=malloc(n*sizeof(int64_t));
    double *Ax=malloc(n*sizeof(double));
    for(int64_t i=0;i<n;i++){ Ap[i]=i; Ai[i]=i; Ax[i]=(double)(i+1)*((i%2)?1.0:-1.0); }
    Ap[n]=n;
    double *lam=calloc(k,sizeof(double)); speigs_opts o; speigs_default_opts(&o);
    speigs_info inf; double t=wall();
    speigs(n,Ap,Ai,Ax,k,SPEIGS_SM,&o,lam,NULL,NULL,&inf); t=wall()-t;
    printf("    path=%d (0 = diagonal) in %.5f s\n",(int)inf.path,t);
    check(inf.path==SPEIGS_PATH_DIAG,"detected diagonal structure",inf.path,0,0);
    check(fabs(fabs(lam[0])-1.0)<1e-12,"smallest |lambda| == 1",fabs(lam[0]),1.0,1e-12);
    check(fabs(fabs(lam[3])-4.0)<1e-12,"4th smallest |lambda| == 4",fabs(lam[3]),4.0,1e-12);
    free(Ap);free(Ai);free(Ax);free(lam);
}

static void test_indefinite(int64_t N,int64_t k){
    printf("\n[6] Indefinite matrix (A - cI) -- LU fallback, eigenvalues nearest 0\n");
    int64_t *Ap,*Ai,n; double *Ax; lap3d(N,1,&Ap,&Ai,&Ax,&n);
    int64_t cnt; double *ex=exact_spectrum(N,1,&cnt);
    /* Shift to a point BETWEEN two eigenvalues. Shifting by an exact
     * eigenvalue makes A-cI exactly singular, which is the pathological case
     * for shift-invert; that is tested separately in test_pathological(). */
    int64_t mid=cnt/2; while (mid+1<cnt && ex[mid+1]-ex[mid] < 1e-6) mid++;
    double c = 0.5*(ex[mid]+ex[mid+1]);
    for(int64_t col=0;col<n;col++)
        for(int64_t p=Ap[col];p<Ap[col+1];p++) if(Ai[p]==col) Ax[p]-=c;
    for(int64_t i=0;i<cnt;i++) ex[i]-=c;
    /* analytic answer: the k values of ex closest to zero */
    double *byabs=malloc(cnt*sizeof(double));
    memcpy(byabs,ex,cnt*sizeof(double));
    for(int64_t i=0;i<cnt;i++) byabs[i]=fabs(byabs[i]);
    qsort(byabs,(size_t)cnt,sizeof(double),cmpd);
    double *lam=calloc(k,sizeof(double)),*res=calloc(k,sizeof(double));
    speigs_opts o; speigs_default_opts(&o); speigs_info inf;
    speigs(n,Ap,Ai,Ax,k,SPEIGS_SM,&o,lam,NULL,res,&inf);
    printf("    path=%d (5 = UMFPACK LU) nops=%lld\n",(int)inf.path,(long long)inf.nops);
    check(inf.path==SPEIGS_PATH_LU,"fell back to LU for indefinite A",inf.path,5,0);
    check(inf.flag==0, "converged flag (not partial)", inf.flag, 0, 0);
    for(int64_t i=0;i<k;i++){
        char b[64]; snprintf(b,sizeof b,"|lambda[%lld]| nearest zero",(long long)i);
        check(fabs(fabs(lam[i])-byabs[i])<=1e-7*fmax(1.0,byabs[i]),b,fabs(lam[i]),byabs[i],1e-7);
    }
    free(Ap);free(Ai);free(Ax);free(ex);free(byabs);free(lam);free(res);
}

/* Deliberately pathological: shift by an EXACT eigenvalue, so A-cI is exactly
 * singular. Shift-invert cannot certify a 1e-14 residual here. What matters is
 * that the values are still right and the flag honestly reports partial
 * convergence rather than claiming success. */
static void test_pathological(int64_t N,int64_t k){
    printf("\n[7] Pathological: sigma sits exactly ON an eigenvalue (A-cI singular)\n");
    int64_t *Ap,*Ai,n; double *Ax; lap3d(N,1,&Ap,&Ai,&Ax,&n);
    int64_t cnt; double *ex=exact_spectrum(N,1,&cnt);
    double c=ex[cnt/2];
    for(int64_t col=0;col<n;col++)
        for(int64_t p=Ap[col];p<Ap[col+1];p++) if(Ai[p]==col) Ax[p]-=c;
    double *lam=calloc(k,sizeof(double)),*res=calloc(k,sizeof(double));
    speigs_opts o; speigs_default_opts(&o); o.maxit=40; speigs_info inf;
    int rc=speigs(n,Ap,Ai,Ax,k,SPEIGS_SM,&o,lam,NULL,res,&inf);
    printf("    rc=%d path=%d flag=%d  |lambda[0]|=%.3e\n",
           rc,(int)inf.path,inf.flag,fabs(lam[0]));
    check(rc==SPEIGS_OK,"returned cleanly, did not crash",rc,0,0);
    check(fabs(lam[0])<1e-6,"still located the near-zero eigenvalue",fabs(lam[0]),0.0,1e-6);
    check(inf.flag==1,"honestly reported PARTIAL convergence",inf.flag,1,0);
    free(Ap);free(Ai);free(Ax);free(ex);free(lam);free(res);
}


/* ---- block diagonal: two disconnected Laplacians ------------------------- */
static void test_blocks(int64_t N1,int64_t N2,int64_t k){
    printf("\n[8] Block-diagonal (two disconnected Laplacians) -- component splitting\n");
    int64_t *P1,*I1,n1,*P2,*I2,n2; double *X1,*X2;
    lap3d(N1,1,&P1,&I1,&X1,&n1); lap3d(N2,1,&P2,&I2,&X2,&n2);
    int64_t n=n1+n2, nz=P1[n1]+P2[n2];
    int64_t *Ap=malloc((n+1)*sizeof(int64_t)),*Ai=malloc(nz*sizeof(int64_t));
    double *Ax=malloc(nz*sizeof(double)); int64_t q=0;
    for(int64_t c=0;c<n1;c++){ Ap[c]=q;
        for(int64_t p=P1[c];p<P1[c+1];p++){Ai[q]=I1[p];Ax[q]=X1[p];q++;} }
    for(int64_t c=0;c<n2;c++){ Ap[n1+c]=q;
        for(int64_t p=P2[c];p<P2[c+1];p++){Ai[q]=n1+I2[p];Ax[q]=X2[p];q++;} }
    Ap[n]=q;
    /* analytic: union of the two spectra */
    int64_t c1,c2; double *e1=exact_spectrum(N1,1,&c1),*e2=exact_spectrum(N2,1,&c2);
    double *all=malloc((c1+c2)*sizeof(double));
    memcpy(all,e1,c1*sizeof(double)); memcpy(all+c1,e2,c2*sizeof(double));
    qsort(all,(size_t)(c1+c2),sizeof(double),cmpd);

    double *lam=calloc(k,sizeof(double)),*res=calloc(k,sizeof(double));
    double *V=calloc((size_t)n*k,sizeof(double));
    speigs_opts o; speigs_default_opts(&o); speigs_info inf;
    speigs(n,Ap,Ai,Ax,k,SPEIGS_SM,&o,lam,V,res,&inf);
    printf("    n=%lld  path=%d (4 = blocks)  ncomp=%lld  nops=%lld\n",
        (long long)n,(int)inf.path,(long long)inf.ncomp,(long long)inf.nops);
    check(inf.path==SPEIGS_PATH_BLOCKS,"took the component-splitting path",inf.path,4,0);
    check(inf.ncomp==2,"found exactly 2 components",(double)inf.ncomp,2,0);
    for(int64_t i=0;i<k;i++){
        char b[64]; snprintf(b,sizeof b,"lambda[%lld] vs union spectrum",(long long)i);
        check(fabs(lam[i]-all[i])<=1e-9*fmax(1.0,fabs(all[i])),b,lam[i],all[i],1e-9);
    }
    /* each eigenvector must live entirely inside one block */
    int clean=1;
    for(int64_t t2=0;t2<k;t2++){
        double a=0,b2=0;
        for(int64_t i=0;i<n1;i++) a+=V[(size_t)t2*n+i]*V[(size_t)t2*n+i];
        for(int64_t i=n1;i<n;i++) b2+=V[(size_t)t2*n+i]*V[(size_t)t2*n+i];
        if (fmin(a,b2)>1e-16) clean=0;
    }
    check(clean,"each eigenvector confined to one block",clean,1,0);
    free(P1);free(I1);free(X1);free(P2);free(I2);free(X2);
    free(Ap);free(Ai);free(Ax);free(e1);free(e2);free(all);free(lam);free(res);free(V);
}

/* ---- banded: 1-D Laplacian, bandwidth 1 after RCM ----------------------- */
static void test_banded(int64_t n,int64_t k){
    printf("\n[9] Banded matrix (1-D Laplacian, bandwidth 1) -- LAPACK dpbtrf path\n");
    int64_t *Ap=malloc((n+1)*sizeof(int64_t)),*Ai=malloc(3*n*sizeof(int64_t));
    double *Ax=malloc(3*n*sizeof(double)); int64_t q=0;
    for(int64_t c=0;c<n;c++){ Ap[c]=q;
        if(c>0){Ai[q]=c-1;Ax[q]=-1.0;q++;}
        Ai[q]=c;Ax[q]=2.0;q++;
        if(c<n-1){Ai[q]=c+1;Ax[q]=-1.0;q++;} }
    Ap[n]=q;
    double *lam=calloc(k,sizeof(double)),*res=calloc(k,sizeof(double));
    speigs_opts o; speigs_default_opts(&o); speigs_info inf;
    double t=wall(); speigs(n,Ap,Ai,Ax,k,SPEIGS_SM,&o,lam,NULL,res,&inf); t=wall()-t;
    printf("    n=%lld  path=%d (6 = banded)  nops=%lld  %.4f s\n",
        (long long)n,(int)inf.path,(long long)inf.nops,t);
    check(inf.path==SPEIGS_PATH_BANDED,"took the banded path",inf.path,6,0);
    for(int64_t i=0;i<k;i++){
        double ex=2.0-2.0*cos((i+1)*M_PI/(n+1));
        char b[64]; snprintf(b,sizeof b,"lambda[%lld] vs analytic",(long long)i);
        check(fabs(lam[i]-ex)<=1e-9*fmax(1e-6,fabs(ex)),b,lam[i],ex,1e-9);
    }
    free(Ap);free(Ai);free(Ax);free(lam);free(res);
}

/* ---- isolated nodes / zero rows ----------------------------------------- */
static void test_isolated(void){
    printf("\n[10] Isolated nodes and a numerically zero row\n");
    /* a 1-D Laplacian chain of 2000, plus 3 isolated diagonal entries, plus one
     * row whose only entry is a stored zero (structurally present, no coupling) */
    int64_t m=2000, n=m+4;
    int64_t *Ap=malloc((n+1)*sizeof(int64_t)),*Ai=malloc(3*n*sizeof(int64_t));
    double *Ax=malloc(3*n*sizeof(double)); int64_t q=0;
    for(int64_t c=0;c<m;c++){ Ap[c]=q;
        if(c>0){Ai[q]=c-1;Ax[q]=-1.0;q++;}
        Ai[q]=c;Ax[q]=2.0;q++;
        if(c<m-1){Ai[q]=c+1;Ax[q]=-1.0;q++;} }
    double iso[3]={1e-7,-5e-8,3e-7};
    for(int j=0;j<3;j++){ Ap[m+j]=q; Ai[q]=m+j; Ax[q]=iso[j]; q++; }
    Ap[m+3]=q; Ai[q]=m+3; Ax[q]=0.0; q++;      /* stored zero => zero eigenvalue */
    Ap[n]=q;
    int64_t k=4; double *lam=calloc(k,sizeof(double));
    speigs_opts o; speigs_default_opts(&o); speigs_info inf;
    speigs(n,Ap,Ai,Ax,k,SPEIGS_SM,&o,lam,NULL,NULL,&inf);
    printf("    n=%lld path=%d ncomp=%lld  lambda = %.6g %.6g %.6g %.6g\n",
        (long long)n,(int)inf.path,(long long)inf.ncomp,lam[0],lam[1],lam[2],lam[3]);
    check(inf.ncomp==5,"chain + 3 isolated + 1 zero row = 5 components",(double)inf.ncomp,5,0);
    check(fabs(lam[0])<1e-14,"stored-zero row gives an exact zero eigenvalue",lam[0],0.0,1e-14);
    check(fabs(lam[1]+5e-8)<1e-18,"isolated -5e-8 found",lam[1],-5e-8,1e-18);
    check(fabs(lam[2]-1e-7)<1e-18,"isolated 1e-7 found",lam[2],1e-7,1e-18);
    check(fabs(lam[3]-3e-7)<1e-18,"isolated 3e-7 found",lam[3],3e-7,1e-18);
    free(Ap);free(Ai);free(Ax);free(lam);
}


/* ---- complex Hermitian ---------------------------------------------------
 * A = U M U* with U = diag(exp(i theta_j)) is Hermitian and has EXACTLY the
 * spectrum of M, so the closed-form Laplacian eigenvalues still apply. */
static void test_hermitian(int64_t N,int64_t k){
    printf("\n[11] Complex Hermitian (A = U M U*, U unitary diagonal)\n");
    int64_t *Ap,*Ai,n; double *Ax; lap3d(N,1,&Ap,&Ai,&Ax,&n);
    int64_t cnt; double *ex=exact_spectrum(N,1,&cnt);
    double *th=malloc(n*sizeof(double));
    for(int64_t i=0;i<n;i++) th[i]=2.399963*(double)((i*2654435761u)%1000)/1000.0;
    int64_t nz=Ap[n];
    double *Ar=malloc(nz*sizeof(double)),*Aim=malloc(nz*sizeof(double));
    for(int64_t c=0;c<n;c++)
        for(int64_t p=Ap[c];p<Ap[c+1];p++){
            double d=th[Ai[p]]-th[c];
            Ar[p]=Ax[p]*cos(d); Aim[p]=Ax[p]*sin(d);
        }
    double *lam=calloc(k,sizeof(double)),*res=calloc(k,sizeof(double));
    double *Vr=calloc((size_t)n*k,sizeof(double)),*Vi=calloc((size_t)n*k,sizeof(double));
    speigs_opts o; speigs_default_opts(&o); speigs_info inf;
    int rc=speigs_z(n,Ap,Ai,Ar,Aim,k,SPEIGS_SM,&o,lam,Vr,Vi,res,&inf);
    printf("    n=%lld (embedded 2n=%lld) rc=%d nconv=%lld nops=%lld\n",
        (long long)n,(long long)(2*n),rc,(long long)inf.nconv,(long long)inf.nops);
    check(rc==SPEIGS_OK,"returned cleanly",rc,0,0);
    for(int64_t i=0;i<k;i++){
        char b[64]; snprintf(b,sizeof b,"lambda[%lld] vs analytic",(long long)i);
        check(fabs(lam[i]-ex[i])<=1e-8*fmax(1.0,fabs(ex[i])),b,lam[i],ex[i],1e-8);
    }
    /* independent check: complex residual ||A v - lambda v|| computed here */
    double worst=0;
    for(int64_t c2=0;c2<k;c2++){
        double *wr=calloc(n,sizeof(double)),*wi=calloc(n,sizeof(double));
        for(int64_t c=0;c<n;c++)
            for(int64_t p=Ap[c];p<Ap[c+1];p++){
                int64_t r=Ai[p];
                double vr=Vr[(size_t)c2*n+c], vi=Vi[(size_t)c2*n+c];
                wr[r]+=Ar[p]*vr-Aim[p]*vi;
                wi[r]+=Ar[p]*vi+Aim[p]*vr;
            }
        double s2=0;
        for(int64_t i=0;i<n;i++){
            double dr=wr[i]-lam[c2]*Vr[(size_t)c2*n+i];
            double di=wi[i]-lam[c2]*Vi[(size_t)c2*n+i];
            s2+=dr*dr+di*di;
        }
        worst=fmax(worst,sqrt(s2)); free(wr); free(wi);
    }
    check(worst<=1e-8,"complex residual ||Av-lv|| (checked independently)",worst,0.0,1e-8);
    /* eigenvectors must not be i-multiples of one another */
    double maxov=0;
    for(int64_t a=0;a<k;a++)for(int64_t b=a+1;b<k;b++){
        double re=0,im=0;
        for(int64_t i=0;i<n;i++){
            double ur=Vr[(size_t)a*n+i],ui=Vi[(size_t)a*n+i];
            double vr=Vr[(size_t)b*n+i],vi=Vi[(size_t)b*n+i];
            re+=ur*vr+ui*vi; im+=ur*vi-ui*vr;
        }
        maxov=fmax(maxov,sqrt(re*re+im*im));
    }
    check(maxov<=1e-7,"distinct complex directions (no i-multiples kept)",maxov,0.0,1e-7);
    free(Ap);free(Ai);free(Ax);free(ex);free(th);free(Ar);free(Aim);
    free(lam);free(res);free(Vr);free(Vi);
}

int main(int argc,char**argv){
    int64_t N = (argc>1)?atoll(argv[1]):16;
    printf("=========================================================\n");
    printf(" speigs validation   (grid N=%lld  =>  n=%lld)\n",
           (long long)N,(long long)(N*N*N));
    printf("=========================================================\n");
    test_dirichlet(N,6);
    test_neumann_singular(N,5);
    test_multiplicity(N);
    test_lm(N,4);
    test_diagonal();
    test_indefinite(12,4);
    test_pathological(12,3);
    test_blocks(12,10,6);
    test_banded(20000,4);
    test_isolated();
    test_hermitian(12,5);
    printf("\n=========================================================\n");
    printf(" %d checks, %d failures\n", checks, fails);
    printf("=========================================================\n");
    return fails?1:0;
}

/* MEX gateway for speigs.  Thin by design: all numerics live in speigs_core.c,
 * which has no MATLAB dependency and is tested standalone.
 *
 *   d          = speigs(A,k)
 *   [V,D]      = speigs(A,k)
 *   [V,D,flag] = speigs(A,k,'sm'|'lm')
 *   [...]      = speigs(A,k,opts)     opts: tol,maxit,ncv,dense_max,seed,shift0
 *
 * Version adaptivity: the MATLAB release is detected at COMPILE time via
 * MX_HAS_INTERLEAVED_COMPLEX (defined from R2018a) and MX_API_VER, so the same
 * source builds on R2017a and on current releases with no manual configuration.
 *
 * GPL-3.0.
 */
#include "mex.h"
#include "matrix.h"
#include "speigs.h"
#include <string.h>
#include <stdlib.h>

/* R2018a introduced the interleaved-complex API. mxGetPr remains correct for
 * REAL arrays in both worlds, but mxGetDoubles is the supported spelling where
 * it exists, so prefer it and fall back automatically. */
#if defined(MX_HAS_INTERLEAVED_COMPLEX) && MX_HAS_INTERLEAVED_COMPLEX
  #define SPEIGS_GET_PR(a) mxGetDoubles(a)
  #define SPEIGS_API_NOTE  "interleaved-complex API (R2018a+)"
#else
  #define SPEIGS_GET_PR(a) mxGetPr(a)
  #define SPEIGS_API_NOTE  "separate-complex API (R2017b and earlier)"
#endif

/* Ctrl-C support. utIsInterruptPending is undocumented but universally used;
 * it needs -lut, so it is opt-in via -DSPEIGS_USE_UT to keep the build simple. */
#ifdef SPEIGS_USE_UT
extern bool utIsInterruptPending(void);
#endif

static double get_scalar_field(const mxArray *s, const char *f, double dflt){
    if (!s || !mxIsStruct(s)) return dflt;
    const mxArray *v = mxGetField(s, 0, f);
    if (!v || mxIsEmpty(v) || !mxIsNumeric(v)) return dflt;
    return mxGetScalar(v);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2)
        mexErrMsgIdAndTxt("speigs:nargin",
            "Usage: [V,D,flag] = speigs(A,k[,'sm'|'lm'|opts])");

    const mxArray *Am = prhs[0];
    if (!mxIsSparse(Am) || !mxIsDouble(Am))
        mexErrMsgIdAndTxt("speigs:type","A must be a sparse double matrix.");
    if (mxIsComplex(Am))
        mexErrMsgIdAndTxt("speigs:complex",
            "A must be real symmetric. Complex Hermitian input is not yet supported.");

    mwSize n = mxGetM(Am);
    if (n != mxGetN(Am))
        mexErrMsgIdAndTxt("speigs:square","A must be square.");
    if (n == 0)
        mexErrMsgIdAndTxt("speigs:empty","A must be non-empty.");

    double kd = mxGetScalar(prhs[1]);
    if (kd != (double)(speigs_int)kd || kd < 1)
        mexErrMsgIdAndTxt("speigs:k","k must be a positive integer.");
    speigs_int k = (speigs_int)kd;
    if (k > (speigs_int)n)
        mexErrMsgIdAndTxt("speigs:k","k must not exceed size(A,1).");

    /* ---- options ---- */
    speigs_which which = SPEIGS_SM;
    speigs_opts o; speigs_default_opts(&o);
    if (nrhs >= 3 && !mxIsEmpty(prhs[2])){
        if (mxIsChar(prhs[2])){
            char buf[16]; mxGetString(prhs[2], buf, sizeof buf);
            for (char *p=buf; *p; ++p) if (*p>='A'&&*p<='Z') *p += 32;
            if      (!strcmp(buf,"sm") || !strcmp(buf,"smallestabs")) which = SPEIGS_SM;
            else if (!strcmp(buf,"lm") || !strcmp(buf,"largestabs"))  which = SPEIGS_LM;
            else mexErrMsgIdAndTxt("speigs:which","sigma must be 'sm' or 'lm'.");
        } else if (mxIsStruct(prhs[2])){
            o.tol       = get_scalar_field(prhs[2],"tol",0.0);
            o.maxit     = (speigs_int)get_scalar_field(prhs[2],"maxit",0);
            o.ncv       = (speigs_int)get_scalar_field(prhs[2],"ncv",0);
            o.dense_max = (int)get_scalar_field(prhs[2],"dense_max",-1);
            o.seed      = (uint64_t)get_scalar_field(prhs[2],"seed",0);
            o.shift0    = get_scalar_field(prhs[2],"shift0",0.0);
            const mxArray *wf = mxGetField(prhs[2],0,"which");
            if (wf && mxIsChar(wf)){
                char buf[16]; mxGetString(wf,buf,sizeof buf);
                if (buf[0]=='l'||buf[0]=='L') which = SPEIGS_LM;
            }
        } else mexErrMsgIdAndTxt("speigs:opts","Third argument must be a string or a struct.");
    }

    /* ---- bind the CSC arrays ----
     * MATLAB's sparse format IS CHOLMOD's, so when mwIndex and speigs_int have
     * the same width we bind with zero copies. They differ only in 32-bit
     * builds (mex without -largeArrayDims), where we must convert. */
    const mwIndex *jc = mxGetJc(Am), *ir = mxGetIr(Am);
    const double  *pr = SPEIGS_GET_PR(Am);
    mwSize nnz = jc[n];

    speigs_int *Ap, *Ai;
    int owns = 0;
    if (sizeof(mwIndex) == sizeof(speigs_int)){
        Ap = (speigs_int*)jc;  Ai = (speigs_int*)ir;      /* zero copy */
    } else {
        owns = 1;
        Ap = (speigs_int*)mxMalloc((n+1)*sizeof(speigs_int));
        Ai = (speigs_int*)mxMalloc((nnz?nnz:1)*sizeof(speigs_int));
        for (mwSize i=0;i<=n;i++)  Ap[i]=(speigs_int)jc[i];
        for (mwSize i=0;i<nnz;i++) Ai[i]=(speigs_int)ir[i];
    }

    int want_vec = (nlhs >= 2);
    double *lam = (double*)mxCalloc(k, sizeof(double));
    double *res = (double*)mxCalloc(k, sizeof(double));
    mxArray *Vm = want_vec ? mxCreateDoubleMatrix(n, k, mxREAL) : NULL;
    double  *V  = want_vec ? SPEIGS_GET_PR(Vm) : NULL;

    speigs_info info;
    int rc = speigs((speigs_int)n, Ap, Ai, pr, k, which, &o, lam, V, res, &info);

    if (owns){ mxFree(Ap); mxFree(Ai); }

    if (rc != SPEIGS_OK){
        if (Vm) mxDestroyArray(Vm);
        mxFree(lam); mxFree(res);
        mexErrMsgIdAndTxt("speigs:failed","speigs failed: %s", speigs_errmsg(rc));
    }

    if (nlhs <= 1){
        plhs[0] = mxCreateDoubleMatrix(k, 1, mxREAL);
        memcpy(SPEIGS_GET_PR(plhs[0]), lam, (size_t)k*sizeof(double));
        if (Vm) mxDestroyArray(Vm);
    } else {
        plhs[0] = Vm;
        plhs[1] = mxCreateDoubleMatrix(k, k, mxREAL);   /* diagonal, as eigs */
        double *D = SPEIGS_GET_PR(plhs[1]);
        for (speigs_int i=0;i<k;i++) D[i + (size_t)i*k] = lam[i];
        if (nlhs >= 3){
            plhs[2] = mxCreateDoubleScalar((double)info.flag);
        }
    }
    if (nlhs >= 4){   /* undocumented 4th output: diagnostics, for benchmarking */
        const char *fn[] = {"path","nops","restarts","shift","anorm","nconv",
                            "t_analyze","t_factor","t_iter","api"};
        plhs[3] = mxCreateStructMatrix(1,1,10,fn);
        mxSetField(plhs[3],0,"path",     mxCreateDoubleScalar((double)info.path));
        mxSetField(plhs[3],0,"nops",     mxCreateDoubleScalar((double)info.nops));
        mxSetField(plhs[3],0,"restarts", mxCreateDoubleScalar((double)info.restarts));
        mxSetField(plhs[3],0,"shift",    mxCreateDoubleScalar(info.shift));
        mxSetField(plhs[3],0,"anorm",    mxCreateDoubleScalar(info.anorm));
        mxSetField(plhs[3],0,"nconv",    mxCreateDoubleScalar((double)info.nconv));
        mxSetField(plhs[3],0,"t_analyze",mxCreateDoubleScalar(info.t_analyze));
        mxSetField(plhs[3],0,"t_factor", mxCreateDoubleScalar(info.t_factor));
        mxSetField(plhs[3],0,"t_iter",   mxCreateDoubleScalar(info.t_iter));
        mxSetField(plhs[3],0,"api",      mxCreateString(SPEIGS_API_NOTE));
    }
    mxFree(lam); mxFree(res);
}

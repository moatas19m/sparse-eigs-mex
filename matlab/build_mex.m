function build_mex(varargin)
%BUILD_MEX  Compile the speigs MEX function.
%
%   build_mex()            compile with defaults
%   build_mex('debug')     -g, no optimisation, bounds checks
%
%  -largeArrayDims is MANDATORY: pre-R2018a MATLAB defaults mwIndex to 32-bit,
%  which would silently disagree with the 64-bit cholmod_l_* API and corrupt
%  results on matrices with more than 2^31 nonzeros.

here   = fileparts(mfilename('fullpath'));
root   = fileparts(here);
arch   = computer('arch');
ssdir  = fullfile(root,'third_party',['suitesparse-' local_arch()]);

if ~exist(fullfile(ssdir,'lib'),'dir')
    error('speigs:noSuiteSparse', ...
      ['Static SuiteSparse not found at %s\n' ...
       'Run:  ./build_suitesparse.sh   from the repo root first.'], ssdir);
end

dbg  = any(strcmpi(varargin,'debug'));
srcs = fullfile(root,'src',{'speigs_mex.c'});          % solver sources (not yet written)
srcs = srcs(cellfun(@(f) exist(f,'file')==2, srcs));
if isempty(srcs)
    error('speigs:noSources','No sources in %s yet — nothing to build.', fullfile(root,'src'));
end

flags = {'-largeArrayDims', ['-I' fullfile(root,'include')], ...
         ['-I' fullfile(ssdir,'include','suitesparse')], ...
         ['-L' fullfile(ssdir,'lib')], ...
         '-lcholmod','-lumfpack','-lamd','-lcamd','-lcolamd','-lccolamd', ...
         '-lsuitesparseconfig','-lmwlapack','-lmwblas'};
if dbg, flags = [flags {'-g','-DDEBUG'}];
else,   flags = [flags {'-O','-DNDEBUG'}]; end

fprintf('Building speigs for %s ...\n', arch);
mex(flags{:}, srcs{:}, '-output', fullfile(root,'matlab','speigs'));
fprintf('Done: %s\n', fullfile(root,'matlab',['speigs.' mexext]));
end

function a = local_arch()
c = computer('arch');
if strncmp(c,'maci',4) && ~isempty(strfind(c,'a64')), a='arm64'; %#ok<STREMP>
elseif strncmp(c,'maci',4), a='x86_64';
elseif strncmp(c,'glnx',4), a='x86_64';
else, a='x86_64'; end
end

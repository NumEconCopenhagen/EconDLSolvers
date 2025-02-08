//////////////////////////
// 1. external includes //
//////////////////////////

// standard C++ libraries

#include <windows.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>

///////////////
// 2. macros //
///////////////

#define MAX(X,Y) ((X)>(Y)?(X):(Y))
#define MIN(X,Y) ((X)<(Y)?(X):(Y))
#define BOUND(X,A,B) MIN(MAX(X,A),B)

#define EPS 1e-8
#define EPS_GRAD 1e-6
#define EXPORT extern "C" __declspec(dllexport)
#define MAX_D 3

/////////////////////////
// 3. generic includes //
/////////////////////////

// a. generic
#include "nlopt-2.4.2-dll64\nlopt.h"
#include "HighResTimer_class.hpp"
#include "logs.cpp"
#include "index.cpp"

#ifndef LINEAR_INTERP
#include "linear_interp.cpp"
#endif

// b. structs
#include "par_struct.cpp"
#include "vfi_struct.cpp"
#include "sim_struct.cpp"

typedef struct {
    
    par_struct *par;
    vfi_struct *vfi;
    long long t, i_p, func_evals, Nchoices;
    double m,n,p, c, d, exp_share, c_share, sav_share;

} solver_struct;

///////////////////////
// 4. local includes //
///////////////////////

// c. local modules
#ifndef UTILITY
#include "utility.cpp"
#endif

#ifndef COMPUTE_W
#include "compute_w.cpp"
#endif

#ifndef SOLVE_KEEP
#include "solve_keep.cpp"
#endif

#ifndef SOLVE_ADJ
#include "solve_adj.cpp"
#endif
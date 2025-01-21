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
#include "egm_struct.cpp"
#include "sim_struct.cpp"

typedef struct {
    
    par_struct *par;
    egm_struct *egm;
    long long t, i_p, func_evals, Nchoices;
    long long* Nns;
    double m, c, m_pd_fac;
    double *n, *d;

} solver_struct;

//////////////////
// 4. auxiliary //
//////////////////

void fill_NNs(long long* Nns, par_struct* par, egm_struct* egm){

    for (long long j = 0; j < MAX_D; j++){
        if (j < par->D){
            Nns[j] = egm->Nn;
        } else {
            Nns[j] = 1;
        }
    }

}

double interp(
    par_struct *par, egm_struct *egm, 
    double p, double* n, double m, 
    double *values)
{

    double res;

    if(par->D == 1){
        res = linear_interp::interp_3d(
            egm->p_grid,egm->n_grid,egm->m_grid, // grids
            egm->Np,egm->Nn,egm->Nm, // dimensions
            values, // values
            p,n[0],m); // points
    } else if(par->D == 2){
        res = linear_interp::interp_4d(
            egm->p_grid,egm->n_grid,egm->n_grid,egm->m_grid, // grids
            egm->Np,egm->Nn,egm->Nn,egm->Nm, // dimensions
            values, // values
            p,n[0],n[1],m); // points
    } else if(par->D == 3){
        res = linear_interp::interp_5d(
            egm->p_grid,egm->n_grid,egm->n_grid,egm->n_grid,egm->m_grid, // grids
            egm->Np,egm->Nn,egm->Nn,egm->Nn,egm->Nm, // dimensions
            values, // values
            p,n[0],n[1],n[2],m); // points
    }

    return res;

}

double interp_c_keep(
    par_struct *par, egm_struct *egm, 
    double* n, double m, 
    double *values)
{

    double res;

    if(par->D == 1){
        res = linear_interp::interp_2d(
            egm->n_grid,egm->m_keep_grid, // grids
            egm->Nn,egm->Nm_keep, // dimensions
            values, // values
            n[0],m); // points
    } else if(par->D == 2){
        res = linear_interp::interp_3d(
            egm->n_grid,egm->n_grid,egm->m_keep_grid, // grids
            egm->Nn,egm->Nn,egm->Nm_keep, // dimensions
            values, // values
            n[0],n[1],m); // points
    } else if(par->D == 3){
        res = linear_interp::interp_4d(
            egm->n_grid,egm->n_grid,egm->n_grid,egm->m_keep_grid, // grids
            egm->Nn,egm->Nn,egm->Nn,egm->Nm_keep, // dimensions
            values, // values
            n[0],n[1],n[2],m); // points
    }

    return res;

}

double interp_pd(
    par_struct *par, egm_struct *egm, 
    double* d, double m_pd, 
    double *values)
{

    double res;

    if(par->D == 1){
        res = linear_interp::interp_2d(
            egm->n_grid,egm->m_pd_grid, // grids
            egm->Nn,egm->Nm_pd, // dimensions
            values, // values
            d[0],m_pd); // points
    } else if(par->D == 2){
        res = linear_interp::interp_3d(
            egm->n_grid,egm->n_grid,egm->m_pd_grid, // grids
            egm->Nn,egm->Nn,egm->Nm_pd, // dimensions
            values, // values
            d[0],d[1],m_pd); // points
    } else if(par->D == 3){
        res = linear_interp::interp_4d(
            egm->n_grid,egm->n_grid,egm->n_grid,egm->m_pd_grid,// grids
            egm->Nn,egm->Nn,egm->Nn,egm->Nm_pd,// dimensions
            values, // values
            d[0],d[1],d[2],m_pd); // points
    }

    return res;

}

///////////////////////
// 5. local includes //
///////////////////////

// c. local modules
#ifndef UTILITY
#include "utility.cpp"
#endif

#ifndef BUDGET
#include "budget.cpp"
#endif

#ifndef COMPUTE_WQ
#include "compute_wq.cpp"
#endif

#ifndef SOLVE_C
#include "solve_c.cpp"
#endif

#ifndef SOLVE_D
#include "solve_d.cpp"
#endif
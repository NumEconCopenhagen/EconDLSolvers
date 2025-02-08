//////////////////////////
// 1. external includes //
//////////////////////////

// standard C++ libraries
#include <windows.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <typeinfo>

///////////////
// 2. macros //
///////////////

#define MAX(X,Y) ((X)>(Y)?(X):(Y))
#define MIN(X,Y) ((X)<(Y)?(X):(Y))
#define BOUND(X,A,B) MIN(MAX(X,A),B)

#define EXPORT extern "C" __declspec(dllexport)

/////////////////
// 3. includes //
/////////////////

#include "index.cpp"
#include "linear_interp.cpp"
#include "logs.cpp"

////////////////
// 4. structs //
////////////////

#include "par_struct.cpp"
#include "egm_struct.cpp"
#include "sim_struct.cpp"

/////////////
// 5. main //
/////////////

double marg_util_con(par_struct* par, double c){

    double mu = pow(c,-1.0);

    return mu;

} // marg_util_con

double inverse_marg_util_con(par_struct* par, double mu){

    double c = pow(mu,-1.0);

    return c;

} // inverse_marg_util_con

double compute_q(par_struct* par, egm_struct* egm, long long t, double sigma_xi, double sigma_psi, double m_pd, double p, double rho_p){

    double q = 0.0; // initialize

    for(long long i_psi = 0; i_psi < par->Npsi; i_psi++){
        for(long long i_xi = 0; i_xi < par->Nxi; i_xi++){

            // i. node adjustment
            double xi_base = par->xi[i_xi];
            double xi_ = xi_base*sigma_xi;
            double xi = exp(xi_ - 0.5*pow(sigma_xi,2.0));

            double psi_base = par->psi[i_psi];
            double psi_ = psi_base*sigma_psi;
            double psi = exp(psi_ - 0.5*pow(sigma_psi,2.0));

            // ii. next-period states
            double p_plus = pow(p,rho_p)*xi;
            double m_plus;
            if(t < par->T_retired){
                m_plus = par->R*m_pd + psi*p_plus*par->kappa[t];
            }
            else {
                m_plus = par->R*m_pd  + par->kappa[t];
            }

            // iii. next-period consumption and marginal utility
            long long i_sol_interp_pre = egm->Np*egm->Nsigma_xi*egm->Nsigma_psi*egm->Nrho_p*egm->Nm;
            long long i_sol_interp = (t+1)*i_sol_interp_pre;

            double c_plus;
            if (par->Nstates_fixed==0){

                c_plus = linear_interp::interp_2d(
                egm->p_grid, egm->m_grid, // grids
                egm->Np, egm->Nm, // dimensions
                &egm->sol_con[i_sol_interp], // sol_con
                p_plus, m_plus); // points

            }
            else if (par->Nstates_fixed==1){

                c_plus = linear_interp::interp_3d(
                egm->p_grid, egm->sigma_xi_grid, egm->m_grid, // grids
                egm->Np, egm->Nsigma_xi, egm->Nm, // dimensions
                &egm->sol_con[i_sol_interp], // sol_con
                p_plus, sigma_xi, m_plus); // points

            }
            else if (par->Nstates_fixed==2) {

                c_plus = linear_interp::interp_4d(
                egm->p_grid, egm->sigma_xi_grid, egm->sigma_psi_grid, egm->m_grid, // grids
                egm->Np, egm->Nsigma_xi, egm->Nsigma_psi, egm->Nm, // dimensions
                &egm->sol_con[i_sol_interp], // sol_con
                p_plus, sigma_xi, sigma_psi, m_plus); // points

            }
            else {

                c_plus = linear_interp::interp_5d(
                egm->p_grid, egm->sigma_xi_grid, egm->sigma_psi_grid, egm->rho_p_grid, egm->m_grid, // grids
                egm->Np, egm->Nsigma_xi, egm->Nsigma_psi, egm->Nrho_p, egm->Nm, // dimensions
                &egm->sol_con[i_sol_interp], // sol_con
                p_plus, sigma_xi, sigma_psi, rho_p, m_plus); // points
            }

            double mu_plus = marg_util_con(par,c_plus);

            // iv. sum up
            q += par->psi_w[i_psi]*par->xi_w[i_xi]*mu_plus;

        } // xi
    } // psi

    return q;

} // compute q

void interp_to_common_grid( par_struct* par, egm_struct* egm, double* q_grid, 
                            double* m_temp, double* c_temp, long long q_index, long long sol_con_index){

    // a. unpack
    double* m_grid = egm->m_grid;
    double* sol_con = egm->sol_con;

    // b. endogenous grid
    for (long long i_m_pd = 0; i_m_pd < egm->Nm_pd; i_m_pd++){

        // i. m_pd
        double m_pd = egm->m_pd_grid[i_m_pd];
        long long i_q = q_index + i_m_pd;

        // ii. c
        c_temp[i_m_pd+1] = inverse_marg_util_con(par,par->beta*par->R*q_grid[i_q]);

        // iii. m
        m_temp[i_m_pd+1] = m_pd + c_temp[i_m_pd+1];

    } // a

    // c. conversion to common grid
    linear_interp::interp_1d_vec_mon_noprep(
        m_temp,egm->Nm_pd+1,c_temp,
        m_grid,&sol_con[sol_con_index],egm->Nm
    );
    
    } 

EXPORT void solve_all(par_struct* par, egm_struct* egm){

    logs::write("log.txt",0,"par->cpp = %d threads\n",par->cppthreads);
    #pragma omp parallel num_threads(par->cppthreads)
    {

    double* m_temp = new double[egm->Nm_pd+1];
    double* c_temp = new double[egm->Nm_pd+1];
    
    m_temp[0] = 0.0;
    c_temp[0] = 0.0;

    for(long long t = par->T-1; t >= 0; t--){

        #pragma omp master
        logs::write("log.txt",1,"t = %d\n",t);

        // a. last period
        if(t == par->T-1){
            
            #pragma omp for collapse(5)
            for(long long i_p = 0; i_p < egm->Np; i_p++){
            for(long long i_sigma_xi = 0; i_sigma_xi < egm->Nsigma_xi; i_sigma_xi++){
            for(long long i_sigma_psi = 0; i_sigma_psi < egm->Nsigma_psi; i_sigma_psi++){
            for(long long i_rho_p = 0; i_rho_p < egm->Nrho_p; i_rho_p++){
            for(long long i_m = 0; i_m < egm->Nm; i_m++){

                long long i_sol = index::d6(
                    t,i_p,i_sigma_xi,i_sigma_psi,i_rho_p,i_m,
                    par->T,egm->Np,egm->Nsigma_xi,egm->Nsigma_psi,egm->Nrho_p,egm->Nm
                );
                
                egm->sol_con[i_sol] = egm->m_grid[i_m]/(1.0+par->bequest);

            } // m-loop
            } // rho_p loop
            } // sigma_psi loop
            } // sigma_xi loop
            } // p_loop

        } // final period

        // b. other periods
        else {

            // i. Initialize q_grid
            long long q_size = egm->Np*egm->Nsigma_xi*egm->Nsigma_psi*egm->Nrho_p*egm->Nm_pd;
            double* q_grid = new double[q_size];

            // ii. compute q
            #pragma omp for collapse(5)
            for (long long i_p = 0; i_p < egm->Np; i_p++){
            for (long long i_sigma_xi = 0; i_sigma_xi < egm->Nsigma_xi; i_sigma_xi++){
            for (long long i_sigma_psi = 0; i_sigma_psi < egm->Nsigma_psi; i_sigma_psi++){
            for (long long i_rho_p = 0; i_rho_p < egm->Nrho_p; i_rho_p++){
            for (long long i_m_pd = 0; i_m_pd < egm->Nm_pd; i_m_pd++){
            
                double p = egm->p_grid[i_p];
                double sigma_xi = egm->sigma_xi_grid[i_sigma_xi];
                double sigma_psi = egm->sigma_psi_grid[i_sigma_psi];
                double rho_p = egm->rho_p_grid[i_rho_p];
                double a = egm->m_pd_grid[i_m_pd];
                                            
                if (par->Nstates_fixed == 0){
                    sigma_xi = par->sigma_xi_base;
                    sigma_psi = par->sigma_psi_base;
                    rho_p = par->rho_p_base;
                }
                else if (par->Nstates_fixed == 1){
                    sigma_psi = par->sigma_psi_base;
                    rho_p = par->rho_p_base;
                }
                else if (par->Nstates_fixed == 2){
                    rho_p = par->rho_p_base;
                }

                double q = compute_q(par,egm,t,sigma_xi,sigma_psi,a,p,rho_p);

                long long i_q = index::d5(i_p,i_sigma_xi,i_sigma_psi,i_rho_p,i_m_pd,
                    egm->Np,egm->Nsigma_xi,egm->Nsigma_psi,egm->Nrho_p,egm->Nm_pd
                );
                
                q_grid[i_q] = q;
                                        
            } // rho_p loop
            } // sigma_psi loop
            } // sigma_xi_loop
            } // a-loop
            } // p_loop
            
            // iii. endogenous grid and conversion to common grid
            #pragma omp for collapse(4)
            for (long long i_p = 0; i_p < egm->Np; i_p++){
            for (long long i_sigma_xi = 0; i_sigma_xi < egm->Nsigma_xi; i_sigma_xi++){
            for (long long i_sigma_psi = 0; i_sigma_psi < egm->Nsigma_psi; i_sigma_psi++){
            for (long long i_rho_p = 0; i_rho_p < egm->Nrho_p; i_rho_p++){
            
                    long long q_index = index::d5(
                        i_p,i_sigma_xi,i_sigma_psi,i_rho_p,0,
                        egm->Np,egm->Nsigma_xi,egm->Nsigma_psi,egm->Nrho_p,egm->Nm_pd
                    );
                    
                    long long sol_con_index = index::d6(
                        t,i_p,i_sigma_xi,i_sigma_psi,i_rho_p,0,
                        par->T,egm->Np,egm->Nsigma_xi,egm->Nsigma_psi,egm->Nrho_p,egm->Nm
                    );
                    
                    interp_to_common_grid(par,egm,q_grid,m_temp,c_temp,q_index,sol_con_index);
            
            } // rho_p loop
            } // sigma_psi loop
            } // sigma_xi
            } // p
            
            delete[] q_grid;

        } // not last period
        
    } // t

    delete[] m_temp;
    delete[] c_temp;

    } // parallel

} // solve
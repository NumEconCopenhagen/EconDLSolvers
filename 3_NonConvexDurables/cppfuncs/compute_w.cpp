#ifndef MAIN
#define COMPUTE_W
#include "header.cpp"
#endif

double logsumexp(double v_keep, double v_adj, par_struct* par){ // utility function

    double vmax = fmax(v_keep,v_adj);
    double logsumexp = vmax + par->sigma_eps*log(exp((v_keep-vmax)/par->sigma_eps) + exp((v_adj-vmax)/par->sigma_eps));

    return logsumexp;

}

void compute_w(par_struct *par, vfi_struct *vfi, long long t){

    // compute post-decision value (w)

    #pragma omp parallel num_threads(par->cppthreads)
    {

    // loop over states
    #pragma omp for
    for(long long i_p = 0; i_p < vfi->Np; i_p++){
    for(long long i_n = 0; i_n < vfi->Nn; i_n++){
    for(long long i_a = 0; i_a < vfi->Na; i_a++){

        double d[MAX_D], n_plus[MAX_D];

        // a. unpack
        double p = vfi->p_grid[i_p];
        double n = vfi->n_grid[i_n];
        double a = vfi->a_grid[i_a];

        // b. initialize expectations
        double w_val = 0.0;

        // c. loop over quadrature points
        for(long long i_xi = 0; i_xi < par->Nxi; i_xi++){
        for(long long i_psi = 0; i_psi < par->Npsi; i_psi++){

            // i. unpack
            double xi = par->xi[i_xi];
            double psi = par->psi[i_psi];

            // ii. compute future states

            // next period permanent income
            double p_plus = pow(p,par->eta)*xi;
            if (p_plus > vfi->p_max){p_plus = vfi->p_max;} // enforce upper bound
            if (p_plus < vfi->p_min){p_plus = vfi->p_min;} // enforce lower bound

            // next period durable
            double n_plus = (1-par->delta)*n;
            if (n_plus > vfi->n_max){n_plus = vfi->n_max;} // enforce upper bound
            if (n_plus < vfi->n_min){n_plus = vfi->n_min;} // enforce lower bound

            // next period cash on hand
            double m_plus = par->R*a + psi*p_plus;
            if (m_plus > vfi->m_max){m_plus = vfi->m_max;} // enforce upper bound
            if (m_plus < vfi->m_min){m_plus = vfi->m_min;} // enforce lower bound

            // iii. interpolate future value
            long long i_plus_interp = index::d4(t+1,0,0,0,par->T,vfi->Np,vfi->Nn,vfi->Nm);

            // keeper value
            double v_plus_keep = linear_interp::interp_3d(
                vfi->p_grid,vfi->n_grid,vfi->m_grid, // grids
                vfi->Np,vfi->Nn,vfi->Nm, // dimensions
                &vfi->sol_v_keep[i_plus_interp], // values
                p_plus,n_plus,m_plus); // points        

            // adjuster value
            double v_plus_adj = linear_interp::interp_3d(
                vfi->p_grid,vfi->n_grid,vfi->m_grid, // grids
                vfi->Np,vfi->Nn,vfi->Nm, // dimensions
                &vfi->sol_v_adj[i_plus_interp], // values
                p_plus,n_plus,m_plus); // points

            // iv. logsumexp
            double logsum = logsumexp(v_plus_keep,v_plus_adj,par);

            // v. accumulate
            w_val += par->xi_w[i_xi]*par->psi_w[i_psi]*logsum;  

        } // i_psi
        } // i_xi

        // d. store
        long long i_wq = index::d4(t,i_p,i_n,i_a,par->T-1,vfi->Np,vfi->Nn,vfi->Na);
        vfi->sol_w[i_wq] = w_val;

    } // i_a
    }  // i_n
    } // i_p

    } // parallel
    
}
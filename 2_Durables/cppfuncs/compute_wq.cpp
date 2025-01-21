#ifndef MAIN
#define COMPUTE_WQ
#include "header.cpp"
#endif

void compute_wq(par_struct *par, egm_struct *egm, long long t, long long *Nns){

    // compute post-decision value (w) and post-decision marginal value of cash (q)

    #pragma omp parallel num_threads(par->cppthreads)
    {

    // loop over states
    #pragma omp for collapse(5)
    for(long long i_p = 0; i_p < egm->Np; i_p++){
    for(long long i_n1 = 0; i_n1 < Nns[0]; i_n1++){
    for(long long i_n2 = 0; i_n2 < Nns[1]; i_n2++){
    for(long long i_n3 = 0; i_n3 < Nns[2]; i_n3++){
    for(long long i_m_pd = 0; i_m_pd < egm->Nm_pd; i_m_pd++){

        double d[MAX_D], n_plus[MAX_D];

        // a. unpack
        double p = egm->p_grid[i_p];
        d[0] = egm->n_grid[i_n1];
        if (par->D >= 2){d[1] = egm->n_grid[i_n2];}
        if (par->D >= 3){d[2] = egm->n_grid[i_n3];}
        double m_pd = egm->m_pd_grid[i_m_pd];

        // b. initialize expectations
        double w_val = 0.0;
        double q_val = 0.0;

        // c. loop over quadrature points
        for(long long i_xi = 0; i_xi < par->Nxi; i_xi++){
        for(long long i_psi = 0; i_psi < par->Npsi; i_psi++){

            // i. unpack
            double xi = par->xi[i_xi];
            double psi = par->psi[i_psi];

            // ii. compute future states
            double p_plus = pow(p,par->rho_p)*xi;
            if (p_plus > egm->p_max){p_plus = egm->p_max;} // enforce upper bound
            if (p_plus < egm->p_min){p_plus = egm->p_min;} // enforce lower bound

            // compute income
            double y = 0.0;
            if(t >= par->T_retired){
                y = par->kappa[t];
            }
            else{
                y = p_plus*psi*par->kappa[t];
            }

            // next period cash on hand
            double m_plus = par->R*m_pd + y;
                        
            // next period durables endowment
            for(long long j = 0; j < par->D; j++){
                n_plus[j] = (1-par->delta[j])*d[j];
                if (n_plus[j] > egm->n_max){n_plus[j] = egm->n_max;} // enforce upper bound
            }

            // iii. interpolate future value and marginal value of cash
            long long i_plus_interp = index::d6(t+1,0,0,0,0,0,par->T,egm->Np,Nns[0],Nns[1],Nns[2],egm->Nm);

            double v_plus = interp(par,egm,p_plus,n_plus,m_plus,&egm->sol_v[i_plus_interp]);
            double vm_plus;
            if(!egm->pure_vfi){                
                vm_plus = interp(par,egm,p_plus,n_plus,m_plus,&egm->sol_vm[i_plus_interp]);
            }

            // iv. accumulate
            w_val += par->xi_w[i_xi]*par->psi_w[i_psi]*par->beta*v_plus;  
            if(!egm->pure_vfi){q_val += par->xi_w[i_xi]*par->psi_w[i_psi]*vm_plus;}

        } // i_psi
        } // i_xi

        // d. store
        long long i_wq = index::d5(i_p,i_n1,i_n2,i_n3,i_m_pd,egm->Np,Nns[0],Nns[1],Nns[2],egm->Nm_pd);
        egm->sol_q[i_wq] = q_val;
        egm->sol_w[i_wq] = w_val;

    } // i_m_pd
    } } } // i_n3 i_n2 i_n1
    } // i_p

    } // parallel
    
}
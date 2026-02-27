#ifndef MAIN
#define COMPUTE_W
#include "header.cpp"
#endif

////////
// 1D //
////////

void compute_w_1d(par_struct *par, vfi_struct *vfi, long long t){

    #pragma omp parallel num_threads(par->cppthreads)
    {

    // loop over states
    #pragma omp for
    for(long long i_p = 0; i_p < vfi->Np; i_p++){
    for(long long i_n = 0; i_n < vfi->Nn; i_n++){
    for(long long i_a = 0; i_a < vfi->Na; i_a++){

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
            double n_plus = (1-par->delta[0])*n;
            if (n_plus > vfi->n_max){n_plus = vfi->n_max;} // enforce upper bound
            if (n_plus < vfi->n_min){n_plus = vfi->n_min;} // enforce lower bound

            // next period cash on hand
            double m_plus = par->R*a + par->kappa[t]*psi*p_plus;
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
            double x_adj = m_plus + (1-par->nu)*n_plus;
            if (x_adj > vfi->x_max){x_adj = vfi->x_max;} // enforce upper bound
            if (x_adj < vfi->x_min){x_adj = vfi->x_min;} // enforce lower bound
            long long i_plus_adj_interp = index::d3(t+1,0,0,par->T,vfi->Np,vfi->Nx);
            double v_plus_adj = linear_interp::interp_2d(
                vfi->p_grid,vfi->x_grid, // grids
                vfi->Np,vfi->Nx, // dimensions
                &vfi->sol_v_adj[i_plus_adj_interp], // values
                p_plus,x_adj); // points

            // iv. logsumexp
            double logsum = utility::logsumexp_1d(v_plus_keep,v_plus_adj,par);

            // v. accumulate
            w_val += par->xi_w[i_xi]*par->psi_w[i_psi]*logsum;  

        } // i_psi
        } // i_xi

        // d. store
        long long i_wq = index::d4(t,i_p,i_n,i_a,par->T-1,vfi->Np,vfi->Nn,vfi->Na);
        vfi->sol_w[i_wq] = w_val;

    } // i_a
    } // i_n
    } // i_p

    } // parallel
    
} // compute_w_1d

////////
// 2D //
////////

void compute_w_2d(par_struct *par, vfi_struct *vfi, long long t){

    #pragma omp parallel num_threads(par->cppthreads)
    {

    // loop over states
    #pragma omp for
    for(long long i_p = 0; i_p < vfi->Np; i_p++){
    for(long long i_n1 = 0; i_n1 < vfi->Nn; i_n1++){
    for(long long i_n2 = 0; i_n2 < vfi->Nn; i_n2++){
    for(long long i_a = 0; i_a < vfi->Na; i_a++){

        double d[2], n_plus[2];

        // a. unpack
        double p = vfi->p_grid[i_p];
        double n1 = vfi->n_grid[i_n1];
        double n2 = vfi->n_grid[i_n2];
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
            double n_plus1 = (1-par->delta[0])*n1;
            if (n_plus1 > vfi->n_max){n_plus1 = vfi->n_max;} // enforce upper bound
            if (n_plus1 < vfi->n_min){n_plus1 = vfi->n_min;} // enforce lower bound

            double n_plus2 = (1-par->delta[1])*n2;
            if (n_plus2 > vfi->n_max){n_plus2 = vfi->n_max;} // enforce upper bound
            if (n_plus2 < vfi->n_min){n_plus2 = vfi->n_min;} // enforce lower bound
            
            // next period cash on hand
            double m_plus = par->R*a + par->kappa[t]*psi*p_plus;
            if (m_plus > vfi->m_max){m_plus = vfi->m_max;} // enforce upper bound
            if (m_plus < vfi->m_min){m_plus = vfi->m_min;} // enforce lower bound

            // iii. interpolate future value
            long long i_plus_interp = index::d5(t+1,0,0,0,0,par->T,vfi->Np,vfi->Nn,vfi->Nn,vfi->Nm);

            // keeper value
            double v_plus_keep = linear_interp::interp_4d(
                vfi->p_grid,vfi->n_grid,vfi->n_grid,vfi->m_grid, // grids
                vfi->Np,vfi->Nn,vfi->Nn,vfi->Nm, // dimensions
                &vfi->sol_v_keep[i_plus_interp], // values
                p_plus,n_plus1, n_plus2,m_plus); // points        

            // adjust durable 1 value
            double x_adj1 = m_plus + (1-par->nu)*n_plus1;
            if (x_adj1 > vfi->x_max){x_adj1 = vfi->x_max;}
            if (x_adj1 < vfi->x_min){x_adj1 = vfi->x_min;}
            long long i_plus_adj1_interp = index::d4(t+1,0,0,0,par->T,vfi->Np,vfi->Nn,vfi->Nx);
            double v_plus_adj1 = linear_interp::interp_3d(
                vfi->p_grid,vfi->n_grid,vfi->x_grid, // grids
                vfi->Np,vfi->Nn,vfi->Nx, // dimensions
                &vfi->sol_v_adj1[i_plus_adj1_interp], // values
                p_plus,n_plus2,x_adj1); // points

            // adjust durable 2 value
            double x_adj2 = m_plus + (1-par->nu)*n_plus2;
            if (x_adj2 > vfi->x_max){x_adj2 = vfi->x_max;}
            if (x_adj2 < vfi->x_min){x_adj2 = vfi->x_min;}
            long long i_plus_adj2_interp = index::d4(t+1,0,0,0,par->T,vfi->Np,vfi->Nn,vfi->Nx);
            double v_plus_adj2 = linear_interp::interp_3d(
                vfi->p_grid,vfi->n_grid,vfi->x_grid, // grids
                vfi->Np,vfi->Nn,vfi->Nx, // dimensions
                &vfi->sol_v_adj2[i_plus_adj2_interp], // values
                p_plus,n_plus1,x_adj2); // points
            
            // adjust both durables value
            double x_adj12 = m_plus + (1-par->nu)*(n_plus1+n_plus2);
            if (x_adj12 > vfi->x_max){x_adj12 = vfi->x_max;}
            if (x_adj12 < vfi->x_min){x_adj12 = vfi->x_min;}
            long long i_plus_adj12_interp = index::d3(t+1,0,0,par->T,vfi->Np,vfi->Nx);
            double v_plus_adj12 = linear_interp::interp_2d(
                vfi->p_grid,vfi->x_grid, // grids
                vfi->Np,vfi->Nx, // dimensions
                &vfi->sol_v_adj12[i_plus_adj12_interp], // values
                p_plus,x_adj12); // points

            // iv. logsumexp
            double logsum = utility::logsumexp_2d(v_plus_keep,v_plus_adj1,v_plus_adj2,v_plus_adj12,par);

            // v. accumulate
            w_val += par->xi_w[i_xi]*par->psi_w[i_psi]*logsum;  

        } // i_psi
        } // i_xi

        // d. store
        long long i_wq = index::d5(t,i_p,i_n1,i_n2,i_a,par->T-1,vfi->Np,vfi->Nn,vfi->Nn,vfi->Na);
        vfi->sol_w[i_wq] = w_val;

    } // i_a
    } // i_n1
    } // i_n2
    } // i_p

    } // parallel
    
} // compute_w_2d
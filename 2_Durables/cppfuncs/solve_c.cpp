#ifndef MAIN
#define SOLVE_C
#include "header.cpp"
#endif

void upperenvelope(double *grid_m_pd, int Nm_pd, double *m_vec, double *c_vec, double *w_vec,
                   double *grid_m, int Nm,
                   double *c_ast_vec, double *v_ast_vec, double *d, par_struct *par)
{
        
    for(int im = 0; im < Nm; im++){
        c_ast_vec[im] = 0.0;
        v_ast_vec[im] = -HUGE_VAL;
    }

    // constraint
    // the constraint is binding if the common m is smaller
    // than the smallest m implied by the EGM step (m_vec[0])

    int im = 0;
    while(im < Nm && grid_m[im] <= m_vec[0]){
            
        // a. consume all
        c_ast_vec[im] = grid_m[im];

        // b. value of choice
        double u = utility::func(c_ast_vec[im],d,par);
        v_ast_vec[im] = u + w_vec[0];
        im += 1;
    
    }

    // upper envellope
    // apply the upper envelope algorithm
    
    for(int im_pd = 0; im_pd < Nm_pd-1; im_pd++){

        // a. a inteval and w slope
        double m_pd_low  = grid_m_pd[im_pd];
        double m_pd_high = grid_m_pd[im_pd+1];
        
        double w_low  = w_vec[im_pd];
        double w_high = w_vec[im_pd+1];

        if(m_pd_low > m_pd_high){continue;}

        double w_slope = (w_high-w_low)/(m_pd_high-m_pd_low);
        
        // b. m inteval and c slope
        double m_low  = m_vec[im_pd];
        double m_high = m_vec[im_pd+1];

        double c_low  = c_vec[im_pd];
        double c_high = c_vec[im_pd+1];

        double c_slope = (c_high-c_low)/(m_high-m_low);

        // c. loop through common grid
        for(int im = 0; im < Nm; im++){

            // i. current m
            double m = grid_m[im];

            // ii. interpolate?
            bool interp = (m >= m_low) && (m <= m_high);     
            bool extrap_above = im_pd == Nm_pd-2 && m > m_vec[Nm_pd-1];

            // iii. interpolation (or extrapolation)
            if(interp | extrap_above){

                // o. implied guess
                double c_guess = c_low + c_slope * (m - m_low);
                double m_pd_guess = m - c_guess;

                // oo. implied post-decision value function
                double w = w_low + w_slope * (m_pd_guess - m_pd_low);               

                // ooo. value-of-choice
                double u = utility::func(c_guess,d,par);
                double v_guess = u + w;

                // oooo. update
                if(v_guess > v_ast_vec[im]){
                    v_ast_vec[im] = v_guess;
                    c_ast_vec[im] = c_guess;
                }
            } // interp / extrap
        } // im
    } // im_pd

} // upperenvelope

void solve_c(par_struct* par, egm_struct* egm, long long t, long long* Nns){

    // solve for consumption given durable choice and 1-3 durables

    #pragma omp parallel num_threads(par->cppthreads)
    {

    //  loop over non-wealth states
    #pragma omp for collapse(4)
    for(long long i_p = 0; i_p < egm->Np; i_p++){  
    for(long long i_n1 = 0; i_n1 < Nns[0]; i_n1++){
    for(long long i_n2 = 0; i_n2 < Nns[1]; i_n2++){
    for(long long i_n3 = 0; i_n3 < Nns[2]; i_n3++){
    
        double d[MAX_D];

        // a. allocate temporary arrays
        double* m_temp = new double[egm->Nm_pd];
        double* c_temp = new double[egm->Nm_pd];
        double* v_ast_temp = new double[egm->Nm_keep];

        // b. unpack
        double p = egm->p_grid[i_p];
        d[0] = egm->n_grid[i_n1];
        if (par->D >= 2) d[1] = egm->n_grid[i_n2];
        if (par->D >= 3) d[2] = egm->n_grid[i_n3];

        // c. loop over post-dec assets and compute consumption on the endogenous grid
        for(long long i_m_pd = 0; i_m_pd < egm->Nm_pd; i_m_pd++){

            // o. unpack
            double m_pd = egm->m_pd_grid[i_m_pd];

            // oo. compute c with EGM
            long long i_q = index::d5(i_p,i_n1,i_n2,i_n3,i_m_pd,egm->Np,Nns[0],Nns[1],Nns[2],egm->Nm_pd);
            double q = egm->sol_q[i_q];
            double c = utility::inverse_marg_func_c(par->beta*par->R*q,d,par);
            double m = m_pd + c;

            // ooo. store
            m_temp[i_m_pd] = m;
            c_temp[i_m_pd] = c;
        
        } // i_m_pd
    
        // d. interpolate c on exogenous grid
        long long i_c_keep = index::d5(i_p,i_n1,i_n2,i_n3,0,egm->Np,Nns[0],Nns[1],Nns[2],egm->Nm_keep);
        long long i_w = index::d5(i_p,i_n1,i_n2,i_n3,0,egm->Np,Nns[0],Nns[1],Nns[2],egm->Nm_pd);
        upperenvelope(
            egm->m_pd_grid,egm->Nm_pd,m_temp,c_temp,&egm->sol_w[i_w],
            egm->m_keep_grid,egm->Nm_keep,
            &egm->sol_c_keep[i_c_keep],v_ast_temp,d,par
        );

        // e. cleanup
        delete [] m_temp;
        delete [] c_temp;
        delete [] v_ast_temp;
    
    } // i_n3
    } // i_n2
    } // i_n1
    } // i_p

    } // parallel
    
} // solve_c
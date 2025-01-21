#ifndef MAIN
#define SOLVE_D
#include "header.cpp"
#endif

double obj(double *choices, void *solver_data_in){ 
    
    solver_struct *solver_data = (solver_struct *) solver_data_in;
    solver_data->func_evals += 1;

    // i. unpack
    auto par = solver_data->par;
    auto egm = solver_data->egm;
    long long t = solver_data->t;
    long long i_p = solver_data->i_p;
    double* n = solver_data->n;
    double* d = solver_data->d;
    double m = solver_data->m;
    long long* Nns = solver_data->Nns;

    // i. compute d and mbar
    double mbar = budget::get_d_and_mbar(m,n,choices,d,par,egm);

    // ii. compute c
    double c, m_pd;
    if(egm->pure_vfi){
        
        m_pd = choices[par->D]*mbar;
        c = mbar - m_pd;

    } else {

        if(t == par->T-1){
            c = mbar;
        } else {
            long long i_c_keep = index::d5(i_p,0,0,0,0,egm->Np,Nns[0],Nns[1],Nns[2],egm->Nm_keep);
            c = interp_c_keep(par,egm,d,mbar,&egm->sol_c_keep[i_c_keep]);
            c = MIN(c,mbar);
        }
        solver_data->c = c; // store consumption
        m_pd = mbar - c;
        solver_data->m_pd_fac = m_pd/mbar; // store savings rate

    }

    /// iii. compute utility
    double u = utility::func(c,d,par);

    // iv. compute objective
    double value_of_choice;
    if(t == par->T-1){
        value_of_choice = u;
    } else {
        long long i_w = index::d5(i_p,0,0,0,0,egm->Np,Nns[0],Nns[1],Nns[2],egm->Nm_pd);
        double w = interp_pd(par,egm,d,m_pd,&egm->sol_w[i_w]);
        value_of_choice  = u + w;
    }

    return -value_of_choice;

}
    
double obj_nlopt(unsigned nx, const double *choices, double *grad, void *solver_data_in){
    
    double choices_[MAX_D];

    solver_struct *solver_data = (solver_struct *) solver_data_in;
    auto par = solver_data->par;

    // i. copy choices
    for(long long j = 0; j < solver_data->Nchoices; j++){
        choices_[j] = choices[j];
    }

    // ii. compute objective
    double obj_val = obj(choices_,solver_data_in);

    // compute gradient
    if(grad){
        
        for(long long j = 0; j < solver_data->Nchoices; j++){

            for(long long k = 0; k < solver_data->Nchoices; k++){
                choices_[k] = choices[k];
            }

            choices_[j] += EPS_GRAD;
            
            double obj_val_forward = obj(choices_,solver_data_in);
            grad[j] = (obj_val_forward - obj_val) / EPS_GRAD;

        }

    }

    return obj_val;

} // obj_nlopt

void solve_d(par_struct* par, egm_struct* egm, long long t, long long* Nns){
    
    #pragma omp parallel num_threads(par->cppthreads)
    {

        double choices[MAX_D+1], choices_low[MAX_D+1], choices_high[MAX_D+1];

        // a. setup for optimizers
        solver_struct* solver_data = new solver_struct;
        solver_data->n = new double[MAX_D];
        solver_data->d = new double[MAX_D];
        solver_data->Nchoices = egm->pure_vfi ? par->D + 1: par->D;

        nlopt_opt opt;
        if(egm->solver == 0){
            opt = nlopt_create(NLOPT_LN_NELDERMEAD,solver_data->Nchoices);
        } else if(egm->solver == 1){
            opt = nlopt_create(NLOPT_LD_SLSQP,solver_data->Nchoices);
        } else if(egm->solver == 2){
            opt = nlopt_create(NLOPT_LD_MMA,solver_data->Nchoices);
        }

        nlopt_set_min_objective(opt,obj_nlopt,solver_data);
        nlopt_set_xtol_rel(opt,1e-6); // relative tolerance
        nlopt_set_maxeval(opt,200); // maximum number of function evaluations

        // b. loop over states
        #pragma omp for collapse(5)
        for(long long i_p = 0; i_p < egm->Np; i_p++){
        for(long long i_n1 = 0; i_n1 < Nns[0]; i_n1++){
        for(long long i_n2 = 0; i_n2 < Nns[1]; i_n2++){
        for(long long i_n3 = 0; i_n3 < Nns[2]; i_n3++){
        for(long long i_m = 0; i_m < egm->Nm; i_m++){

            // i. create solver data
            solver_data->par = par;
            solver_data->egm = egm;
            solver_data->Nns = Nns;
            solver_data->t = t;
            solver_data->i_p = i_p;
            
            solver_data->n[0] = egm->n_grid[i_n1];
            if(par->D >= 2){solver_data->n[1] = egm->n_grid[i_n2];}
            if(par->D >= 3){solver_data->n[2] = egm->n_grid[i_n3];}
            solver_data->m = egm->m_grid[i_m];
            
            double*n = solver_data->n;
            double m = solver_data->m;

            // ii. bounds
            for(long long j = 0; j < solver_data->Nchoices; j++){
                choices_low[j] = egm->min_action;
                choices_high[j] = egm->max_action;          
            }

            nlopt_set_lower_bounds(opt,choices_low);
            nlopt_set_upper_bounds(opt,choices_high);
        
            // iii. call solver
            for(long long j = 0; j < solver_data->Nchoices; j++){
                choices[j] = 0.10;
            }

            solver_data->func_evals = 0;
            double minf;
            int flag = nlopt_optimize(opt,choices,&minf);

            // iv. finalize
            double value_of_choice = obj(choices,solver_data);

            // v. store
            long long i_sol = index::d6(t,i_p,i_n1,i_n2,i_n3,i_m,par->T,egm->Np,Nns[0],Nns[1],Nns[2],egm->Nm);

            // value and marginal value
            egm->sol_v[i_sol] = -value_of_choice;
            if(!egm->pure_vfi){egm->sol_vm[i_sol] = utility::marg_func_c(solver_data->c,solver_data->d,par);}

            // durables
            if(par->nonnegative_investment){ // forced 0.0 at upper bound
                egm->sol_d1_fac[i_sol] = i_n1 == Nns[0]-1 ? 0.0 : choices[0];
                if(par->D >= 2){egm->sol_d2_fac[i_sol] = i_n2 == Nns[1]-1 ? 0.0 : choices[1];}
                if(par->D >= 3){egm->sol_d3_fac[i_sol] = i_n3 == Nns[2]-1 ? 0.0 : choices[2];}
            } else {
                egm->sol_d1_fac[i_sol] = choices[0];
                if(par->D >= 2){egm->sol_d2_fac[i_sol] = choices[1];}
                if(par->D >= 3){egm->sol_d3_fac[i_sol] = choices[2];}
            }

            // savings rate
            if(egm->pure_vfi){
                egm->sol_m_pd_fac[i_sol] = choices[par->D];
            } else {
                egm->sol_m_pd_fac[i_sol] = solver_data->m_pd_fac;
            }

            // flags
            egm->sol_flag[i_sol] = flag;
            egm->sol_func_evals[i_sol] = solver_data->func_evals;

        } // i_m
        } } } // i_n3, i_n2, i_n1
        } // i_p

        // cleanup
        delete [] solver_data->n;
        delete [] solver_data->d;
        nlopt_destroy(opt);
        delete solver_data;

    } // parallel

} // solved
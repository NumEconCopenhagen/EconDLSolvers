#ifndef MAIN
#define SOLVE_ADJ
#include "header.cpp"
#endif

double obj_adj(double *choices, void *solver_data_in){ 
    
    solver_struct *solver_data = (solver_struct *) solver_data_in;
    solver_data->func_evals += 1;

    // i. unpack
    auto par = solver_data->par;
    auto vfi = solver_data->vfi;
    long long t = solver_data->t;
    double n = solver_data->n;
    double m = solver_data->m;
    double p = solver_data->p;

    // ii. compute mbar
    double mbar = m + (1-par->kappa)*n;

    // iii. compute choices
    double c,d, a;
    double exp_share  = choices[0];
    double c_share = choices[1];
    double expend_adj = (1-exp_share)*mbar;
    c = c_share*expend_adj;
    d = (1-c_share)*expend_adj;

    // iv. compute post-decision states
    a = exp_share*mbar;

    /// v. compute utility
    double u = utility::func(c,d,par);

    // vi. compute objective
    double value_of_choice;
    
    if(t == par->T-1){
        
        value_of_choice = u;

    } else {

        long long i_w = index::d4(t,0,0,0,par->T-1,vfi->Np,vfi->Nn,vfi->Na);

        double w = linear_interp::interp_3d(
            vfi->p_grid,vfi->n_grid,vfi->a_grid, // grids
            vfi->Np,vfi->Nn,vfi->Na, // dimensions
            &vfi->sol_w[i_w], // values
            p,d,a); // points

        value_of_choice  = u + par->beta * w;

    }

    return -value_of_choice;

}

double obj_nlopt_adj(unsigned nx, const double *choices, double *grad, void *solver_data_in){
    
    solver_struct *solver_data = (solver_struct *) solver_data_in;
    auto par = solver_data->par;

    // i. copy choices
    double choices_[2];
    choices_[0] = choices[0];
    choices_[1] = choices[1];

    // ii. compute objective
    double obj_val = obj_adj(choices_,solver_data_in);

    // compute gradient
    if(grad){

        // a. gradient of first choice
        choices_[0] = choices[0] + EPS_GRAD;
        if(choices_[0] > 1.0){
            choices_[0] = 0.999999;
        }
        choices_[1] = choices[1];
        double obj_val_forward1 = obj_adj(choices_,solver_data_in);
            
        grad[0] = (obj_val_forward1 - obj_val) / EPS_GRAD;

        // b. gradient of second choice
        choices_[0] = choices[0];
        choices_[1] = choices[1] + EPS_GRAD;
        if(choices_[1] > 1.0){
            choices_[1] = 0.999999;
        }
        double obj_val_forward2 = obj_adj(choices_,solver_data_in);

        grad[1] = (obj_val_forward2 - obj_val) / EPS_GRAD;

        }

    return obj_val;

} // obj_nlopt

void solve_adj(par_struct* par, vfi_struct* vfi, long long t){
    
    #pragma omp parallel num_threads(par->cppthreads)
    {

        // a. setup for optimizers
        solver_struct *solver_data = new solver_struct;

        nlopt_opt opt;
        if(vfi->solver == 0){
            opt = nlopt_create(NLOPT_LN_NELDERMEAD,2);
        } else if(vfi->solver == 1){
            opt = nlopt_create(NLOPT_LD_SLSQP,2);
        } else if(vfi->solver == 2){
            opt = nlopt_create(NLOPT_LD_MMA,2);
        }

        nlopt_set_min_objective(opt,obj_nlopt_adj,solver_data);
        nlopt_set_xtol_rel(opt,1e-6); // relative tolerance
        nlopt_set_maxeval(opt,200); // maximum number of function evaluations

        // b. loop over states
        #pragma omp for
        for(long long i_p = 0; i_p < vfi->Np; i_p++){
        for(long long i_n = 0; i_n < vfi->Nn; i_n++){
        for(long long i_m = 0; i_m < vfi->Nm; i_m++){

            // i. create solver data
            solver_data->par = par;
            solver_data->vfi = vfi;
            solver_data->t = t;
            solver_data->p = vfi->p_grid[i_p];
            solver_data->n = vfi->n_grid[i_n];
            solver_data->m = vfi->m_grid[i_m];
            
            double n = solver_data->n;
            double m = solver_data->m;
            double p = solver_data->p;

            // ii. bounds
            double choices_low[2] = {0.0,1e-8};
            double choices_high[2] = {0.9999999,1.0};

            nlopt_set_lower_bounds(opt,choices_low);
            nlopt_set_upper_bounds(opt,choices_high);

            // iii. multistart solving
            double start_vals[5] = {0.1,0.2,0.3,0.4,0.5};
            double choices[2];
            double best_choice[2];

            int evals_best;
            int flag_best;

            double best_value = 1e100;

            for (int i = 0; i < 5; i++){

                // o. set start values
                choices[0] = start_vals[i];
                choices[1] = start_vals[i];
                    
                solver_data->func_evals = 0;
                double minf;
                int flag = nlopt_optimize(opt,choices,&minf);

                // iv. finalize
                double value_of_choice = obj_adj(choices,solver_data);

                // v. check if best
                if(value_of_choice < best_value){
                    best_value = value_of_choice;
                    best_choice[0] = choices[0];
                    best_choice[1] = choices[1];
                    evals_best = solver_data->func_evals;
                    flag_best = flag;
                }

            }

        // }

            // v. store
            long long i_sol = index::d4(t,i_p,i_n,i_m,par->T,vfi->Np,vfi->Nn,vfi->Nm);

            // value and marginal value
            vfi->sol_v_adj[i_sol] = -best_value;

            // choice
            vfi->sol_exp_share_adj[i_sol] = best_choice[0];
            vfi->sol_c_share_adj[i_sol] = best_choice[1];  

            // flags
            vfi->sol_flag_adj[i_sol] = flag_best;
            vfi->sol_func_evals_adj[i_sol] = evals_best;

        } // i_m
        } // i_n
        } // i_p

        // cleanup
        nlopt_destroy(opt);
        delete solver_data;

    } // parallel

} // solved
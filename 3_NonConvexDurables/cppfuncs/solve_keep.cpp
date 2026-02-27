#ifndef MAIN
#define SOLVE_KEEP
#include "header.cpp"
#endif

/////////
// obj //
/////////

double obj_keep_1d(double *choices, void *solver_data_in){ 
    
    solver_struct *solver_data = (solver_struct *) solver_data_in;
    solver_data->func_evals += 1;

    // i. unpack
    auto par = solver_data->par;
    auto vfi = solver_data->vfi;
    long long t = solver_data->t;
    double n = solver_data->n;
    double m = solver_data->m;
    double p = solver_data->p;

    // ii. compute c
    double sav_share = choices[0];
    double c = m*(1-sav_share);
    double a = m*sav_share;

    /// iii. compute utility
    double u = utility::func_1d(c,n,par);

    // iv. compute objective
    double value_of_choice;
    if(t == par->T-1){
        
        value_of_choice = u;

    } else {
        
        long long i_w = index::d4(t,0,0,0,par->T-1,vfi->Np,vfi->Nn,vfi->Na);
        
        double w = linear_interp::interp_3d(
            vfi->p_grid,vfi->n_grid,vfi->a_grid, // grids
            vfi->Np,vfi->Nn,vfi->Na, // dimensions
            &vfi->sol_w[i_w], // values
            p,n,a); // points
        
        value_of_choice  = u + par->beta * w;
        
    }

    return -value_of_choice;

} // obj_keep_1d

double obj_keep_2d(double *choices, void *solver_data_in){ 
    
    solver_struct *solver_data = (solver_struct *) solver_data_in;
    solver_data->func_evals += 1;

    // i. unpack
    auto par = solver_data->par;
    auto vfi = solver_data->vfi;
    long long t = solver_data->t;
    double n1 = solver_data->n1;
    double n2 = solver_data->n2;
    double m = solver_data->m;
    double p = solver_data->p;

    // ii. compute c
    double c, a;
    double sav_share  = choices[0];
    c = m*(1-sav_share);
    a = m*sav_share;

    /// iii. compute utility
    double u = utility::func_2d(c,n1,n2,par);

    // iv. compute objective
    double value_of_choice;
    if(t == par->T-1){
        
        value_of_choice = u;

    } else {

        long long i_w = index::d5(t,0,0,0,0,par->T-1,vfi->Np,vfi->Nn,vfi->Nn,vfi->Na);

        double w = linear_interp::interp_4d(
            vfi->p_grid,vfi->n_grid,vfi->n_grid,vfi->a_grid, // grids
            vfi->Np,vfi->Nn,vfi->Nn,vfi->Na, // dimensions
            &vfi->sol_w[i_w], // values
            p,n1,n2,a); // points
            
        value_of_choice  = u + par->beta * w;
        
    }

    return -value_of_choice;

} // obj_keep_2d

///////////////////
// nlopt wrapper //
///////////////////

double obj_nlopt_keep(unsigned nx, const double *choices, double *grad, void *solver_data_in){
    
    solver_struct *solver_data = (solver_struct *) solver_data_in;
    auto par = solver_data->par;

    // i. copy choices
    double choices_[1];
    choices_[0] = choices[0];

    // ii. choose 1D or 2D objective
    double obj_val;
    if (solver_data->use_2d_objective) {
        obj_val = obj_keep_2d(choices_, solver_data_in);
    } else {
        obj_val = obj_keep_1d(choices_, solver_data_in);
    }

    // iii. compute gradient (finite differences)
    if (grad) {
        choices_[0] += EPS_GRAD;

        double obj_val_forward;
        if (solver_data->use_2d_objective) {
            obj_val_forward = obj_keep_2d(choices_, solver_data_in);
        } else {
            obj_val_forward = obj_keep_1d(choices_, solver_data_in);
        }

        grad[0] = (obj_val_forward - obj_val) / EPS_GRAD;
    }

    return obj_val;

} // obj_nlopt_keep

///////////
// solve //
///////////

void solve_keep_1d(par_struct* par, vfi_struct* vfi, long long t){
    
    #pragma omp parallel num_threads(par->cppthreads)
    {

        // a. setup for optimizers
        solver_struct *solver_data = new solver_struct;

        nlopt_opt opt;
        if(vfi->solver == 0){
            opt = nlopt_create(NLOPT_LN_NELDERMEAD,1);
        } else if(vfi->solver == 1){
            opt = nlopt_create(NLOPT_LD_SLSQP,1);
        } else if(vfi->solver == 2){
            opt = nlopt_create(NLOPT_LD_MMA,1);
        }

        nlopt_set_min_objective(opt,obj_nlopt_keep,solver_data);
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
            solver_data->use_2d_objective = false;
            
            double n = solver_data->n;
            double m = solver_data->m;
            double p = solver_data->p;

            // ii. bounds
            double choices_low[1] = {0.0};
            double choices_high[1] = {0.999999};

            nlopt_set_lower_bounds(opt,choices_low);
            nlopt_set_upper_bounds(opt,choices_high);
                    
            // iii. multistart solving
            double start_vals[5] = {0.1,0.2,0.3,0.4,0.5};
            double choices[1];
            double best_choice[1];

            int flag_best;
            
            double best_value = 1e100;
            solver_data->func_evals = 0;

            for(int i_start = 0; i_start < 5; i_start++){
            
                // o. set starting values
                choices[0] = start_vals[i_start];

                double minf;
                int flag;
                if (t == par->T-1){
                    choices[0] = 0.0;
                    flag = -10;
                } else{
                    flag = nlopt_optimize(opt,choices,&minf);
                }

                // oo. objective
                double value_of_choice = obj_keep_1d(choices,solver_data);

                // ooo. check if better
                if(value_of_choice < best_value){
                    best_value = value_of_choice;
                    best_choice[0] = choices[0];
                    flag_best = flag;
                }

            }

            // iv. store
            long long i_sol = index::d4(t,i_p,i_n,i_m,par->T,vfi->Np,vfi->Nn,vfi->Nm);

            // value
            vfi->sol_v_keep[i_sol] = -best_value;

            // choice
            vfi->sol_sav_share_keep[i_sol] = best_choice[0];

            // flags
            vfi->sol_flag_keep[i_sol] = flag_best;
            vfi->sol_func_evals_keep[i_sol] = solver_data->func_evals;

        } // i_m
        } // i_n
        } // i_p

        // cleanup
        nlopt_destroy(opt);
        delete solver_data;

    } // parallel

} // solved

void solve_keep_2d(par_struct* par, vfi_struct* vfi, long long t){
    
    #pragma omp parallel num_threads(par->cppthreads)
    {

        // a. setup for optimizers
        solver_struct *solver_data = new solver_struct;

        nlopt_opt opt;
        if(vfi->solver == 0){
            opt = nlopt_create(NLOPT_LN_NELDERMEAD,1);
        } else if(vfi->solver == 1){
            opt = nlopt_create(NLOPT_LD_SLSQP,1);
        } else if(vfi->solver == 2){
            opt = nlopt_create(NLOPT_LD_MMA,1);
        }

        nlopt_set_min_objective(opt,obj_nlopt_keep,solver_data);
        nlopt_set_xtol_rel(opt,1e-6); // relative tolerance
        nlopt_set_maxeval(opt,200); // maximum number of function evaluations

        // b. loop over states
        #pragma omp for
        for(long long i_p = 0; i_p < vfi->Np; i_p++){
        for(long long i_n1 = 0; i_n1 < vfi->Nn; i_n1++){
        for(long long i_n2 = 0; i_n2 < vfi->Nn; i_n2++){
        for(long long i_m = 0; i_m < vfi->Nm; i_m++){

            // i. create solver data
            solver_data->par = par;
            solver_data->vfi = vfi;
            solver_data->t = t;
            solver_data->p = vfi->p_grid[i_p];
            solver_data->n1 = vfi->n_grid[i_n1];
            solver_data->n2 = vfi->n_grid[i_n2];
            solver_data->m = vfi->m_grid[i_m];
            solver_data->use_2d_objective = true;

            double n1 = solver_data->n1;
            double n2 = solver_data->n2;
            double m = solver_data->m;
            double p = solver_data->p;

            // ii. bounds
            double choices_low[1] = {0.0};
            double choices_high[1] = {0.999999};

            nlopt_set_lower_bounds(opt,choices_low);
            nlopt_set_upper_bounds(opt,choices_high);
                    
            // iii. multistart solving
            double start_vals[5] = {0.2, 0.5, 0.8};
            double choices[1];
            double best_choice[1];

            int evals_best;
            int flag_best;
            
            double best_value = 1e100;

            solver_data->func_evals = 0;
            for(int i_start = 0; i_start < 5; i_start++){
            
                // o. set starting values
                choices[0] = start_vals[i_start];

                double minf;
                int flag;
                if (t == par->T-1){
                    choices[0] = 0.0;
                    flag = -10;
                } else{
                    flag = nlopt_optimize(opt,choices,&minf);
                }

                // iv. objective
                double value_of_choice = obj_keep_2d(choices,solver_data);

                // v. check if better
                if(value_of_choice < best_value){
                    best_value = value_of_choice;
                    best_choice[0] = choices[0];
                    flag_best = flag;
                }

            }

            // v. store
            long long i_sol = index::d5(t,i_p,i_n1,i_n2,i_m,par->T,vfi->Np,vfi->Nn,vfi->Nn,vfi->Nm);

            // value and marginal value
            vfi->sol_v_keep[i_sol] = -best_value;

            // choice
            vfi->sol_sav_share_keep[i_sol] = best_choice[0];

            // flags
            vfi->sol_flag_keep[i_sol] = flag_best;
            vfi->sol_func_evals_keep[i_sol] = solver_data->func_evals;

        } // i_m
        } // i_n1
        } // i_n2
        } // i_p

        // cleanup
        nlopt_destroy(opt);
        delete solver_data;

    } // parallel

} // solve_keep_2d
#ifndef MAIN
#define SOLVE_ADJ
#include "header.cpp"
#endif

/////////
// obj //
/////////

double obj_adj_1d(double *choices, void *solver_data_in){ 
    
    solver_struct *solver_data = (solver_struct *) solver_data_in;
    solver_data->func_evals += 1;

    // i. unpack
    auto par = solver_data->par;
    auto vfi = solver_data->vfi;
    long long t = solver_data->t;
    double x = solver_data->x;
    double p = solver_data->p;

    // ii. compute choices
    double exp_share  = choices[0];
    double c_share = choices[1];

    double sav_share = 1 - exp_share;

    double exp = exp_share*x;
    double c = c_share*exp;
    double d = (1-c_share)*exp;

    // iii. compute post-decision states
    double a = sav_share*x;

    /// iv. compute utility
    double u = utility::func_1d(c,d,par);

    // v. compute objective
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

} // obj_adj_1d

double obj_adj1_2d(double *choices, void *solver_data_in){ 
    
    solver_struct *solver_data = (solver_struct *) solver_data_in;
    solver_data->func_evals += 1;

    // i. unpack
    auto par = solver_data->par;
    auto vfi = solver_data->vfi;
    long long t = solver_data->t;
    double n2 = solver_data->n2;
    double x = solver_data->x;
    double p = solver_data->p;

    // ii. compute choices
    double exp_share  = choices[0];
    double c_share = choices[1];

    double sav_share = 1-exp_share;

    double exp = exp_share*x;
    double c = c_share * exp;
    double d1 = (1-c_share)*exp;

    // iii. compute post-decision states
    double a = sav_share*x;

    /// iv. compute utility
    double u = utility::func_2d(c,d1,n2,par);

    // v. compute objective
    double value_of_choice;
    
    if(t == par->T-1){
        
        value_of_choice = u;

    } else {

        long long i_w = index::d5(t,0,0,0,0,par->T-1,vfi->Np,vfi->Nn,vfi->Nn,vfi->Na);

        double w = linear_interp::interp_4d(
            vfi->p_grid,vfi->n_grid,vfi->n_grid,vfi->a_grid, // grids
            vfi->Np,vfi->Nn,vfi->Nn,vfi->Na, // dimensions
            &vfi->sol_w[i_w], // values
            p,d1,n2,a); // points

        value_of_choice  = u + par->beta * w;

    }

    return -value_of_choice;

} // obj_adj1_2d

double obj_adj2_2d(double *choices, void *solver_data_in){ 
    
    solver_struct *solver_data = (solver_struct *) solver_data_in;
    solver_data->func_evals += 1;

    // i. unpack
    auto par = solver_data->par;
    auto vfi = solver_data->vfi;
    long long t = solver_data->t;
    double n1 = solver_data->n1;
    double x = solver_data->x;
    double p = solver_data->p;

    // ii. compute choices
    double exp_share  = choices[0];
    double c_share = choices[1];
    
    double sav_share = 1 - exp_share;

    double exp = exp_share*x;
    double c = c_share*exp;
    double d2 = (1-c_share)*exp;

    // iii. compute post-decision states
    double a = sav_share*x;

    /// iv. compute utility
    double u = utility::func_2d(c,n1,d2,par);

    // v. compute objective
    double value_of_choice;
    
    if(t == par->T-1){
        
        value_of_choice = u;

    } else {

        long long i_w = index::d5(t,0,0,0,0,par->T-1,vfi->Np,vfi->Nn,vfi->Nn,vfi->Na);

        double w = linear_interp::interp_4d(
            vfi->p_grid,vfi->n_grid,vfi->n_grid,vfi->a_grid, // grids
            vfi->Np,vfi->Nn,vfi->Nn,vfi->Na, // dimensions
            &vfi->sol_w[i_w], // values
            p,n1,d2,a); // points

        value_of_choice  = u + par->beta * w;

    }

    return -value_of_choice;

} // obj_adj2_2d

double obj_adj12_2d(double *choices, void *solver_data_in){ 
    
    solver_struct *solver_data = (solver_struct *) solver_data_in;
    solver_data->func_evals += 1;

    // i. unpack
    auto par = solver_data->par;
    auto vfi = solver_data->vfi;
    long long t = solver_data->t;
    double x = solver_data->x;
    double p = solver_data->p;

    // ii. compute choices
    double exp_share = choices[0];
    double d1_share = choices[1];
    double c_share = choices[2];

    double sav_share = 1-exp_share;
    double exp = exp_share*x;

    double c = c_share*exp;
    double exp_d = (1-c_share)*exp;
    double d1 = d1_share*exp_d;
    double d2 = (1-d1_share)*exp_d;

    // iii. compute post-decision states
    double a = sav_share*x;

    /// iv. compute utility
    double u = utility::func_2d(c,d1,d2,par);

    // v. compute objective
    double value_of_choice;
    
    if(t == par->T-1){
        
        value_of_choice = u;

    } else {

        long long i_w = index::d5(t,0,0,0,0,par->T-1,vfi->Np,vfi->Nn,vfi->Nn,vfi->Na);

        double w = linear_interp::interp_4d(
            vfi->p_grid,vfi->n_grid,vfi->n_grid,vfi->a_grid, // grids
            vfi->Np,vfi->Nn,vfi->Nn,vfi->Na, // dimensions
            &vfi->sol_w[i_w], // values
            p,d1,d2,a); // points

        value_of_choice  = u + par->beta * w;

    }

    return -value_of_choice;

} // obj_adj12_2d

////////////////////
// nlopt wrappers //
////////////////////

double obj_nlopt_adj_2choices(unsigned nx, const double *choices, double *grad, void *solver_data_in){
    
    solver_struct *solver_data = (solver_struct *) solver_data_in;
    auto par = solver_data->par;

    // i. copy choices
    double choices_[2];
    choices_[0] = choices[0];
    choices_[1] = choices[1];

    // ii. compute objective

    // adjust on
    double obj_val;
    if (par->D == 1) {
        obj_val = obj_adj_1d(choices_,solver_data_in);
    }
    else if (par->D == 2 && solver_data->adjustment_type == 0) { // adjust durable 1
        obj_val = obj_adj1_2d(choices_,solver_data_in);
    }
    else{ // adjust durable 2
        obj_val = obj_adj2_2d(choices_,solver_data_in);
    }
    
    // compute gradient
    if(grad){

        // a. gradient of first choice
        choices_[0] = choices[0] + EPS_GRAD;
        if(choices_[0] > 1.0){ choices_[0] = 0.999999; }
        choices_[1] = choices[1];

        double obj_val_forward1;
        if (par->D == 1) {
            obj_val_forward1 = obj_adj_1d(choices_,solver_data_in);
        }
        else if (par->D == 2 && solver_data->adjustment_type == 0) { // adjust durable 1
            obj_val_forward1 = obj_adj1_2d(choices_,solver_data_in);
        }
        else{ // adjust durable 2
            obj_val_forward1 = obj_adj2_2d(choices_,solver_data_in);
        }

        grad[0] = (obj_val_forward1 - obj_val) / EPS_GRAD;

        // b. gradient of second choice
        choices_[0] = choices[0];
        choices_[1] = choices[1] + EPS_GRAD;
        if(choices_[1] > 1.0){choices_[1] = 0.999999;}

        double obj_val_forward2;
        if (par->D == 1) {
            obj_val_forward2 = obj_adj_1d(choices_,solver_data_in);
        }
        else if (par->D == 2 && solver_data->adjustment_type == 0) { // adjust durable 1
            obj_val_forward2 = obj_adj1_2d(choices_,solver_data_in);
        }
        else{ // adjust durable 2
            obj_val_forward2 = obj_adj2_2d(choices_,solver_data_in);
        }

        grad[1] = (obj_val_forward2 - obj_val) / EPS_GRAD;

        }

    return obj_val;

} // obj_nlopt_adj_2choices


double obj_nlopt_adj_3choices(unsigned nx, const double *choices, double *grad, void *solver_data_in){
    
    solver_struct *solver_data = (solver_struct *) solver_data_in;
    auto par = solver_data->par;

    // i. copy choices
    double choices_[3];
    choices_[0] = choices[0];
    choices_[1] = choices[1];
    choices_[2] = choices[2];

    // ii. compute objective
    double obj_val = obj_adj12_2d(choices_,solver_data_in);

    // compute gradient
    if(grad){

        // a. gradient of first choice
        choices_[0] = choices[0] + EPS_GRAD;
        if(choices_[0] > 1.0){choices_[0] = 0.999999;}
        choices_[1] = choices[1];
        choices_[2] = choices[2];
        double obj_val_forward1 = obj_adj12_2d(choices_,solver_data_in);

        grad[0] = (obj_val_forward1 - obj_val) / EPS_GRAD;

        // b. gradient of second choice
        choices_[0] = choices[0];
        choices_[1] = choices[1] + EPS_GRAD;        
        if(choices_[1] > 1.0){choices_[1] = 0.999999;}
        choices_[2] = choices[2];
        double obj_val_forward2 = obj_adj12_2d(choices_,solver_data_in);
        grad[1] = (obj_val_forward2 - obj_val) / EPS_GRAD;

        // c. gradient of third choice
        choices_[0] = choices[0];
        choices_[1] = choices[1];
        choices_[2] = choices[2] + EPS_GRAD;
        if(choices_[2] > 1.0){choices_[2] = 0.999999;}
        double obj_val_forward3 = obj_adj12_2d(choices_,solver_data_in);
        grad[2] = (obj_val_forward3 - obj_val) / EPS_GRAD;
    }

    return obj_val;

} // obj_nlopt_adj_3choices

///////////
// solve //
///////////

void solve_adj_1d(par_struct* par, vfi_struct* vfi, long long t){
    
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

        nlopt_set_min_objective(opt,obj_nlopt_adj_2choices,solver_data);
        nlopt_set_xtol_rel(opt,1e-6); // relative tolerance
        nlopt_set_maxeval(opt,200); // maximum number of function evaluations

        // b. loop over states
        #pragma omp for
        for(long long i_p = 0; i_p < vfi->Np; i_p++){
        for(long long i_x = 0; i_x < vfi->Nx; i_x++){

            // i. create solver data
            solver_data->par = par;
            solver_data->vfi = vfi;
            solver_data->t = t;
            solver_data->p = vfi->p_grid[i_p];
            solver_data->x = vfi->x_grid[i_x];
            
            double x = solver_data->x;
            double p = solver_data->p;

            // ii. bounds
            double choices_low[2] = {1e-8,1e-8};
            double choices_high[2] = {1.0,1.0};

            nlopt_set_lower_bounds(opt,choices_low);
            nlopt_set_upper_bounds(opt,choices_high);

            // iii. multistart solving
            double start_vals[3] = {0.2,0.5,0.8};
            double choices[2];
            double best_choice[2];

            int flag_best;

            double best_value = 1e100;
            solver_data->func_evals = 0;

            for (int i = 0; i < 3; i++){

                // o. set start values
                choices[0] = start_vals[i];
                choices[1] = start_vals[i];
                    
                double minf;
                int flag = nlopt_optimize(opt,choices,&minf);

                // iv. finalize
                double value_of_choice = obj_adj_1d(choices,solver_data);

                // v. check if best
                if(value_of_choice < best_value){
                    best_value = value_of_choice;
                    best_choice[0] = choices[0];
                    best_choice[1] = choices[1];
                    flag_best = flag;
                }

            }

            // v. store
            long long i_sol = index::d3(t,i_p,i_x,par->T,vfi->Np,vfi->Nx);

            // value and marginal value
            vfi->sol_v_adj[i_sol] = -best_value;

            // choice
            vfi->sol_exp_share_adj[i_sol] = best_choice[0];
            vfi->sol_c_share_adj[i_sol] = best_choice[1];  

            // flags
            vfi->sol_flag_adj[i_sol] = flag_best;
            vfi->sol_func_evals_adj[i_sol] = solver_data->func_evals;

        } // i_x
        } // i_p

        // cleanup
        nlopt_destroy(opt);
        delete solver_data;

    } // parallel

} // solve_adj_1d

void solve_adj_single_2d(par_struct* par, vfi_struct* vfi, long long t, long long adjustment_type){
    
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

        nlopt_set_min_objective(opt,obj_nlopt_adj_2choices,solver_data);
        nlopt_set_xtol_rel(opt,1e-6); // relative tolerance
        nlopt_set_maxeval(opt,200); // maximum number of function evaluations

        // b. loop over states
        #pragma omp for
        for(long long i_p = 0; i_p < vfi->Np; i_p++){
        for(long long i_n = 0; i_n < vfi->Nn; i_n++){
        for(long long i_x = 0; i_x < vfi->Nx; i_x++){

            // i. create solver data
            solver_data->par = par;
            solver_data->vfi = vfi;
            solver_data->t = t;
            solver_data->p = vfi->p_grid[i_p];
            double n1, n2;
            if (adjustment_type == 0) { // adjust durable 1
                solver_data->n2 = vfi->n_grid[i_n];
                n1 = 0.0;
                n2 = solver_data->n2;                
            }
            else { // adjust durable 2
                solver_data->n1 = vfi->n_grid[i_n];
                n2 = 0.0;
                n1 = solver_data->n1;
            }
            solver_data->x = vfi->x_grid[i_x];
            solver_data->adjustment_type = adjustment_type;

            double x = solver_data->x;
            double p = solver_data->p;

            // ii. bounds
            double choices_low[2] = {1e-8,1e-8};
            double choices_high[2] = {1.0,1.0};

            nlopt_set_lower_bounds(opt,choices_low);
            nlopt_set_upper_bounds(opt,choices_high);

            // iii. multistart solving
            double start_vals_c[2] = {0.3,0.8};
            double start_vals_exp[2] = {0.3,0.8};
            double choices[2];
            double best_choice[2];

            int flag_best;

            double best_value = 1e100;
            solver_data->func_evals = 0;

            for (int i_c = 0; i_c < 2 ; i_c++){
            for (int i_exp = 0; i_exp < 2 ; i_exp++){

                // o. set start values
                choices[0] = start_vals_exp[i_exp];
                choices[1] = start_vals_c[i_c];

                double minf;
                int flag = nlopt_optimize(opt,choices,&minf);

                // iv. finalize
                double value_of_choice;
                if (solver_data->adjustment_type == 0){ // adjust durable 1
                    value_of_choice = obj_adj1_2d(choices,solver_data);
                } else { // adjust durable 2
                    value_of_choice = obj_adj2_2d(choices,solver_data);
                }

                // v. check if best
                if(value_of_choice < best_value){
                    best_value = value_of_choice;
                    best_choice[0] = choices[0];
                    best_choice[1] = choices[1];
                    flag_best = flag;
                }
            
            } // i_c
            } // i_exp

            // v. store
            long long i_sol = index::d4(t,i_p,i_n,i_x,par->T,vfi->Np,vfi->Nn,vfi->Nx);

            if (solver_data->adjustment_type == 0){ // adjust durable 1

                vfi->sol_v_adj1[i_sol] = -best_value;
                vfi->sol_exp_share_adj1[i_sol] = best_choice[0];
                vfi->sol_c_share_adj1[i_sol] = best_choice[1];
                vfi->sol_flag_adj1[i_sol] = flag_best;
                vfi->sol_func_evals_adj1[i_sol] = solver_data->func_evals;

            } else { // adjust durable 2

                vfi->sol_v_adj2[i_sol] = -best_value;
                vfi->sol_exp_share_adj2[i_sol] = best_choice[0];
                vfi->sol_c_share_adj2[i_sol] = best_choice[1];
                vfi->sol_flag_adj2[i_sol] = flag_best;
                vfi->sol_func_evals_adj2[i_sol] = solver_data->func_evals;
            }


        } // i_x
        } // i_n1
        } // i_p

        // cleanup
        nlopt_destroy(opt);
        delete solver_data;

    } // parallel

} // solve_adj_2d

void solve_adj_double_2d(par_struct* par, vfi_struct* vfi, long long t){
    
    #pragma omp parallel num_threads(par->cppthreads)
    {

        // a. setup for optimizers
        solver_struct *solver_data = new solver_struct;

        nlopt_opt opt;
        if(vfi->solver == 0){
            opt = nlopt_create(NLOPT_LN_NELDERMEAD,3);
        } else if(vfi->solver == 1){
            opt = nlopt_create(NLOPT_LD_SLSQP,3);
        } else if(vfi->solver == 2){
            opt = nlopt_create(NLOPT_LD_MMA,3);
        }

        nlopt_set_min_objective(opt,obj_nlopt_adj_3choices,solver_data);
        nlopt_set_xtol_rel(opt,1e-6); // relative tolerance
        nlopt_set_maxeval(opt,200); // maximum number of function evaluations

        // b. loop over states
        #pragma omp for
        for(long long i_p = 0; i_p < vfi->Np; i_p++){
        for(long long i_x = 0; i_x < vfi->Nx; i_x++){

            // i. create solver data
            solver_data->par = par;
            solver_data->vfi = vfi;
            solver_data->t = t;
            solver_data->p = vfi->p_grid[i_p];
            solver_data->x = vfi->x_grid[i_x];
            double x = solver_data->x;
            double p = solver_data->p;

            // ii. bounds
            double choices_low[3] = {1e-8,1e-8,0.0};
            double choices_high[3] = {1.0,1.0,1.0};

            nlopt_set_lower_bounds(opt,choices_low);
            nlopt_set_upper_bounds(opt,choices_high);

            // iii. multistart solving
            double start_vals_c[2] = {0.3,0.8};
            double start_vals_d1[2] = {0.3,0.8};
            double start_vals_exp[2] = {0.3,0.8};
            double choices[3];
            double best_choice[3];

            int flag_best;

            double best_value = 1e100;
            solver_data->func_evals = 0;

            for (int i_c = 0; i_c < 2; i_c++){
            for (int i_d1 = 0; i_d1 < 2; i_d1++){
            for (int i_exp = 0; i_exp < 2; i_exp++){

                // o. set start values
                choices[0] = start_vals_exp[i_exp];
                choices[1] = start_vals_d1[i_d1];
                choices[2] = start_vals_c[i_c];

                double minf;
                int flag = nlopt_optimize(opt,choices,&minf);

                // iv. finalize
                double value_of_choice = obj_adj12_2d(choices,solver_data);

                // v. check if best
                if(value_of_choice < best_value){
                    best_value = value_of_choice;
                    best_choice[0] = choices[0];
                    best_choice[1] = choices[1];
                    best_choice[2] = choices[2];
                    flag_best = flag;
                }

            } } }

            // v. store
            long long i_sol = index::d3(t,i_p,i_x,par->T,vfi->Np,vfi->Nx);

            vfi->sol_v_adj12[i_sol] = -best_value; 
            vfi->sol_exp_share_adj12[i_sol] = best_choice[0]; 
            vfi->sol_d1_share_adj12[i_sol] = best_choice[1];
            vfi->sol_c_share_adj12[i_sol] = best_choice[2];
            vfi->sol_flag_adj12[i_sol] = flag_best;
            vfi->sol_func_evals_adj12[i_sol] = solver_data->func_evals;

        } // i_x
        } // i_p

        // cleanup
        nlopt_destroy(opt);
        delete solver_data;

    } // parallel

} // solve_adj_2d
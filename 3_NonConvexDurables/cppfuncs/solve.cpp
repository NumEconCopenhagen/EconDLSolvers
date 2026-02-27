#ifndef MAIN
#include "header.cpp"
#endif

void solve_1d(par_struct* par, vfi_struct* vfi){
    
    logs::write("log.txt",0,"Starting...\n");
    logs::write("log.txt",1,"par->cpp = %d threads\n",par->cppthreads);

    //  loop backwards over time
    HighResTimer timer;
    for(long long t = par->T-1; t >= 0; t--){

        logs::write("log.txt",1,"t = %d\n",t);
        timer.StartTimer();

        // i. compute the post-decision value
        if(t < par->T-1){
            logs::write("log.txt",1," compute_wq\n",t);
            compute_w_1d(par,vfi,t);
        }

        // ii. solve the keeper problem
        logs::write("log.txt",1," solve_keep\n",t);
        solve_keep_1d(par,vfi,t);

        // iii. solve the adjuster problem
        logs::write("log.txt",1," solve_adj\n",t);
        solve_adj_1d(par,vfi,t);

        double time = timer.StopTimer();
        logs::write("log.txt",1," in %.1f secs\n",time);

    } // t
    
} // solve

void solve_2d(par_struct* par, vfi_struct* vfi){
    
    // solve model

    logs::write("log.txt",0,"Starting...\n");
    logs::write("log.txt",1,"par->cpp = %d threads\n",par->cppthreads);

    //  loop backwards over time
    HighResTimer timer_total;
    HighResTimer timer_w;
    HighResTimer timer_keep;
    HighResTimer timer_adj1;
    HighResTimer timer_adj2;
    HighResTimer timer_adj12;

    for(long long t = par->T-1; t >= 0; t--){

        logs::write("log.txt",1,"t = %d\n",t);
        timer_total.StartTimer();

        // i. compute the post-decision value and marginal value of cash-in-hand
        timer_w.StartTimer();
        if(t < par->T-1){
            logs::write("log.txt",1," compute wq\n",t);
            compute_w_2d(par,vfi,t);
        }
        double time_w = timer_w.StopTimer();
        logs::write("log.txt",1," compute wq in %.1f secs\n",time_w);

        // ii. solve the keeper problem
        logs::write("log.txt",1," solve keeper\n",t);
        timer_keep.StartTimer();
        solve_keep_2d(par,vfi,t);
        double time_keep = timer_keep.StopTimer();
        logs::write("log.txt",1," solve keeper in %.1f secs\n",time_keep);

        // iii. solve the adjuster problem for durable 1
        logs::write("log.txt",1," solve durable 1 adjustment \n",t);
        timer_adj1.StartTimer();
        long long adjustment_type = 0; // adjust durable 1
        solve_adj_single_2d(par,vfi,t,adjustment_type);
        double time_adj1 = timer_adj1.StopTimer();
        logs::write("log.txt",1," solve durable 1 adjustment in %.1f secs\n",time_adj1);

        // iv. solve the adjuster problem for durable 2
        adjustment_type = 1; // adjust durable 2
        logs::write("log.txt",1," solve durable 2 adjustment \n",t);
        timer_adj2.StartTimer();
        solve_adj_single_2d(par,vfi,t,adjustment_type);
        double time_adj2 = timer_adj2.StopTimer();
        logs::write("log.txt",1," solve durable 2 adjustment in %.1f secs\n",time_adj2);

        // v. solve the double adjustment problem
        logs::write("log.txt",1," solve double adjustment \n",t);
        timer_adj12.StartTimer();
        solve_adj_double_2d(par,vfi,t);
        double time_adj12 = timer_adj12.StopTimer();
        logs::write("log.txt",1," solve double adjustment in %.1f secs\n",time_adj12);

        double time = timer_total.StopTimer();
        logs::write("log.txt",1," in %.1f secs\n",time);

    } // t
    
} // solve

EXPORT void solve_all(par_struct* par, vfi_struct* vfi){
    
    // solve model
    if (par->D == 1) {
        solve_1d(par,vfi);
    } else if (par->D == 2) {
        solve_2d(par,vfi);
    } else {
        logs::write("log.txt",0,"Error: par->D must be 1 or 2.\n");
        exit(1);
    }
    
} // solve

EXPORT void simulate_cpp(par_struct* par, vfi_struct* vfi, sim_struct* sim){

    // currently only 1D implemented
    if (par->D == 1) {
        simulate_1D_cpp(par,vfi,sim);
    } else if (par->D == 2) {
        simulate_2D_cpp(par,vfi,sim);
    } else {
        logs::write("log.txt",0,"Error: par->D must be 1 or 2.\n");
        exit(1);
    }

} // simulate

EXPORT void compute_euler_errors_cpp(par_struct* par, sim_struct* sim, vfi_struct* vfi){

    // currently only 1D implemented¨
    if (par-> D == 1) {
        compute_euler_errors_1D_cpp(par,sim,vfi);
    }
        else if (par-> D == 2) {
    compute_euler_errors_2D_cpp(par,sim,vfi);
    } else {
        logs::write("log.txt",0,"Error: par->D must be 1 or 2.\n");
        exit(1);
    }

} // compute euler errors
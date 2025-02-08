#ifndef MAIN
#include "header.cpp"
#endif

EXPORT void solve_all(par_struct* par, vfi_struct* vfi){
    
    // solve model

    logs::write("log.txt",0,"Starting...\n");
    logs::write("log.txt",1,"par->cpp = %d threads\n",par->cppthreads);

    //  loop backwards over time
    HighResTimer timer;
    for(long long t = par->T-1; t >= 0; t--){

        logs::write("log.txt",1,"t = %d\n",t);
        timer.StartTimer();

        // i. compute the post-decision value and marginal value of cash-in-hand
        if(t < par->T-1){
            logs::write("log.txt",1," compute_wq\n",t);
            compute_w(par,vfi,t);
        }

        // ii. solve the consumption problem with VFI for given durable choice
        logs::write("log.txt",1," solve_keeper\n",t);
        solve_keep(par,vfi,t);

        // iii. solve the durable choice problem with optimizer (also consumption in last period)
        logs::write("log.txt",1," solve_d\n",t);
        solve_adj(par,vfi,t);

        double time = timer.StopTimer();
        logs::write("log.txt",1," in %.1f secs\n",time);

    } // t
    
} // solve
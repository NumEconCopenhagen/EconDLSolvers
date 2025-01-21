#ifndef MAIN
#include "header.cpp"
#endif

EXPORT void solve_all(par_struct* par, egm_struct* egm){
    
    // solve model

    logs::write("log.txt",0,"Starting...\n");
    logs::write("log.txt",1,"par->cpp = %d threads\n",par->cppthreads);

    // a. precompute number of states
    long long Nns[MAX_D];
    fill_NNs(Nns,par,egm);

    // b. loop backwards over time
    HighResTimer timer;
    for(long long t = par->T-1; t >= 0; t--){

        logs::write("log.txt",1,"t = %d\n",t);
        timer.StartTimer();

        // i. compute the post-decision value and marginal value of cash-in-hand
        if(t < par->T-1){
            logs::write("log.txt",1," compute_wq\n",t);
            compute_wq(par,egm,t,Nns);
        }

        // ii. solve the consumption problem with EGM for given durable choice
        if(t < par->T-1 && !egm->pure_vfi){
            logs::write("log.txt",1," solve_c\n",t);
            solve_c(par,egm,t,Nns);
        } 

        // iii. solve the durable choice problem with optimizer (also consumption in last period)
        logs::write("log.txt",1," solve_d\n",t);
        solve_d(par,egm,t,Nns);

        double time = timer.StopTimer();
        logs::write("log.txt",1," in %.1f secs\n",time);

    } // t
    
} // solve
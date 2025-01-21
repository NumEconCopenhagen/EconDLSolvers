#ifndef MAIN
#include "header.cpp"
#endif

EXPORT void simulate(par_struct* par, egm_struct* egm, sim_struct* sim){

    // a. precompute number of states
    long long Nns[MAX_D];
    fill_NNs(Nns,par,egm);

    // b. simulate
    #pragma omp parallel num_threads(par->cppthreads)
    {

    #pragma omp for
    for(long long i = 0; i < sim->N; i++){
    for(long long t = 0; t < par->T; t++){

        long long index = index::d2(t,i,par->T,sim->N);

        // i. states
        long long index_states = index::d3(t,i,0,par->T,sim->N,par->Nstates);
        double m = sim->states[index_states+0];
        double p = sim->states[index_states+1];
        double* n = &sim->states[index_states+2];

        // ii. actions
        long long index_actions = index::d3(t,i,0,par->T,sim->N,par->Nactions);
        long long index_sol = index::d6(t,0,0,0,0,0,par->T,egm->Np,Nns[0],Nns[1],Nns[2],egm->Nm);

        sim->actions[index_actions] = BOUND(interp(par,egm,p,n,m,&egm->sol_m_pd_fac[index_sol]),egm->min_action,egm->max_action);
        sim->actions[index_actions+1] = BOUND(interp(par,egm,p,n,m,&egm->sol_d1_fac[index_sol]),egm->min_action,egm->max_action);
        if(par->D >= 2){sim->actions[index_actions+2] = BOUND(interp(par,egm,p,n,m,&egm->sol_d2_fac[index_sol]),egm->min_action,egm->max_action);}
        if(par->D >= 3){sim->actions[index_actions+3] = BOUND(interp(par,egm,p,n,m,&egm->sol_d3_fac[index_sol]),egm->min_action,egm->max_action);}

        // iii. outcomes
        long long index_outcomes = index::d3(t,i,0,par->T,sim->N,par->Noutcomes);

        double* c = &sim->outcomes[index_outcomes];
        double* d = &sim->outcomes[index_outcomes+1];
        double* m_pd = &sim->outcomes[index_outcomes+1+par->D];

        // iv. d and mbar
        double mbar = budget::get_d_and_mbar(m,n,&sim->actions[index_actions+1],d,par,egm);

        // v. a and con
        m_pd[0] = sim->actions[index_actions]*mbar;
        c[0] = mbar-m_pd[0];

        // vi. reward
        sim->reward[index] = utility::func(c[0],d,par);

        // vii. post-decision states
        long long index_states_pd = index::d3(t,i,0,par->T,sim->N,par->Nstates);
        sim->states_pd[index_states_pd+0] = m_pd[0];
        sim->states_pd[index_states_pd+1] = p;
        for(long long j = 0; j < par->D; j++){
            sim->states_pd[index_states_pd+2+j] = d[j];
        }

        // viii. next period states
        if(t < par->T-1){
            
            // i. unpack
            long long index_states_plus = index::d3(t+1,i,0,par->T,sim->N,par->Nstates);
            double* m_plus = &sim->states[index_states_plus+0];
            double* p_plus = &sim->states[index_states_plus+1];
            double* n_plus = &sim->states[index_states_plus+2];

            long long index_shocks_plus = index::d3(t+1,i,0,par->T,sim->N,par->Nshocks);
            double xi_plus = sim->shocks[index_shocks_plus+0];
            double psi_plus = sim->shocks[index_shocks_plus+1];

            // ii. permanent income
            p_plus[0] = pow(p,par->rho_p)*xi_plus;

            // iii. durables
            for(long long j = 0; j < par->D; j++){
                n_plus[j] = (1-par->delta[j])*d[j];
            }

            // iv. cash-on-hand
            double y_plus;
            if(t < par->T_retired){
                y_plus  = par->kappa[t]*p_plus[0]*psi_plus;
            } else {
                y_plus  = par->kappa[t];
            }

            m_plus[0] = par->R*m_pd[0] + y_plus;

        } // t < T-1

    } // t
    } // i

    } // parallel
    
} // simulate
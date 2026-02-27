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




EXPORT void compute_euler_errors(par_struct* par, egm_struct* egm, sim_struct* sim){
    // Compute Euler errors for c_t from u_c(c_t, d_t) = beta * R * E[ u_c(c_{t+1}, d_{t+1}) ].

    // Precompute size of n-grids used in the sol_*_fac indexing
    long long Nns[MAX_D];
    fill_NNs(Nns, par, egm);

    #pragma omp parallel for num_threads(par->cppthreads)
    for (long long i = 0; i < sim->N; ++i){
        for (long long t = 0; t < par->T-1; ++t){

            // --- indices
            const long long idx2 = index::d2(t, i, par->T, sim->N); // for flat arrays like reward/euler_error
            const long long idx_out = index::d3(t, i, 0, par->T, sim->N, par->Noutcomes);
            const long long idx_spd = index::d3(t, i, 0, par->T, sim->N, par->Nstates);

            // --- today's outcomes and post-decision states
            const double  c_today  = sim->outcomes[idx_out + 0];
            double* d_today  = &sim->outcomes[idx_out + 1];                // length D
            const double  m_pd_t   = sim->outcomes[idx_out + 1 + par->D];

            const double  p_t      = sim->states_pd[idx_spd + 1];                // p_t (post-decision)
            // states_pd stores d_t already (post-decision); equals d_today
            // const double* n_t    = &sim->states_pd[idx_spd + 2];

            // --- expectation of next-period marginal utility
            double Emu_next = 0.0;

            for (long long i_xi = 0; i_xi < par->Nxi; ++i_xi){
                const double xi   = par->xi[i_xi];
                const double w_xi = par->xi_w[i_xi];

                for (long long i_psi = 0; i_psi < par->Npsi; ++i_psi){
                    const double psi   = par->psi[i_psi];
                    const double w_psi = par->psi_w[i_psi];

                    // --- (t+1) states before decision
                    const double p_next = pow(p_t, par->rho_p) * xi;

                    double n_next[MAX_D];
                    for (long long j = 0; j < par->D; ++j){
                        n_next[j] = (1.0 - par->delta[j]) * d_today[j];
                    }

                    double y_next;
                    if (t < par->T_retired){
                        y_next = par->kappa[t] * p_next * psi;
                    } else {
                        y_next = par->kappa[t];
                    }

                    const double m_next = par->R * m_pd_t + y_next;

                    // --- interpolate policy at t+1
                    const long long idx_sol_next =
                        index::d6(t+1, 0, 0, 0, 0, 0,
                                  par->T, egm->Np, Nns[0], Nns[1], Nns[2], egm->Nm);

                    double a_vec[1 + MAX_D]; // [share_to_a, shares_to_d1..dD]
                    a_vec[0] = BOUND(interp(par, egm, p_next, n_next, m_next, &egm->sol_m_pd_fac[idx_sol_next]),
                                     egm->min_action, egm->max_action);
                    if (par->D >= 1)
                        a_vec[1] = BOUND(interp(par, egm, p_next, n_next, m_next, &egm->sol_d1_fac[idx_sol_next]),
                                         egm->min_action, egm->max_action);
                    if (par->D >= 2)
                        a_vec[2] = BOUND(interp(par, egm, p_next, n_next, m_next, &egm->sol_d2_fac[idx_sol_next]),
                                         egm->min_action, egm->max_action);
                    if (par->D >= 3)
                        a_vec[3] = BOUND(interp(par, egm, p_next, n_next, m_next, &egm->sol_d3_fac[idx_sol_next]),
                                         egm->min_action, egm->max_action);

                    // --- map actions -> (d_{t+1}, c_{t+1})
                    double d_next[MAX_D];
                    const double mbar_next = budget::get_d_and_mbar(m_next, n_next, &a_vec[1], d_next, par, egm);
                    const double m_pd_next = a_vec[0] * mbar_next;
                    const double c_next    = mbar_next - m_pd_next;

                    // --- marginal utility at t+1 and accumulate expectation
                    const double mu_next = utility::marg_func_c(c_next, d_next, par);
                    Emu_next += w_xi * w_psi * mu_next;
                }
            }

            // --- implied c_t from Euler and Euler error
            const double mu_rhs   = par->beta * par->R * Emu_next;
            const double c_euler  = utility::inverse_marg_func_c(mu_rhs, d_today, par);
            sim->euler_error[idx2] = c_euler / c_today - 1.0;
        }
    }
}



// EXPORT void compute_euler_errors(par_struct* par, egm_struct* egm, sim_struct* sim){
//     // compute euler errors

//     #pragma omp parallel num_threads(par->cppthreads)
//     {

//         #pragma omp for
//         for (long long i = 0; i < sim->N; i++){
//         for (long long t = 0; t < par->T-1; t++){

//             long long index = index::d2(t,i,par->T,sim->N);
//             double* c = &sim->outcomes[index];
//             double* d = &sim->outcomes[index+1];
//             double* m_pd = &sim->outcomes[index+1+par->D];
//             double p = sim->states_pd[index+1];
//             double* n = &sim->states_pd[index+2];


//             double exp_marg_util_next = 0.0;
//             for  (long long i_xi = 0; i_xi < par->Nxi; i_xi++){
//             for (long long i_psi = 0; i_psi < par->Npsi; i_psi++){


//             } // for i_psi
//             } // for i_xi

//         } // for t
//         } // for i

//     } // parallel


// } // compute_euler_errors


    // def compute_euler_errors_DP(self):
    //     """ compute euler error"""

    //     par = self.par
    //     sim = self.sim
    //     train = self.train

    //     # a. get consumption and states today
    //     c = sim.outcomes[:par.T-1,:,0]
    //     d = sim.outcomes[:par.T-1,:,1:1+par.D]
    //     states = sim.states[:par.T-1,]
    //     states_pd = sim.states_pd[:par.T-1,]
    //     actions = sim.actions[:par.T-1]
    //     outcomes = sim.outcomes[:par.T-1]

    //     # c. compute next-period expected marginal utility
    //     exp_marg_util_next = np.zeros_like(c)
    //     for t in range(par.T-1):
    //         print(f't={t}')
    //         for i_xi, xi in enumerate(par.xi):
    //             for i_psi, psi in enumerate(par.psi):

    //                 # i. state transition
    //                 p_next = states_pd[t,:,1]**par.rho_p*xi
    //                 n_next = states_pd[t,:,2:2+par.D] * (1-par.delta)
    //                 y = p_next * par.kappa[t] * psi
    //                 m_next = par.R*states_pd[t,:,0] + y
    //                 states_next = np.concatenate((m_next[:,None],p_next[:,None],n_next),axis=1)

    //                 # ii. next period actions
    //                 actions_next = self.interp_actions(states_next,t)

    //                 # iii. next period marginal utility of consumption
    //                 c_next,d_next,a_next = compute_d_and_c_from_action(par,states_next,actions_next)
    //                 marg_util_next = marg_util_c_np(c_next,d_next,par,train)
    //                 exp_marg_util_next[t] += par.xi_w[i_xi]*par.psi_w[i_psi]*marg_util_next
        
    //     # d. euler error
    //     sim.euler_error[:par.T-1] = inv_marg_util_c_np(par.R*par.beta*exp_marg_util_next,d,par,train) / c - 1
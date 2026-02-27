#ifndef MAIN
#define EULER_ERRORS
#include "header.cpp"
#endif

////////
// 1D //
////////

void compute_euler_errors_1D_cpp(par_struct* par, sim_struct* sim, vfi_struct* vfi){
    
    logs::write("log.txt",0,"Computing Euler errors 1D...\n");

    const long long T = par->T;
    const long long N = sim->N;

    #pragma omp parallel num_threads(par->cppthreads)
    {

        for(long long t = 0; t < T-1; t++){

            #pragma omp single
            logs::write("log.txt",1," t = %lld\n",t);

            #pragma omp for
            for(long long i = 0; i < N; i++){

                // i) unpack current post-decision state and today's (c,d)
                // states_pd[t, i, :] = [a_pd, p_pd, n_pd]
                long long a_idx_pd = index::d3(t,i,0,T,N,par->Nstates_pd);
                long long p_idx_pd = index::d3(t,i,1,T,N,par->Nstates_pd);
                long long n_idx_pd = index::d3(t,i,2,T,N,par->Nstates_pd);

                double a_pd = sim->states_pd[a_idx_pd];
                double p_pd = sim->states_pd[p_idx_pd];
                double n_pd = sim->states_pd[n_idx_pd];

                // today's choices
                long long c_idx = index::d2(t,i,T,N);
                double c_today  = sim->c[c_idx];

                long long d_idx_today = index::d2(t,i,T,N);
                double d_today = sim->d1[d_idx_today]; // 1D durable

                double RHS = 0.0; // E_t[u_c(c_{t+1}, d_{t+1})]

                // ii) integrate over (xi, psi) shocks
                for(long long i_xi = 0; i_xi < par->Nxi; i_xi++){

                    double xi = par->xi[i_xi];
                    double p_xi = par->xi_w[i_xi];
                    double p_plus = std::pow(p_pd, par->eta) * xi;  // next persistent state

                    for(long long i_psi = 0; i_psi < par->Npsi; i_psi++){

                        double psi = par->psi[i_psi];
                        double p_psi = par->psi_w[i_psi];

                        // iii) form next-period cash-on-hand and pre-adjustment stocks
                        double income = par->kappa[t] * p_plus * psi;
                        double m_plus = par->R * a_pd + income;

                        double n_plus = (1.0 - par->delta[0]) * n_pd;
                        double x_adj1 = m_plus + (1.0 - par->nu) * n_plus;

                        // iv) next-period values (t+1) and choice probabilities

                        // value_keep = interp_3d(sol_v_keep[t+1], p_plus, n_plus, m_plus)
                        long long vkeep_base = index::d4(
                            t+1, 0, 0, 0,
                            T, vfi->Np, vfi->Nn, vfi->Nm
                        );
                        double value_keep = linear_interp::interp_3d(
                            vfi->p_grid, vfi->n_grid, vfi->m_grid,
                            vfi->Np, vfi->Nn, vfi->Nm,
                            &vfi->sol_v_keep[vkeep_base],
                            p_plus, n_plus, m_plus
                        );

                        // value_adj1 = interp_2d(sol_v_adj[t+1], p_plus, x_adj1)
                        long long vadj_base = index::d3(
                            t+1, 0, 0,
                            T, vfi->Np, vfi->Nx
                        );
                        double value_adj1 = linear_interp::interp_2d(
                            vfi->p_grid, vfi->x_grid,
                            vfi->Np, vfi->Nx,
                            &vfi->sol_v_adj[vadj_base],
                            p_plus, x_adj1
                        );

                        // softmax_1D over [value_keep, value_adj1] with sigma_eps
                        double sigma = par->sigma_eps;
                        double v0 = value_keep;
                        double v1 = value_adj1;
                        double vmax = (v0 > v1) ? v0 : v1;

                        double e0 = std::exp((v0 - vmax)/sigma);
                        double e1 = std::exp((v1 - vmax)/sigma);
                        double denom = e0 + e1;

                        double prob_keep = e0 / denom; // choice_prob[0]
                        double prob_adj1 = e1 / denom; // choice_prob[1]

                        // v) expected marginal utility components at t+1

                        // ----- keep branch -----
                        double sav_share_keep = linear_interp::interp_3d(
                            vfi->p_grid, vfi->n_grid, vfi->m_grid,
                            vfi->Np, vfi->Nn, vfi->Nm,
                            &vfi->sol_sav_share_keep[vkeep_base],
                            p_plus, n_plus, m_plus
                        );
                        // clamp to [0, 0.9999]
                        if(sav_share_keep < 0.0) sav_share_keep = 0.0;
                        if(sav_share_keep > 0.9999) sav_share_keep = 0.9999;

                        double c_plus_keep = (1.0 - sav_share_keep) * m_plus;
                        double d_plus_keep = n_plus;

                        double uc_plus_keep = utility::marg_u_c_func_1D(
                            c_plus_keep, d_plus_keep, par
                        ) * prob_keep;

                        // ----- adjust-1 branch -----
                        double exp_share_1 = linear_interp::interp_2d(
                            vfi->p_grid, vfi->x_grid,
                            vfi->Np, vfi->Nx,
                            &vfi->sol_exp_share_adj[vadj_base],
                            p_plus, x_adj1
                        );
                        if(exp_share_1 < 1e-8) exp_share_1 = 1e-8;
                        if(exp_share_1 > 1.0) exp_share_1 = 1.0;

                        double c_share_1 = linear_interp::interp_2d(
                            vfi->p_grid, vfi->x_grid,
                            vfi->Np, vfi->Nx,
                            &vfi->sol_c_share_adj[vadj_base],
                            p_plus, x_adj1
                        );
                        if(c_share_1 < 1e-8) c_share_1 = 1e-8;
                        if(c_share_1 > 1.0) c_share_1 = 1.0;

                        double exp_adj_1   = exp_share_1 * x_adj1;
                        double c_plus_adj1 = c_share_1 * exp_adj_1;
                        double d_plus_adj1 = (1.0 - c_share_1) * exp_adj_1;

                        double uc_plus_adj1 = utility::marg_u_c_func_1D(
                            c_plus_adj1, d_plus_adj1, par
                        ) * prob_adj1;

                        double uc_plus = uc_plus_keep + uc_plus_adj1;

                        // vi) accumulate expectation over shocks
                        RHS += p_xi * p_psi * uc_plus;

                    } // i_psi
                } // i_xi

                // vii) Euler error at (t,i)
                double mu_next = par->beta * par->R * RHS;

                double c_implied = utility::inv_marg_u_c_func_1D(
                    mu_next, d_today, par
                );

                long long ee_idx = index::d2(t,i,T,N);
                sim->euler_error_c[ee_idx] = c_implied / c_today - 1.0;

            } // i
        } // t

    } // parallel
}


////////
// 2D //
////////

void compute_euler_errors_2D_cpp(par_struct* par, sim_struct* sim, vfi_struct* vfi){

    logs::write("log.txt",0,"Computing Euler errors 2D...\n");

    const long long T = par->T;
    const long long N = sim->N;

    #pragma omp parallel num_threads(par->cppthreads)
    {

        for(long long t = 0; t < T-1; t++){

            #pragma omp single
            logs::write("log.txt",1," t = %lld\n",t);

            #pragma omp for
            for(long long i = 0; i < N; i++){

                // i) unpack current post-decision state and today's (c,d)

                // states_pd[t,i,:] = [a_pd, p_pd, n1_pd, n2_pd]
                long long a_pd_idx  = index::d3(t,i,0,T,N,par->Nstates_pd);
                long long p_pd_idx  = index::d3(t,i,1,T,N,par->Nstates_pd);
                long long n1_pd_idx = index::d3(t,i,2,T,N,par->Nstates_pd);
                long long n2_pd_idx = index::d3(t,i,3,T,N,par->Nstates_pd);

                double a_pd  = sim->states_pd[a_pd_idx];
                double p_pd  = sim->states_pd[p_pd_idx];
                double n1_pd = sim->states_pd[n1_pd_idx];
                double n2_pd = sim->states_pd[n2_pd_idx];

                long long c_idx  = index::d2(t,i,T,N);
                long long d1_idx = index::d2(t,i,T,N);
                long long d2_idx = index::d2(t,i,T,N);

                double c_today  = sim->c[c_idx];
                double d1_today = sim->d1[d1_idx];
                double d2_today = sim->d2[d2_idx];

                double RHS = 0.0; // E_t[u_c(c_{t+1}, d_{t+1})]

                // ii) integrate over (xi, psi) shocks
                for(long long j = 0; j < par->Nxi; j++){

                    double xi_next = par->xi[j];
                    double p_xi    = par->xi_w[j];

                    double p_plus = std::pow(p_pd, par->eta) * xi_next;  // next persistent state

                    for(long long k = 0; k < par->Npsi; k++){

                        double psi_next = par->psi[k];
                        double p_psi    = par->psi_w[k];

                        // iii) form next-period cash-on-hand and pre-adjustment stocks
                        double income = par->kappa[t+1] * p_plus * psi_next;
                        double m_plus = par->R * a_pd + income;

                        double n1_plus = (1.0 - par->delta[0]) * n1_pd;
                        double n2_plus = (1.0 - par->delta[1]) * n2_pd;

                        // x for each menu at t+1
                        double x_adj1 = m_plus + (1.0 - par->nu) * n1_plus;
                        double x_adj2 = m_plus + (1.0 - par->nu) * n2_plus;
                        double x_adj12 = m_plus + (1.0 - par->nu) * (n1_plus + n2_plus);

                        // iv) next-period values (t+1) and choice probabilities

                        // keep: V_{t+1}(p_plus, n1_plus, n2_plus, m_plus)
                        long long base_keep = index::d5(
                            t+1, 0, 0, 0, 0,
                            T, vfi->Np, vfi->Nn, vfi->Nn, vfi->Nm
                        );
                        double value_keep = linear_interp::interp_4d(
                            vfi->p_grid, vfi->n_grid, vfi->n_grid, vfi->m_grid,
                            vfi->Np, vfi->Nn, vfi->Nn, vfi->Nm,
                            &vfi->sol_v_keep[base_keep],
                            p_plus, n1_plus, n2_plus, m_plus
                        );

                        // adj1: V_{t+1}(p_plus, n2_plus, x_adj1)
                        long long base_adj1 = index::d4(
                            t+1, 0, 0, 0,
                            T, vfi->Np, vfi->Nn, vfi->Nx
                        );
                        double value_adj1 = linear_interp::interp_3d(
                            vfi->p_grid, vfi->n_grid, vfi->x_grid,
                            vfi->Np, vfi->Nn, vfi->Nx,
                            &vfi->sol_v_adj1[base_adj1],
                            p_plus, n2_plus, x_adj1
                        );

                        // adj2: V_{t+1}(p_plus, n1_plus, x_adj2)
                        long long base_adj2 = index::d4(
                            t+1, 0, 0, 0,
                            T, vfi->Np, vfi->Nn, vfi->Nx
                        );
                        double value_adj2 = linear_interp::interp_3d(
                            vfi->p_grid, vfi->n_grid, vfi->x_grid,
                            vfi->Np, vfi->Nn, vfi->Nx,
                            &vfi->sol_v_adj2[base_adj2],
                            p_plus, n1_plus, x_adj2
                        );

                        // adj12: V_{t+1}(p_plus, x_adj12)
                        long long base_adj12 = index::d3(
                            t+1, 0, 0,
                            T, vfi->Np, vfi->Nx
                        );
                        double value_adj12 = linear_interp::interp_2d(
                            vfi->p_grid, vfi->x_grid,
                            vfi->Np, vfi->Nx,
                            &vfi->sol_v_adj12[base_adj12],
                            p_plus, x_adj12
                        );

                        // order matches simulate_2d: [keep, adj2, adj1, adj12]
                        double v0 = value_keep;
                        double v1 = value_adj2;
                        double v2 = value_adj1;
                        double v3 = value_adj12;

                        // softmax_1D(value_next, sigma_eps)
                        double sigma = par->sigma_eps;
                        double vmax  = v0;
                        if(v1 > vmax) vmax = v1;
                        if(v2 > vmax) vmax = v2;
                        if(v3 > vmax) vmax = v3;

                        double e0 = std::exp((v0 - vmax) / sigma);
                        double e1 = std::exp((v1 - vmax) / sigma);
                        double e2 = std::exp((v2 - vmax) / sigma);
                        double e3 = std::exp((v3 - vmax) / sigma);
                        double denom = e0 + e1 + e2 + e3;

                        double prob_keep  = e0 / denom; // choice_prob[0]
                        double prob_adj2  = e1 / denom; // choice_prob[1]
                        double prob_adj1  = e2 / denom; // choice_prob[2]
                        double prob_adj12 = e3 / denom; // choice_prob[3]

                        // v) expected marginal utility components at t+1

                        // ----- keep -----
                        double sav_share_keep = linear_interp::interp_4d(
                            vfi->p_grid, vfi->n_grid, vfi->n_grid, vfi->m_grid,
                            vfi->Np, vfi->Nn, vfi->Nn, vfi->Nm,
                            &vfi->sol_sav_share_keep[base_keep],
                            p_plus, n1_plus, n2_plus, m_plus
                        );
                        if(sav_share_keep < 0.0)    sav_share_keep = 0.0;
                        if(sav_share_keep > 0.9999) sav_share_keep = 0.9999;

                        double c_plus_keep = (1.0 - sav_share_keep) * m_plus;
                        double d1_plus_keep = n1_plus;
                        double d2_plus_keep = n2_plus;

                        double uc_plus_keep = prob_keep *
                            utility::marg_u_c_func_2D(c_plus_keep, d1_plus_keep, d2_plus_keep, par);

                        // ----- adjust 1 -----
                        double exp_share_1 = linear_interp::interp_3d(
                            vfi->p_grid, vfi->n_grid, vfi->x_grid,
                            vfi->Np, vfi->Nn, vfi->Nx,
                            &vfi->sol_exp_share_adj1[base_adj1],
                            p_plus, n2_plus, x_adj1
                        );
                        if(exp_share_1 < 1e-5) exp_share_1 = 1e-5;
                        if(exp_share_1 > 1.0)  exp_share_1 = 1.0;

                        double c_share_1 = linear_interp::interp_3d(
                            vfi->p_grid, vfi->n_grid, vfi->x_grid,
                            vfi->Np, vfi->Nn, vfi->Nx,
                            &vfi->sol_c_share_adj1[base_adj1],
                            p_plus, n2_plus, x_adj1
                        );
                        if(c_share_1 < 1e-5) c_share_1 = 1e-5;
                        if(c_share_1 > 1.0)  c_share_1 = 1.0;

                        double exp_adj_1   = exp_share_1 * x_adj1;
                        double c_plus_adj1 = c_share_1 * exp_adj_1;
                        double d1_plus_adj1 = (1.0 - c_share_1) * exp_adj_1;
                        double d2_plus_adj1 = n2_plus;

                        double uc_plus_adj1 = prob_adj1 *
                            utility::marg_u_c_func_2D(c_plus_adj1, d1_plus_adj1, d2_plus_adj1, par);

                        // ----- adjust 2 -----
                        double exp_share_2 = linear_interp::interp_3d(
                            vfi->p_grid, vfi->n_grid, vfi->x_grid,
                            vfi->Np, vfi->Nn, vfi->Nx,
                            &vfi->sol_exp_share_adj2[base_adj2],
                            p_plus, n1_plus, x_adj2
                        );
                        if(exp_share_2 < 1e-5) exp_share_2 = 1e-5;
                        if(exp_share_2 > 1.0)  exp_share_2 = 1.0;

                        double c_share_2 = linear_interp::interp_3d(
                            vfi->p_grid, vfi->n_grid, vfi->x_grid,
                            vfi->Np, vfi->Nn, vfi->Nx,
                            &vfi->sol_c_share_adj2[base_adj2],
                            p_plus, n1_plus, x_adj2
                        );
                        if(c_share_2 < 1e-5) c_share_2 = 1e-5;
                        if(c_share_2 > 1.0)  c_share_2 = 1.0;

                        double exp_adj_2   = exp_share_2 * x_adj2;
                        double c_plus_adj2 = c_share_2 * exp_adj_2;
                        double d1_plus_adj2 = n1_plus;
                        double d2_plus_adj2 = (1.0 - c_share_2) * exp_adj_2;

                        double uc_plus_adj2 = prob_adj2 *
                            utility::marg_u_c_func_2D(c_plus_adj2, d1_plus_adj2, d2_plus_adj2, par);

                        // ----- adjust 12 -----
                        double exp_share_12 = linear_interp::interp_2d(
                            vfi->p_grid, vfi->x_grid,
                            vfi->Np, vfi->Nx,
                            &vfi->sol_exp_share_adj12[base_adj12],
                            p_plus, x_adj12
                        );
                        if(exp_share_12 < 1e-5) exp_share_12 = 1e-5;
                        if(exp_share_12 > 1.0)  exp_share_12 = 1.0;

                        double c_share_12 = linear_interp::interp_2d(
                            vfi->p_grid, vfi->x_grid,
                            vfi->Np, vfi->Nx,
                            &vfi->sol_c_share_adj12[base_adj12],
                            p_plus, x_adj12
                        );
                        if(c_share_12 < 1e-5) c_share_12 = 1e-5;
                        if(c_share_12 > 1.0)  c_share_12 = 1.0;

                        double d1_share_12 = linear_interp::interp_2d(
                            vfi->p_grid, vfi->x_grid,
                            vfi->Np, vfi->Nx,
                            &vfi->sol_d1_share_adj12[base_adj12],
                            p_plus, x_adj12
                        );
                        if(d1_share_12 < 0.0) d1_share_12 = 0.0;
                        if(d1_share_12 > 1.0) d1_share_12 = 1.0;

                        double exp_adj_12   = exp_share_12 * x_adj12;
                        double c_plus_adj12 = c_share_12 * exp_adj_12;

                        double dur_exp_12    = (1.0 - c_share_12) * exp_adj_12;
                        double d1_plus_adj12 = d1_share_12 * dur_exp_12;
                        double d2_plus_adj12 = (1.0 - d1_share_12) * dur_exp_12;

                        double uc_plus_adj12 = prob_adj12 *
                            utility::marg_u_c_func_2D(c_plus_adj12, d1_plus_adj12, d2_plus_adj12, par);

                        // total uc_plus
                        double uc_plus = uc_plus_keep
                                       + uc_plus_adj1
                                       + uc_plus_adj2
                                       + uc_plus_adj12;

                        // vi) accumulate expectation over shocks
                        RHS += p_xi * p_psi * uc_plus;

                    } // psi
                } // xi

                // vii) Euler error at (t,i)
                double mu_next = par->beta * par->R * RHS;

                double c_implied = utility::inv_marg_u_c_func_2D(
                    mu_next, d1_today, d2_today, par
                );

                long long ee_idx = index::d2(t,i,T,N);
                sim->euler_error_c[ee_idx] = c_implied / c_today - 1.0;

            } // i
        } // t

    } // parallel
}
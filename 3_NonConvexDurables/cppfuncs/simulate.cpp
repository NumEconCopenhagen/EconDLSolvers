
#ifndef MAIN
#define SIMULATE
#include "header.cpp"
#endif

////////
// 1D //
////////

void simulate_1D_cpp(par_struct* par, vfi_struct* vfi, sim_struct* sim){

    logs::write("log.txt",0,"Simulating 1D model...\n");

    const double omega = par->omega[0];
    const double rho = par->rho;
    const double d_ubar= par->d_ubar[0];

    long long N = sim->N;
    long long T = par->T;
    long long NDC = par->NDC;

    #pragma omp parallel num_threads(par->cppthreads)
    {
        for(long long t = 0; t < T; t++){

            logs::write("log.txt",1," t = %lld\n",t);

            #pragma omp for
            for(long long i = 0; i < N; i++){

                // a. current states
                long long m_idx = index::d3(t,i,0,T,N,par->Nstates);
                long long p_idx = index::d3(t,i,1,T,N,par->Nstates);
                long long n_idx = index::d3(t,i,2,T,N,par->Nstates);

                double m = sim->states[m_idx];
                double p = sim->states[p_idx];
                double n = sim->states[n_idx];

                // b. values
                
                // value_keep(p,n,m)
                long long vkeep_base = index::d4(t,0,0,0,T,vfi->Np,vfi->Nn,vfi->Nm);
                
                double value_keep = linear_interp::interp_3d(
                    vfi->p_grid, vfi->n_grid, vfi->m_grid,
                    vfi->Np, vfi->Nn, vfi->Nm,
                    &vfi->sol_v_keep[vkeep_base],
                    p,n,m
                );
                
                // value_adj(p,x)
                double x_disc = m + (1.0 - par->nu) * n;
                long long vadj_base = index::d3(t,0,0,T,vfi->Np,vfi->Nx);
                double value_adj = linear_interp::interp_2d(
                    vfi->p_grid, vfi->x_grid,
                    vfi->Np, vfi->Nx,
                    &vfi->sol_v_adj[vadj_base],
                    p,x_disc
                );

                // c. taste shocks
                long long ts_keep_idx = index::d3(t,i,0,T,N,par->NDC);
                long long ts_adj_idx  = index::d3(t,i,1,T,N,par->NDC);
                double taste_keep = sim->taste_shocks[ts_keep_idx];
                double taste_adj  = sim->taste_shocks[ts_adj_idx];                
                value_keep += taste_keep;
                value_adj  += taste_adj;

                // d. discrete choice
                long long dc_idx = index::d2(t,i,T,N);
                int dc = (value_adj > value_keep) ? 1 : 0;
                sim->DC[dc_idx] = dc;

                // e. continuous choice
                long long c_idx  = index::d2(t,i,T,N);
                long long d1_idx = index::d2(t,i,T,N);

                double x; // total "expenditure base"
                double sav_share; // saving share
                double c_t; // consumption
                double d1_t; // durable stock

                if(dc == 0){

                    // keep
                    x = m;

                    double sav_share_keep = linear_interp::interp_3d(
                        vfi->p_grid, vfi->n_grid, vfi->m_grid,
                        vfi->Np, vfi->Nn, vfi->Nm,
                        &vfi->sol_sav_share_keep[vkeep_base],
                        p, n, m
                    );
                    if(sav_share_keep < 0.0) sav_share_keep = 0.0;
                    if(sav_share_keep > 0.9999) sav_share_keep = 0.9999;

                    sav_share = sav_share_keep;

                    c_t  = (1.0 - sav_share) * x;
                    d1_t = n;

                } else {

                    // adjust
                    x = m + (1.0-par->nu)*n;

                    double exp_share = linear_interp::interp_2d(
                        vfi->p_grid, vfi->x_grid,
                        vfi->Np, vfi->Nx,
                        &vfi->sol_exp_share_adj[vadj_base],
                        p,x
                    );
                    if(exp_share < 1e-8) exp_share = 1e-8;
                    if(exp_share > 1.0)  exp_share = 1.0;

                    sav_share = 1.0-exp_share;

                    double c_share = linear_interp::interp_2d(
                        vfi->p_grid, vfi->x_grid,
                        vfi->Np, vfi->Nx,
                        &vfi->sol_c_share_adj[vadj_base],
                        p,x
                    );
                    if(c_share < 1e-8) c_share = 1e-8;
                    if(c_share > 1.0)  c_share = 1.0;

                    double exp_adjustment = exp_share * x;
                    c_t  = c_share * exp_adjustment;
                    d1_t = (1.0 - c_share) * exp_adjustment;
                }

                sim->c[c_idx] = c_t;
                sim->d1[d1_idx] = d1_t;

                // f. reward
                long long rew_idx = index::d3(t,i,dc,T,N,NDC);

                double c_util = std::pow(c_t, 1.0-omega);
                double d_util = std::pow(d1_t + d_ubar, omega);
                double cobb_d = c_util * d_util;

                double u_cd;
                if(std::abs(1.0-rho) < 1e-10){
                    u_cd = std::log(cobb_d);
                } else {
                    u_cd = std::pow(cobb_d, 1.0-rho) / (1.0-rho);
                }

                double extra_taste = taste_keep * (dc == 0) + taste_adj * (dc == 1);
                sim->reward[rew_idx] = u_cd + extra_taste;

                // g. post-decision states
                long long a_idx_pd = index::d3(t,i,0,T,N,par->Nstates_pd);
                long long p_pd_idx = index::d3(t,i,1,T,N,par->Nstates_pd);
                long long n_pd_idx = index::d3(t,i,2,T,N,par->Nstates_pd);

                double a_t = sav_share * x;

                sim->states_pd[a_idx_pd] = a_t;
                sim->states_pd[p_pd_idx] = p;     // p_pd[t,i] = p[t,i]
                sim->states_pd[n_pd_idx] = d1_t;  // n_pd[t,i] = d1[t,i]

                // h. next period states
                if(t < T-1){

                    // shocks at t+1
                    long long xi_idx_next  = index::d3(t+1,i,0,T,N,par->Nshocks);
                    long long psi_idx_next = index::d3(t+1,i,1,T,N,par->Nshocks);
                    double xi_next = sim->shocks[xi_idx_next];
                    double psi_next = sim->shocks[psi_idx_next];

                    // i. permanent income: p[t+1,i] = (p[t,i]^eta) * xi[t+1,i]
                    long long p_next_idx = index::d3(t+1,i,1,T,N,par->Nstates);
                    double p_plus = std::pow(p, par->eta) * xi_next;
                    sim->states[p_next_idx] = p_plus;

                    // ii. income: y = kappa[t]*p[t+1,i]*psi[t+1,i]
                    double y = par->kappa[t] * p_plus * psi_next;

                    // iii. cash-on-hand: m[t+1,i] = R*a[t,i] + y
                    long long m_next_idx = index::d3(t+1,i,0,T,N,par->Nstates);
                    sim->states[m_next_idx] = par->R * a_t + y;

                    // iv. durable stock: n[t+1,i] = d1[t,i]*(1-delta)
                    long long n_next_idx = index::d3(t+1,i,2,T,N,par->Nstates);
                    sim->states[n_next_idx] = d1_t * (1.0 - par->delta[0]);
                }

            } // i
        } // t
    } // parallel

} // simulate_1D_cpp

////////
// 2D //
////////

void simulate_2D_cpp(par_struct* par, vfi_struct* vfi, sim_struct* sim){

    logs::write("log.txt",0,"Simulating 2D model...\n");

    const long long N = sim->N;
    const long long T = par->T;

    const double omega1  = par->omega[0];
    const double omega2  = par->omega[1];
    const double rho     = par->rho;
    const double d_ubar1 = par->d_ubar[0];
    const double d_ubar2 = par->d_ubar[1];

    #pragma omp parallel num_threads(par->cppthreads)
    {
        for(long long t = 0; t < T; t++){

            logs::write("log.txt",1," t = %lld\n",t);

            #pragma omp for
            for(long long i = 0; i < N; i++){

                // a. unpack states
                long long m_idx  = index::d3(t,i,0,T,N,par->Nstates);
                long long p_idx  = index::d3(t,i,1,T,N,par->Nstates);
                long long n1_idx = index::d3(t,i,2,T,N,par->Nstates);
                long long n2_idx = index::d3(t,i,3,T,N,par->Nstates);

                double m  = sim->states[m_idx];
                double p  = sim->states[p_idx];
                double n1 = sim->states[n1_idx];
                double n2 = sim->states[n2_idx];

                // b. values

                long long base_keep = index::d5(
                    t, 0, 0, 0, 0,
                    T, vfi->Np, vfi->Nn, vfi->Nn, vfi->Nm
                );
                long long base_adj1 = index::d4(
                    t, 0, 0, 0,
                    T, vfi->Np, vfi->Nn, vfi->Nx
                );
                long long base_adj2 = index::d4(
                    t, 0, 0, 0,
                    T, vfi->Np, vfi->Nn, vfi->Nx
                );
                long long base_adj12 = index::d3(
                    t, 0, 0,
                    T, vfi->Np, vfi->Nx
                );

                // keep: V(p, n1, n2, m)
                double value_keep = linear_interp::interp_4d(
                    vfi->p_grid, vfi->n_grid, vfi->n_grid, vfi->m_grid,
                    vfi->Np, vfi->Nn, vfi->Nn, vfi->Nm,
                    &vfi->sol_v_keep[base_keep],
                    p, n1, n2, m
                );

                // adjust 1: x_adj1 = m + (1 - nu) * n1
                double x_adj1 = m + (1.0 - par->nu) * n1;
                double value_adj1 = linear_interp::interp_3d(
                    vfi->p_grid, vfi->n_grid, vfi->x_grid,
                    vfi->Np, vfi->Nn, vfi->Nx,
                    &vfi->sol_v_adj1[base_adj1],
                    p, n2, x_adj1
                );

                // adjust 2: x_adj2 = m + (1 - nu) * n2
                double x_adj2 = m + (1.0 - par->nu) * n2;
                double value_adj2 = linear_interp::interp_3d(
                    vfi->p_grid, vfi->n_grid, vfi->x_grid,
                    vfi->Np, vfi->Nn, vfi->Nx,
                    &vfi->sol_v_adj2[base_adj2],
                    p, n1, x_adj2
                );

                // adjust 12: x_adj12 = m + (1 - nu)*(n1 + n2)
                double x_adj12 = m + (1.0 - par->nu) * (n1 + n2);
                double value_adj12 = linear_interp::interp_2d(
                    vfi->p_grid, vfi->x_grid,
                    vfi->Np, vfi->Nx,
                    &vfi->sol_v_adj12[base_adj12],
                    p, x_adj12
                );

                // c. add taste shocks
                long long ts_keep_idx = index::d3(t,i,0,T,N,par->NDC);
                long long ts_adj2_idx = index::d3(t,i,1,T,N,par->NDC);
                long long ts_adj1_idx = index::d3(t,i,2,T,N,par->NDC);
                long long ts_adj12_idx = index::d3(t,i,3,T,N,par->NDC);

                double taste_keep = sim->taste_shocks[ts_keep_idx];
                double taste_adj2 = sim->taste_shocks[ts_adj2_idx];
                double taste_adj1 = sim->taste_shocks[ts_adj1_idx];
                double taste_adj12 = sim->taste_shocks[ts_adj12_idx];

                value_keep += taste_keep;
                value_adj1 += taste_adj1;
                value_adj2 += taste_adj2;
                value_adj12 += taste_adj12;

                // d. discrete choice
                double v0 = value_keep;
                double v1 = value_adj2;
                double v2 = value_adj1;
                double v3 = value_adj12;

                int dc = 0;
                double vmax = v0;
                if(v1 > vmax){ vmax = v1; dc = 1; }
                if(v2 > vmax){ vmax = v2; dc = 2; }
                if(v3 > vmax){ vmax = v3; dc = 3; }

                long long dc_idx = index::d2(t,i,T,N);
                sim->DC[dc_idx] = dc;

                // e. continuous choice
                long long c_idx  = index::d2(t,i,T,N);
                long long d1_idx = index::d2(t,i,T,N);
                long long d2_idx = index::d2(t,i,T,N);

                double x;
                double exp_share;
                double c_t, d1_t, d2_t;

                if(dc == 0){

                    // keep
                    x = m;

                    double sav_share = linear_interp::interp_4d(
                        vfi->p_grid, vfi->n_grid, vfi->n_grid, vfi->m_grid,
                        vfi->Np, vfi->Nn, vfi->Nn, vfi->Nm,
                        &vfi->sol_sav_share_keep[base_keep],
                        p, n1, n2, m
                    );
                    if(sav_share < 0.0) sav_share = 0.0;
                    if(sav_share > 0.9999) sav_share = 0.9999;

                    exp_share = 1.0 - sav_share;

                    c_t  = (1.0 - sav_share) * x;
                    d1_t = n1;
                    d2_t = n2;

                } else if(dc == 2){

                    // adjust durable 1
                    x = x_adj1;

                    exp_share = linear_interp::interp_3d(
                        vfi->p_grid, vfi->n_grid, vfi->x_grid,
                        vfi->Np, vfi->Nn, vfi->Nx,
                        &vfi->sol_exp_share_adj1[base_adj1],
                        p, n2, x
                    );
                    if(exp_share < 1e-8) exp_share = 1e-8;
                    if(exp_share > 1.0) exp_share = 1.0;

                    double c_share = linear_interp::interp_3d(
                        vfi->p_grid, vfi->n_grid, vfi->x_grid,
                        vfi->Np, vfi->Nn, vfi->Nx,
                        &vfi->sol_c_share_adj1[base_adj1],
                        p, n2, x
                    );
                    if(c_share < 1e-8) c_share = 1e-8;
                    if(c_share > 1.0) c_share = 1.0;

                    double exp_adjustment = exp_share * x;

                    c_t  = c_share*exp_adjustment;
                    d1_t = (1.0-c_share)*exp_adjustment;
                    d2_t = n2;

                } else if(dc == 1){

                    // adjust durable 2
                    x = x_adj2;

                    exp_share = linear_interp::interp_3d(
                        vfi->p_grid, vfi->n_grid, vfi->x_grid,
                        vfi->Np, vfi->Nn, vfi->Nx,
                        &vfi->sol_exp_share_adj2[base_adj2],
                        p, n1, x
                    );
                    if(exp_share < 1e-8) exp_share = 1e-8;
                    if(exp_share > 1.0)  exp_share = 1.0;

                    double c_share = linear_interp::interp_3d(
                        vfi->p_grid, vfi->n_grid, vfi->x_grid,
                        vfi->Np, vfi->Nn, vfi->Nx,
                        &vfi->sol_c_share_adj2[base_adj2],
                        p, n1, x
                    );
                    if(c_share < 1e-8) c_share = 1e-8;
                    if(c_share > 1.0) c_share = 1.0;

                    double exp_adjustment = exp_share*x;

                    c_t  = c_share*exp_adjustment;
                    d1_t = n1;
                    d2_t = (1.0-c_share)*exp_adjustment;

                } else {

                    // adjust both durables (dc == 3)
                    x = x_adj12;

                    exp_share = linear_interp::interp_2d(
                        vfi->p_grid, vfi->x_grid,
                        vfi->Np, vfi->Nx,
                        &vfi->sol_exp_share_adj12[base_adj12],
                        p, x
                    );
                    if(exp_share < 1e-8) exp_share = 1e-8;
                    if(exp_share > 1.0) exp_share = 1.0;

                    double c_share = linear_interp::interp_2d(
                        vfi->p_grid, vfi->x_grid,
                        vfi->Np, vfi->Nx,
                        &vfi->sol_c_share_adj12[base_adj12],
                        p, x
                    );
                    if(c_share < 1e-8) c_share = 1e-8;
                    if(c_share > 1.0) c_share = 1.0;

                    double d1_share = linear_interp::interp_2d(
                        vfi->p_grid, vfi->x_grid,
                        vfi->Np, vfi->Nx,
                        &vfi->sol_d1_share_adj12[base_adj12],
                        p, x
                    );
                    if(d1_share < 0.0) d1_share = 0.0;
                    if(d1_share > 1.0) d1_share = 1.0;

                    double exp_adjustment = exp_share * x;
                    c_t = c_share * exp_adjustment;

                    double durable_exp = (1.0 - c_share) * exp_adjustment;
                    d1_t = d1_share * durable_exp;
                    d2_t = (1.0 - d1_share) * durable_exp;
                }

                sim->c[c_idx] = c_t;
                sim->d1[d1_idx] = d1_t;
                sim->d2[d2_idx] = d2_t;

                // f. reward
                long long rew_idx = index::d2(t,i,T,N);

                double c_util  = std::pow(c_t, 1.0 - omega1 - omega2);
                double d1_util = std::pow(d1_t + d_ubar1, omega1);
                double d2_util = std::pow(d2_t + d_ubar2, omega2);
                double cobb_d  = c_util * d1_util * d2_util;

                double u_cd = std::pow(cobb_d, 1.0 - rho) / (1.0 - rho);

                long long ts_sel_idx = index::d3(t,i,dc,T,N,par->NDC);
                double selected_shock = sim->taste_shocks[ts_sel_idx];

                sim->reward[rew_idx] = u_cd + selected_shock;

                // g. post-decision states
                long long a_pd_idx = index::d3(t,i,0,T,N,par->Nstates_pd);
                long long p_pd_idx = index::d3(t,i,1,T,N,par->Nstates_pd);
                long long n1_pd_idx = index::d3(t,i,2,T,N,par->Nstates_pd);
                long long n2_pd_idx = index::d3(t,i,3,T,N,par->Nstates_pd);

                double a_t = x * (1.0 - exp_share);

                sim->states_pd[a_pd_idx] = a_t;
                sim->states_pd[p_pd_idx] = p;
                sim->states_pd[n1_pd_idx] = d1_t;
                sim->states_pd[n2_pd_idx] = d2_t;

                // h. next period states
                if(t < T-1){

                    long long xi_idx_next = index::d3(t+1,i,0,T,N,par->Nshocks);
                    long long psi_idx_next = index::d3(t+1,i,1,T,N,par->Nshocks);
                    double xi_next  = sim->shocks[xi_idx_next];
                    double psi_next = sim->shocks[psi_idx_next];

                    long long p_next_idx = index::d3(t+1,i,1,T,N,par->Nstates);
                    double p_plus = std::pow(p, par->eta) * xi_next;
                    sim->states[p_next_idx] = p_plus;

                    double y = par->kappa[t] * p_plus * psi_next;

                    long long m_next_idx = index::d3(t+1,i,0,T,N,par->Nstates);
                    sim->states[m_next_idx] = par->R * a_t + y;

                    long long n1_next_idx = index::d3(t+1,i,2,T,N,par->Nstates);
                    long long n2_next_idx = index::d3(t+1,i,3,T,N,par->Nstates);

                    sim->states[n1_next_idx] = d1_t * (1.0 - par->delta[0]);
                    sim->states[n2_next_idx] = d2_t * (1.0 - par->delta[1]);

                }

            } // i
        } // t

    } // parallel

} // simulate_2D_cpp
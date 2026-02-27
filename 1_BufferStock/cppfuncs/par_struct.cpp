typedef struct par_struct
{
 int NDC;
 bool full;
 int seed;
 int T;
 int T_retired;
 double beta;
 double bequest;
 double kappa_base;
 double kappa_growth;
 double kappa_growth_decay;
 double kappa_retired;
 double rho_p_base;
 double rho_p_low;
 double rho_p_high;
 double sigma_xi_base;
 double sigma_xi_low;
 double sigma_xi_high;
 int Nxi;
 double sigma_psi_base;
 double sigma_psi_low;
 double sigma_psi_high;
 int Npsi;
 bool MoreShocks;
 double sigma_xi_1;
 double sigma_xi_2;
 double p_xi;
 double sigma_psi_1;
 double sigma_psi_2;
 double p_psi;
 double p_u;
 double u_replacement;
 int Nmc;
 bool do_antithetic_mc;
 double R;
 int Nstates_fixed;
 int Nstates_dynamic;
 int Nstates_dynamic_pd;
 int Nshocks;
 int Noutcomes;
 bool KKT;
 double Euler_error_min_savings;
 double Delta_MPC;
 double mu_m0;
 double sigma_m0;
 double mu_p0;
 double sigma_p0;
 double explore_states_endo_fac;
 double explore_states_fixed_fac_low;
 double explore_states_fixed_fac_high;
 bool init_bias_logodds;
 double w_p;
 bool init_sigmoid_xavier_uniform;
 bool init_sigmoid_xavier_normal;
 bool init_relu_he_uniform;
 bool init_relu_he_normal;
 bool use_sobol;
 int cppthreads;
 double* kappa;
 int Nstates_fixed_pd;
 int Nstates;
 int Nstates_pd;
 int Nactions;
 double* psi_w;
 double* psi;
 double* xi_w;
 double* xi;
 double* u;
 double* mix_psi;
 double* mix_xi;
 double* u_w;
 double* mix_psi_w;
 double* mix_xi_w;
} par_struct;

int get_int_par_struct(par_struct* x, char* name){

 if( strcmp(name,"NDC") == 0 ){ return x->NDC; }
 else if( strcmp(name,"seed") == 0 ){ return x->seed; }
 else if( strcmp(name,"T") == 0 ){ return x->T; }
 else if( strcmp(name,"T_retired") == 0 ){ return x->T_retired; }
 else if( strcmp(name,"Nxi") == 0 ){ return x->Nxi; }
 else if( strcmp(name,"Npsi") == 0 ){ return x->Npsi; }
 else if( strcmp(name,"Nmc") == 0 ){ return x->Nmc; }
 else if( strcmp(name,"Nstates_fixed") == 0 ){ return x->Nstates_fixed; }
 else if( strcmp(name,"Nstates_dynamic") == 0 ){ return x->Nstates_dynamic; }
 else if( strcmp(name,"Nstates_dynamic_pd") == 0 ){ return x->Nstates_dynamic_pd; }
 else if( strcmp(name,"Nshocks") == 0 ){ return x->Nshocks; }
 else if( strcmp(name,"Noutcomes") == 0 ){ return x->Noutcomes; }
 else if( strcmp(name,"cppthreads") == 0 ){ return x->cppthreads; }
 else if( strcmp(name,"Nstates_fixed_pd") == 0 ){ return x->Nstates_fixed_pd; }
 else if( strcmp(name,"Nstates") == 0 ){ return x->Nstates; }
 else if( strcmp(name,"Nstates_pd") == 0 ){ return x->Nstates_pd; }
 else if( strcmp(name,"Nactions") == 0 ){ return x->Nactions; }
 else {return -9999;}

}


bool get_bool_par_struct(par_struct* x, char* name){

 if( strcmp(name,"full") == 0 ){ return x->full; }
 else if( strcmp(name,"MoreShocks") == 0 ){ return x->MoreShocks; }
 else if( strcmp(name,"do_antithetic_mc") == 0 ){ return x->do_antithetic_mc; }
 else if( strcmp(name,"KKT") == 0 ){ return x->KKT; }
 else if( strcmp(name,"init_bias_logodds") == 0 ){ return x->init_bias_logodds; }
 else if( strcmp(name,"init_sigmoid_xavier_uniform") == 0 ){ return x->init_sigmoid_xavier_uniform; }
 else if( strcmp(name,"init_sigmoid_xavier_normal") == 0 ){ return x->init_sigmoid_xavier_normal; }
 else if( strcmp(name,"init_relu_he_uniform") == 0 ){ return x->init_relu_he_uniform; }
 else if( strcmp(name,"init_relu_he_normal") == 0 ){ return x->init_relu_he_normal; }
 else if( strcmp(name,"use_sobol") == 0 ){ return x->use_sobol; }
 else {return false;}

}


double get_double_par_struct(par_struct* x, char* name){

 if( strcmp(name,"beta") == 0 ){ return x->beta; }
 else if( strcmp(name,"bequest") == 0 ){ return x->bequest; }
 else if( strcmp(name,"kappa_base") == 0 ){ return x->kappa_base; }
 else if( strcmp(name,"kappa_growth") == 0 ){ return x->kappa_growth; }
 else if( strcmp(name,"kappa_growth_decay") == 0 ){ return x->kappa_growth_decay; }
 else if( strcmp(name,"kappa_retired") == 0 ){ return x->kappa_retired; }
 else if( strcmp(name,"rho_p_base") == 0 ){ return x->rho_p_base; }
 else if( strcmp(name,"rho_p_low") == 0 ){ return x->rho_p_low; }
 else if( strcmp(name,"rho_p_high") == 0 ){ return x->rho_p_high; }
 else if( strcmp(name,"sigma_xi_base") == 0 ){ return x->sigma_xi_base; }
 else if( strcmp(name,"sigma_xi_low") == 0 ){ return x->sigma_xi_low; }
 else if( strcmp(name,"sigma_xi_high") == 0 ){ return x->sigma_xi_high; }
 else if( strcmp(name,"sigma_psi_base") == 0 ){ return x->sigma_psi_base; }
 else if( strcmp(name,"sigma_psi_low") == 0 ){ return x->sigma_psi_low; }
 else if( strcmp(name,"sigma_psi_high") == 0 ){ return x->sigma_psi_high; }
 else if( strcmp(name,"sigma_xi_1") == 0 ){ return x->sigma_xi_1; }
 else if( strcmp(name,"sigma_xi_2") == 0 ){ return x->sigma_xi_2; }
 else if( strcmp(name,"p_xi") == 0 ){ return x->p_xi; }
 else if( strcmp(name,"sigma_psi_1") == 0 ){ return x->sigma_psi_1; }
 else if( strcmp(name,"sigma_psi_2") == 0 ){ return x->sigma_psi_2; }
 else if( strcmp(name,"p_psi") == 0 ){ return x->p_psi; }
 else if( strcmp(name,"p_u") == 0 ){ return x->p_u; }
 else if( strcmp(name,"u_replacement") == 0 ){ return x->u_replacement; }
 else if( strcmp(name,"R") == 0 ){ return x->R; }
 else if( strcmp(name,"Euler_error_min_savings") == 0 ){ return x->Euler_error_min_savings; }
 else if( strcmp(name,"Delta_MPC") == 0 ){ return x->Delta_MPC; }
 else if( strcmp(name,"mu_m0") == 0 ){ return x->mu_m0; }
 else if( strcmp(name,"sigma_m0") == 0 ){ return x->sigma_m0; }
 else if( strcmp(name,"mu_p0") == 0 ){ return x->mu_p0; }
 else if( strcmp(name,"sigma_p0") == 0 ){ return x->sigma_p0; }
 else if( strcmp(name,"explore_states_endo_fac") == 0 ){ return x->explore_states_endo_fac; }
 else if( strcmp(name,"explore_states_fixed_fac_low") == 0 ){ return x->explore_states_fixed_fac_low; }
 else if( strcmp(name,"explore_states_fixed_fac_high") == 0 ){ return x->explore_states_fixed_fac_high; }
 else if( strcmp(name,"w_p") == 0 ){ return x->w_p; }
 else {return NAN;}

}


double* get_double_p_par_struct(par_struct* x, char* name){

 if( strcmp(name,"kappa") == 0 ){ return x->kappa; }
 else if( strcmp(name,"psi_w") == 0 ){ return x->psi_w; }
 else if( strcmp(name,"psi") == 0 ){ return x->psi; }
 else if( strcmp(name,"xi_w") == 0 ){ return x->xi_w; }
 else if( strcmp(name,"xi") == 0 ){ return x->xi; }
 else if( strcmp(name,"u") == 0 ){ return x->u; }
 else if( strcmp(name,"mix_psi") == 0 ){ return x->mix_psi; }
 else if( strcmp(name,"mix_xi") == 0 ){ return x->mix_xi; }
 else if( strcmp(name,"u_w") == 0 ){ return x->u_w; }
 else if( strcmp(name,"mix_psi_w") == 0 ){ return x->mix_psi_w; }
 else if( strcmp(name,"mix_xi_w") == 0 ){ return x->mix_xi_w; }
 else {return NULL;}

}



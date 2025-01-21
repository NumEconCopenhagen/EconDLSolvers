typedef struct par_struct
{
 bool full;
 int seed;
 int T;
 int T_retired;
 double beta;
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
 double R;
 int Nstates_fixed;
 int Nstates_dynamic;
 int Nstates_dynamic_pd;
 int Nshocks;
 int Noutcomes;
 bool KKT;
 int NDC;
 double m_scaler;
 double p_scaler;
 char* policy_predict;
 double Euler_error_min_savings;
 double Delta_MPC;
 double mu_m0;
 double sigma_m0;
 double mu_p0;
 double sigma_p0;
 double explore_states_endo_fac;
 double explore_states_fixed_fac_low;
 double explore_states_fixed_fac_high;
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
 double* scale_vec_states;
 double* scale_vec_states_pd;
} par_struct;

bool get_bool_par_struct(par_struct* x, char* name){

 if( strcmp(name,"full") == 0 ){ return x->full; }
 else if( strcmp(name,"KKT") == 0 ){ return x->KKT; }
 else {return false;}

}


int get_int_par_struct(par_struct* x, char* name){

 if( strcmp(name,"seed") == 0 ){ return x->seed; }
 else if( strcmp(name,"T") == 0 ){ return x->T; }
 else if( strcmp(name,"T_retired") == 0 ){ return x->T_retired; }
 else if( strcmp(name,"Nxi") == 0 ){ return x->Nxi; }
 else if( strcmp(name,"Npsi") == 0 ){ return x->Npsi; }
 else if( strcmp(name,"Nstates_fixed") == 0 ){ return x->Nstates_fixed; }
 else if( strcmp(name,"Nstates_dynamic") == 0 ){ return x->Nstates_dynamic; }
 else if( strcmp(name,"Nstates_dynamic_pd") == 0 ){ return x->Nstates_dynamic_pd; }
 else if( strcmp(name,"Nshocks") == 0 ){ return x->Nshocks; }
 else if( strcmp(name,"Noutcomes") == 0 ){ return x->Noutcomes; }
 else if( strcmp(name,"NDC") == 0 ){ return x->NDC; }
 else if( strcmp(name,"cppthreads") == 0 ){ return x->cppthreads; }
 else if( strcmp(name,"Nstates_fixed_pd") == 0 ){ return x->Nstates_fixed_pd; }
 else if( strcmp(name,"Nstates") == 0 ){ return x->Nstates; }
 else if( strcmp(name,"Nstates_pd") == 0 ){ return x->Nstates_pd; }
 else if( strcmp(name,"Nactions") == 0 ){ return x->Nactions; }
 else {return -9999;}

}


double get_double_par_struct(par_struct* x, char* name){

 if( strcmp(name,"beta") == 0 ){ return x->beta; }
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
 else if( strcmp(name,"R") == 0 ){ return x->R; }
 else if( strcmp(name,"m_scaler") == 0 ){ return x->m_scaler; }
 else if( strcmp(name,"p_scaler") == 0 ){ return x->p_scaler; }
 else if( strcmp(name,"Euler_error_min_savings") == 0 ){ return x->Euler_error_min_savings; }
 else if( strcmp(name,"Delta_MPC") == 0 ){ return x->Delta_MPC; }
 else if( strcmp(name,"mu_m0") == 0 ){ return x->mu_m0; }
 else if( strcmp(name,"sigma_m0") == 0 ){ return x->sigma_m0; }
 else if( strcmp(name,"mu_p0") == 0 ){ return x->mu_p0; }
 else if( strcmp(name,"sigma_p0") == 0 ){ return x->sigma_p0; }
 else if( strcmp(name,"explore_states_endo_fac") == 0 ){ return x->explore_states_endo_fac; }
 else if( strcmp(name,"explore_states_fixed_fac_low") == 0 ){ return x->explore_states_fixed_fac_low; }
 else if( strcmp(name,"explore_states_fixed_fac_high") == 0 ){ return x->explore_states_fixed_fac_high; }
 else {return NAN;}

}


char* get_char_p_par_struct(par_struct* x, char* name){

 if( strcmp(name,"policy_predict") == 0 ){ return x->policy_predict; }
 else {return NULL;}

}


double* get_double_p_par_struct(par_struct* x, char* name){

 if( strcmp(name,"kappa") == 0 ){ return x->kappa; }
 else if( strcmp(name,"psi_w") == 0 ){ return x->psi_w; }
 else if( strcmp(name,"psi") == 0 ){ return x->psi; }
 else if( strcmp(name,"xi_w") == 0 ){ return x->xi_w; }
 else if( strcmp(name,"xi") == 0 ){ return x->xi; }
 else if( strcmp(name,"scale_vec_states") == 0 ){ return x->scale_vec_states; }
 else if( strcmp(name,"scale_vec_states_pd") == 0 ){ return x->scale_vec_states_pd; }
 else {return NULL;}

}



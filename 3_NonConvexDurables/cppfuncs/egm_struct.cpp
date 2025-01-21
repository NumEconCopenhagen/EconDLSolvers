typedef struct egm_struct
{
 int Nm;
 int Na;
 int Np;
 int Nn;
 double m_max;
 double m_min;
 double a_max;
 double n_min;
 double n_max;
 double p_min;
 double p_max;
 int solver;
 double* m_grid;
 double* a_grid;
 double* p_grid;
 double* n_grid;
 double* sol_sav_share_keep;
 double* sol_v_keep;
 double* sol_exp_share_adj;
 double* sol_c_share_adj;
 double* sol_v_adj;
 double* sol_func_evals_keep;
 double* sol_flag_keep;
 double* sol_func_evals_adj;
 double* sol_flag_adj;
 double* sol_w;
 double* transfer_grid;
 int Ntransfer;
} egm_struct;

int get_int_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"Nm") == 0 ){ return x->Nm; }
 else if( strcmp(name,"Na") == 0 ){ return x->Na; }
 else if( strcmp(name,"Np") == 0 ){ return x->Np; }
 else if( strcmp(name,"Nn") == 0 ){ return x->Nn; }
 else if( strcmp(name,"solver") == 0 ){ return x->solver; }
 else if( strcmp(name,"Ntransfer") == 0 ){ return x->Ntransfer; }
 else {return -9999;}

}


double get_double_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"m_max") == 0 ){ return x->m_max; }
 else if( strcmp(name,"m_min") == 0 ){ return x->m_min; }
 else if( strcmp(name,"a_max") == 0 ){ return x->a_max; }
 else if( strcmp(name,"n_min") == 0 ){ return x->n_min; }
 else if( strcmp(name,"n_max") == 0 ){ return x->n_max; }
 else if( strcmp(name,"p_min") == 0 ){ return x->p_min; }
 else if( strcmp(name,"p_max") == 0 ){ return x->p_max; }
 else {return NAN;}

}


double* get_double_p_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"m_grid") == 0 ){ return x->m_grid; }
 else if( strcmp(name,"a_grid") == 0 ){ return x->a_grid; }
 else if( strcmp(name,"p_grid") == 0 ){ return x->p_grid; }
 else if( strcmp(name,"n_grid") == 0 ){ return x->n_grid; }
 else if( strcmp(name,"sol_sav_share_keep") == 0 ){ return x->sol_sav_share_keep; }
 else if( strcmp(name,"sol_v_keep") == 0 ){ return x->sol_v_keep; }
 else if( strcmp(name,"sol_exp_share_adj") == 0 ){ return x->sol_exp_share_adj; }
 else if( strcmp(name,"sol_c_share_adj") == 0 ){ return x->sol_c_share_adj; }
 else if( strcmp(name,"sol_v_adj") == 0 ){ return x->sol_v_adj; }
 else if( strcmp(name,"sol_func_evals_keep") == 0 ){ return x->sol_func_evals_keep; }
 else if( strcmp(name,"sol_flag_keep") == 0 ){ return x->sol_flag_keep; }
 else if( strcmp(name,"sol_func_evals_adj") == 0 ){ return x->sol_func_evals_adj; }
 else if( strcmp(name,"sol_flag_adj") == 0 ){ return x->sol_flag_adj; }
 else if( strcmp(name,"sol_w") == 0 ){ return x->sol_w; }
 else if( strcmp(name,"transfer_grid") == 0 ){ return x->transfer_grid; }
 else {return NULL;}

}



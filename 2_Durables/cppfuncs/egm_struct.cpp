typedef struct egm_struct
{
 int Np;
 int Nm_keep;
 int Nm_pd;
 int Nm;
 int Nn;
 double m_pd_max;
 double m_max;
 double m_min;
 double p_max;
 double p_min;
 double n_max;
 int solver;
 bool pure_vfi;
 double min_action;
 double max_action;
 double* p_grid;
 double* n_grid;
 double* m_pd_grid;
 double* m_keep_grid;
 double* m_grid;
 double* sol_v;
 double* sol_vm;
 double* sol_d1_fac;
 double* sol_d2_fac;
 double* sol_d3_fac;
 double* sol_m_pd_fac;
 int* sol_func_evals;
 int* sol_flag;
 double* sol_c_keep;
 double* sol_w;
 double* sol_q;
 double* transfer_grid;
 int Ntransfer;
} egm_struct;

int get_int_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"Np") == 0 ){ return x->Np; }
 else if( strcmp(name,"Nm_keep") == 0 ){ return x->Nm_keep; }
 else if( strcmp(name,"Nm_pd") == 0 ){ return x->Nm_pd; }
 else if( strcmp(name,"Nm") == 0 ){ return x->Nm; }
 else if( strcmp(name,"Nn") == 0 ){ return x->Nn; }
 else if( strcmp(name,"solver") == 0 ){ return x->solver; }
 else if( strcmp(name,"Ntransfer") == 0 ){ return x->Ntransfer; }
 else {return -9999;}

}


double get_double_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"m_pd_max") == 0 ){ return x->m_pd_max; }
 else if( strcmp(name,"m_max") == 0 ){ return x->m_max; }
 else if( strcmp(name,"m_min") == 0 ){ return x->m_min; }
 else if( strcmp(name,"p_max") == 0 ){ return x->p_max; }
 else if( strcmp(name,"p_min") == 0 ){ return x->p_min; }
 else if( strcmp(name,"n_max") == 0 ){ return x->n_max; }
 else if( strcmp(name,"min_action") == 0 ){ return x->min_action; }
 else if( strcmp(name,"max_action") == 0 ){ return x->max_action; }
 else {return NAN;}

}


bool get_bool_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"pure_vfi") == 0 ){ return x->pure_vfi; }
 else {return false;}

}


double* get_double_p_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"p_grid") == 0 ){ return x->p_grid; }
 else if( strcmp(name,"n_grid") == 0 ){ return x->n_grid; }
 else if( strcmp(name,"m_pd_grid") == 0 ){ return x->m_pd_grid; }
 else if( strcmp(name,"m_keep_grid") == 0 ){ return x->m_keep_grid; }
 else if( strcmp(name,"m_grid") == 0 ){ return x->m_grid; }
 else if( strcmp(name,"sol_v") == 0 ){ return x->sol_v; }
 else if( strcmp(name,"sol_vm") == 0 ){ return x->sol_vm; }
 else if( strcmp(name,"sol_d1_fac") == 0 ){ return x->sol_d1_fac; }
 else if( strcmp(name,"sol_d2_fac") == 0 ){ return x->sol_d2_fac; }
 else if( strcmp(name,"sol_d3_fac") == 0 ){ return x->sol_d3_fac; }
 else if( strcmp(name,"sol_m_pd_fac") == 0 ){ return x->sol_m_pd_fac; }
 else if( strcmp(name,"sol_c_keep") == 0 ){ return x->sol_c_keep; }
 else if( strcmp(name,"sol_w") == 0 ){ return x->sol_w; }
 else if( strcmp(name,"sol_q") == 0 ){ return x->sol_q; }
 else if( strcmp(name,"transfer_grid") == 0 ){ return x->transfer_grid; }
 else {return NULL;}

}


int* get_int_p_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"sol_func_evals") == 0 ){ return x->sol_func_evals; }
 else if( strcmp(name,"sol_flag") == 0 ){ return x->sol_flag; }
 else {return NULL;}

}



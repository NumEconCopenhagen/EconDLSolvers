typedef struct egm_struct
{
 int Nm_pd;
 int Nm;
 int Np;
 int Nsigma_psi;
 int Nsigma_xi;
 int Nrho_p;
 double m_max;
 double p_max;
 double p_min;
 double* m_pd_grid;
 double* m_grid;
 double* p_grid;
 double* sigma_xi_grid;
 double* sigma_psi_grid;
 double* rho_p_grid;
 double* sol_con;
 double* sol_w;
 double* transfer_grid;
 int Ntransfer;
} egm_struct;

int get_int_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"Nm_pd") == 0 ){ return x->Nm_pd; }
 else if( strcmp(name,"Nm") == 0 ){ return x->Nm; }
 else if( strcmp(name,"Np") == 0 ){ return x->Np; }
 else if( strcmp(name,"Nsigma_psi") == 0 ){ return x->Nsigma_psi; }
 else if( strcmp(name,"Nsigma_xi") == 0 ){ return x->Nsigma_xi; }
 else if( strcmp(name,"Nrho_p") == 0 ){ return x->Nrho_p; }
 else if( strcmp(name,"Ntransfer") == 0 ){ return x->Ntransfer; }
 else {return -9999;}

}


double get_double_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"m_max") == 0 ){ return x->m_max; }
 else if( strcmp(name,"p_max") == 0 ){ return x->p_max; }
 else if( strcmp(name,"p_min") == 0 ){ return x->p_min; }
 else {return NAN;}

}


double* get_double_p_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"m_pd_grid") == 0 ){ return x->m_pd_grid; }
 else if( strcmp(name,"m_grid") == 0 ){ return x->m_grid; }
 else if( strcmp(name,"p_grid") == 0 ){ return x->p_grid; }
 else if( strcmp(name,"sigma_xi_grid") == 0 ){ return x->sigma_xi_grid; }
 else if( strcmp(name,"sigma_psi_grid") == 0 ){ return x->sigma_psi_grid; }
 else if( strcmp(name,"rho_p_grid") == 0 ){ return x->rho_p_grid; }
 else if( strcmp(name,"sol_con") == 0 ){ return x->sol_con; }
 else if( strcmp(name,"sol_w") == 0 ){ return x->sol_w; }
 else if( strcmp(name,"transfer_grid") == 0 ){ return x->transfer_grid; }
 else {return NULL;}

}


